from __future__ import annotations

import logging
import queue
import threading
import uuid

from typing import Any, Sequence

from app.ai.ollama_client import OllamaClient
from app.memory_management.manager import MemoryManager


logger = logging.getLogger(__name__)


_MEMORY_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(
    maxsize=24
)

_WORKER_LOCK = threading.Lock()
_QUEUE_LOCK = threading.Lock()

_WORKER_STARTED = False
_PENDING_TRACE_IDS: set[str] = set()


def _start_worker() -> None:
    global _WORKER_STARTED

    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return

        worker = threading.Thread(
            target=_memory_worker,
            name="zara-automatic-memory",
            daemon=True,
        )

        worker.start()
        _WORKER_STARTED = True


def enqueue_automatic_memory_extraction(
    learning_store: Any,
    user_messages: Sequence[str],
    *,
    trace_id: str | None = None,
) -> bool:
    """Queue memory extraction without blocking the chat response."""

    sources = [
        str(message).strip()[:1800]
        for message in user_messages
        if isinstance(message, str) and message.strip()
    ][-5:]

    if not sources:
        return False

    trace = str(trace_id or uuid.uuid4().hex)[:128]

    job = {
        "learning_store": learning_store,
        "user_messages": sources,
        "trace_id": trace,
    }

    with _QUEUE_LOCK:
        if trace in _PENDING_TRACE_IDS:
            return False

        _PENDING_TRACE_IDS.add(trace)

    try:
        _start_worker()
        _MEMORY_QUEUE.put_nowait(job)

    except queue.Full:
        with _QUEUE_LOCK:
            _PENDING_TRACE_IDS.discard(trace)

        logger.warning(
            "Automatic memory queue is full; this turn was skipped."
        )
        return False

    except Exception as exc:
        with _QUEUE_LOCK:
            _PENDING_TRACE_IDS.discard(trace)

        logger.warning(
            "Automatic memory scheduling failed (%s).",
            type(exc).__name__,
        )
        return False

    return True


def _memory_worker() -> None:
    while True:
        job = _MEMORY_QUEUE.get()
        client = None

        trace_id = str(job.get("trace_id") or "")

        try:
            # Dedicated client: never share the live chat stream session.
            client = OllamaClient()

            manager = MemoryManager(
                job["learning_store"]
            )

            messages = list(job["user_messages"])

            candidates = manager.extract_automatic_candidates(
                messages,
                client,
            )

            if candidates:
                processed = manager.save_automatic_candidates(
                    candidates,
                    messages,
                    trace_id=trace_id,
                )

                logger.info(
                    "Automatic memory job completed "
                    "(trace_id=%s, candidates=%s, processed=%s).",
                    trace_id,
                    len(candidates),
                    processed,
                )

        except Exception as exc:
            logger.warning(
                "Automatic memory extraction failed "
                "(trace_id=%s, type=%s).",
                trace_id,
                type(exc).__name__,
            )

        finally:
            if client is not None:
                try:
                    client.session.close()
                except Exception:
                    pass

            job.clear()

            with _QUEUE_LOCK:
                _PENDING_TRACE_IDS.discard(trace_id)

            _MEMORY_QUEUE.task_done()