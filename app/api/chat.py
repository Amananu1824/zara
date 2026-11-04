"""Local HTTP API. Each session has its own ChatEngine and conversation history."""
from __future__ import annotations

import json

from threading import Event, Lock
from fastapi.responses import StreamingResponse
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.chat_engine import ChatEngine
from app.ai.ollama_client import OllamaError
from app.core.prompt_manager import PromptManager
from app.terminal.language import normalize_language_mode
from app.learning_ai.learning_store import LearningStore
from app.core.config import settings

_shared_learning_store: LearningStore | None = None

router = APIRouter()
_sessions: dict[str, ChatEngine] = {}
_sessions_lock = Lock()
_active_streams: dict[str, Event] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    language_mode: str
    system_results: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None
    display_events: list[dict[str, Any]] = Field(
        default_factory=list
    )

class ResetRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=100)


class ModeRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=100)
    language_mode: str


def _get_engine(session_id: str) -> ChatEngine:
    global _shared_learning_store

    with _sessions_lock:
        engine = _sessions.get(session_id)

        if engine is None:
            if _shared_learning_store is None:
                _shared_learning_store = LearningStore()

            engine = ChatEngine(
                prompt_manager=PromptManager(),
                owner_verified=False,
                learning_store=_shared_learning_store,
                allow_database_tools=(
                    settings.api_allow_database_tools
                    or settings.api_allow_database_mutations
                ),
                allow_database_mutations=(
                    settings.api_allow_database_mutations
                ),
            )
            _sessions[session_id] = engine

        return engine

def _sse_event(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be blank.")
    try:
        engine = _get_engine(request.session_id)
        result = engine.reply_with_result(message)
        return ChatResponse(
            reply=result["assistant_message"],
            session_id=request.session_id,
            language_mode=engine.language_mode,
            system_results=result["system_results"],
            trace_id=result["trace_id"],
            display_events=result["display_events"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OllamaError as exc:
        status_code = (
            409
            if exc.code in {
                "turn_already_active",
                "stream_already_active",
            }
            else 502
        )

        raise HTTPException(
            status_code=status_code,
            detail=str(exc),
        ) from exc

@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be blank.",
        )

    engine = _get_engine(request.session_id)
    cancel_event = Event()

    with _sessions_lock:
        active = _active_streams.get(request.session_id)

        if active is not None:
            raise HTTPException(
                status_code=409,
                detail="A stream is already active for this session.",
            )

        _active_streams[request.session_id] = cancel_event

    def generate():
        stream = None
        completed = False
        emitted_system_count = 0
        collected_answer: list[str] = []

        previous_callback = engine.system_result_callback
        pending_system_events: list[dict[str, Any]] = []

        def collect_system_result(
            result: dict[str, Any],
        ) -> None:
            pending_system_events.append(dict(result))

        def drain_system_events():
            nonlocal emitted_system_count

            while emitted_system_count < len(pending_system_events):
                result = pending_system_events[
                    emitted_system_count
                ]
                emitted_system_count += 1

                yield {
                    "speaker": "System",
                    "status": result["status"],
                    "operation": result["operation"],
                    "message": result["system_message"],
                    "preview": (
                        result.get("data", {}).get(
                            "confirmation_preview"
                        )
                    ),
                    "speak": False,
                }

        engine.system_result_callback = collect_system_result

        try:
            stream = engine.stream_reply(
                message,
                cancel_event=cancel_event,
            )

            for chunk in stream:
                for event in drain_system_events():
                    yield _sse_event("system", event)

                collected_answer.append(chunk)

                yield _sse_event(
                    "token",
                    {"content": chunk},
                )

            for event in drain_system_events():
                yield _sse_event("system", event)

            assistant_message = "".join(collected_answer)

            completed = True

            yield _sse_event(
                "done",
                {
                    "session_id": request.session_id,
                    "turn_committed": True,
                    "assistant_message": assistant_message,
                    "system_results": list(engine.last_task_results),
                    "tool_trace": list(engine.last_database_tool_trace),
                    "metrics": dict(engine.last_turn_metrics),
                    "trace_id": engine.last_improvement_trace_id,
                },
            )

        except GeneratorExit:
            raise

        except OllamaError as exc:
            for event in drain_system_events():
                yield _sse_event("system", event)

            yield _sse_event(
                "error",
                {
                    "code": exc.code,
                    "status": (
                        "cancelled"
                        if exc.code == "cancelled"
                        else "error"
                    ),
                    "message": str(exc),
                    "turn_committed": False,
                    "system_results": list(engine.last_task_results),
                    "tool_trace": list(engine.last_database_tool_trace),
                },
            )

        except Exception:
            for event in drain_system_events():
                yield _sse_event("system", event)

            yield _sse_event(
                "error",
                {
                    "code": "stream_error",
                    "status": "error",
                    "message": "The streaming request failed.",
                    "turn_committed": False,
                    "system_results": list(engine.last_task_results),
                    "tool_trace": list(engine.last_database_tool_trace),
                },
            )

        finally:
            try:
                if not completed:
                    cancel_event.set()

                try:
                    if stream is not None:
                        stream.close()

                finally:
                    if not completed:
                        engine.cancel_active_stream()

            finally:
                engine.system_result_callback = previous_callback

                with _sessions_lock:
                    if _active_streams.get(
                        request.session_id
                    ) is cancel_event:
                        _active_streams.pop(
                            request.session_id,
                            None,
                        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/chat/reset")
def reset_chat(request: ResetRequest) -> dict[str, str]:
    engine = _get_engine(request.session_id)

    try:
        engine.reset()
    except OllamaError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    return {"status": "ok", "message": "Conversation history cleared.", "session_id": request.session_id}


@router.post("/chat/mode")
def set_mode(request: ModeRequest) -> dict[str, str]:
    mode = normalize_language_mode(request.language_mode)
    if not mode:
        raise HTTPException(status_code=400, detail="language_mode must be hinglish, hindi, or english.")
    engine = _get_engine(request.session_id)

    try:
        engine.set_language_mode(mode)
    except OllamaError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    return {"status": "ok", "language_mode": mode, "session_id": request.session_id}


@router.post("/chat/cancel")
def cancel_chat(request: ResetRequest) -> dict[str, Any]:
    """Request cancellation for one active session."""
    with _sessions_lock:
        cancel_event = _active_streams.get(request.session_id)
        engine = _sessions.get(request.session_id)

    if cancel_event is None:
        return {
            "status": "not_running",
            "cancel_requested": False,
            "session_id": request.session_id,
        }

    cancel_event.set()

    transport_interrupted = (
        engine.cancel_active_stream()
        if engine is not None
        else False
    )

    return {
        "status": "cancel_requested",
        "cancel_requested": True,
        "transport_interrupted": bool(transport_interrupted),
        "session_id": request.session_id,
        "message": (
            "Cancellation requested. The active operation will stop "
            "or reach its configured timeout."
        ),
    }