"""Privacy-safe, hash-linked audit logging for Zara tool actions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid

from pathlib import Path
from threading import RLock
from typing import Any


class ToolAuditError(RuntimeError):
    """Audit storage failed or its existing chain is invalid."""


class ToolAuditLogger:
    """Append-only JSONL audit log with bounded rotation."""

    MAX_EVENT_BYTES = 16384

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 65536 <= max_bytes <= 100 * 1024 * 1024
        ):
            raise ValueError("Invalid audit rotation size.")

        if (
            isinstance(backup_count, bool)
            or not isinstance(backup_count, int)
            or not 1 <= backup_count <= 10
        ):
            raise ValueError("Invalid audit backup count.")

        configured_path = (
            path
            or os.environ.get("ZARA_TOOL_AUDIT_PATH")
            or (
                Path(__file__).resolve().parents[3]
                / "logs"
                / "tool_audit.jsonl"
            )
        )

        self.path = Path(configured_path).expanduser().resolve()

        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self._lock = RLock()
        self._last_error: str | None = None

        try:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._last_hash = self._load_last_hash()

            integrity = self.verify_chain()

            if not integrity.get("valid"):
                raise ToolAuditError(
                    "Existing audit chain failed integrity verification."
                )

            if integrity.get("last_hash") != self._last_hash:
                raise ToolAuditError(
                    "Audit chain head does not match the stored hash."
                )

        except Exception as exc:
            raise ToolAuditError(
                "Audit storage initialization failed."
            ) from exc

    @property
    def is_healthy(self) -> bool:
        return self._last_error is None

    def _paths_oldest_first(self) -> list[Path]:

        rotated = [
            self.path.with_name(
                f"{self.path.name}.{index}"
            )
            for index in range(self.backup_count, 0, -1)
        ]

        return rotated + [self.path]

    def _load_last_hash(self) -> str | None:

        candidates = [self.path] + [
            self.path.with_name(
                f"{self.path.name}.{index}"
            )
            for index in range(1, self.backup_count + 1)
        ]

        for candidate in candidates:

            if not candidate.exists():
                continue

            try:
                with candidate.open("rb") as handle:
                    lines = [
                        line
                        for line in handle.read().splitlines()
                        if line.strip()
                    ]

                if not lines:
                    continue

                event = json.loads(lines[-1])

                digest = event.get("event_hash")

                if (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", digest)
                ):
                    raise ValueError("Invalid event hash.")

                return digest

            except Exception as exc:
                raise ToolAuditError(
                    "Existing audit log cannot be safely continued."
                ) from exc

        return None

    @staticmethod
    def _canonical_bytes(
        event: dict[str, Any],
    ) -> bytes:

        return json.dumps(
            event,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def _rotate_locked(self) -> None:

        if (
            not self.path.exists()
            or self.path.stat().st_size == 0
        ):
            return

        oldest = self.path.with_name(
            f"{self.path.name}.{self.backup_count}"
        )

        if oldest.exists():
            oldest.unlink()

        for index in range(self.backup_count - 1, 0, -1):

            source = self.path.with_name(
                f"{self.path.name}.{index}"
            )

            target = self.path.with_name(
                f"{self.path.name}.{index + 1}"
            )

            if source.exists():
                os.replace(source, target)

        os.replace(
            self.path,
            self.path.with_name(f"{self.path.name}.1"),
        )

    def record_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Write only approved audit metadata.

        Raw arguments, personal content, file contents,
        prompts and confirmation secrets are never stored.
        """

        if not isinstance(event, dict):
            raise ToolAuditError("Audit event must be an object.")

        event_type = event.get(
            "event_type",
            "action_result",
        )

        tool_name = event.get(
            "tool",
            event.get("tool_name", "unknown_tool"),
        )

        status = event.get("status", "unknown")
        risk = event.get("risk_level", "unknown")

        slug = r"[a-z][a-z0-9_.-]{0,63}"

        if (
            not isinstance(event_type, str)
            or not re.fullmatch(slug, event_type)
        ):
            raise ToolAuditError("Invalid audit event type.")

        if (
            not isinstance(tool_name, str)
            or not re.fullmatch(slug, tool_name)
        ):
            tool_name = "unknown_tool"

        if (
            not isinstance(status, str)
            or len(status) > 64
            or not status
        ):
            status = "unknown"

        if (
            not isinstance(risk, str)
            or risk not in {
                "low",
                "medium",
                "high",
                "critical",
                "unknown",
            }
        ):
            risk = "unknown"

        session_id = event.get("session_id")

        session_hash = (
            hashlib.sha256(
                session_id.encode("utf-8")
            ).hexdigest()
            if isinstance(session_id, str)
            else None
        )

        permission = event.get("permission")

        if permission not in {"read", "mutate", None}:
            permission = None

        confirmation = event.get(
            "confirmation_required",
            False,
        )

        if not isinstance(confirmation, bool):
            confirmation = False

        duration = event.get("duration_ms")

        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
        ):
            duration = None

        def safe_id(value: Any) -> str | None:

            if isinstance(value, str) and len(value) <= 128:
                return value

            return None

        error_code = event.get("error_code")

        if (
            not isinstance(error_code, str)
            or not re.fullmatch(
                r"[a-z0-9_]{1,100}",
                error_code,
            )
        ):
            error_code = None

        argument_digest = event.get("argument_digest")

        if (
            not isinstance(argument_digest, str)
            or not re.fullmatch(
                r"[a-f0-9]{64}",
                argument_digest,
            )
        ):
            argument_digest = None

        payload = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "timestamp": time.time(),
            "tool": tool_name,
            "status": status,
            "risk_level": risk,
            "permission": permission,
            "confirmation_required": confirmation,
            "interface": safe_id(event.get("interface")),
            "source": safe_id(event.get("source")),
            "session_hash": session_hash,
            "call_id": safe_id(event.get("call_id")),
            "trace_id": safe_id(event.get("trace_id")),
            "duration_ms": duration,
            "error_code": error_code,
            "argument_digest": argument_digest,
            "previous_hash": None,
        }

        encoded_size = len(self._canonical_bytes(payload))

        if encoded_size > self.MAX_EVENT_BYTES:
            raise ToolAuditError("Audit event exceeds its size limit.")

        with self._lock:

            payload["previous_hash"] = self._last_hash

            digest = hashlib.sha256(
                self._canonical_bytes(payload)
            ).hexdigest()

            complete_event = {
                **payload,
                "event_hash": digest,
            }

            line = (
                self._canonical_bytes(complete_event)
                + b"\n"
            )

            try:

                if (
                    self.path.exists()
                    and (
                        self.path.stat().st_size + len(line)
                        > self.max_bytes
                    )
                ):
                    self._rotate_locked()

                flags = (
                    os.O_APPEND
                    | os.O_CREAT
                    | os.O_WRONLY
                )

                descriptor = os.open(
                    self.path,
                    flags,
                    0o600,
                )

                try:
                    with os.fdopen(descriptor, "ab") as handle:
                        handle.write(line)
                        handle.flush()
                        os.fsync(handle.fileno())

                except Exception:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    raise

                self._last_hash = digest
                self._last_error = None

            except Exception as exc:
                self._last_error = type(exc).__name__

                raise ToolAuditError(
                    "Audit event could not be persisted."
                ) from exc

        return dict(complete_event)

    def verify_chain(self) -> dict[str, Any]:
        """Verify event hashes and links across retained log files."""

        previous_hash: str | None = None
        checked = 0

        paths = self._paths_oldest_first()

        retained_files = [
            path for path in paths if path.exists()
        ]

        oldest_backup = self.path.with_name(
            f"{self.path.name}.{self.backup_count}"
        )

        history_complete = not oldest_backup.exists()

        try:

            for path in retained_files:

                with path.open("rb") as handle:

                    for raw_line in handle:

                        if not raw_line.strip():
                            continue

                        event = json.loads(raw_line)

                        supplied_hash = event.get("event_hash")
                        supplied_previous = event.get("previous_hash")

                        payload = {
                            key: value
                            for key, value in event.items()
                            if key != "event_hash"
                        }

                        calculated = hashlib.sha256(
                            self._canonical_bytes(payload)
                        ).hexdigest()

                        if supplied_hash != calculated:
                            raise ValueError("Event hash mismatch.")

                        if (
                            previous_hash is not None
                            and supplied_previous != previous_hash
                        ):
                            raise ValueError("Audit chain link mismatch.")

                        if (
                            previous_hash is None
                            and checked == 0
                            and supplied_previous is not None
                        ):
                            history_complete = False

                        previous_hash = supplied_hash
                        checked += 1

            return {
                "valid": True,
                "events_checked": checked,
                "history_complete": history_complete,
                "last_hash": previous_hash,
            }

        except Exception as exc:
            return {
                "valid": False,
                "events_checked": checked,
                "history_complete": False,
                "error_code": "audit_chain_integrity_failed",
            }