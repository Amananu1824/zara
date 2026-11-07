"""Reliable execution runtime for Zara's registered tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import random
import time
import uuid
import re
import hmac

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Condition, Event, Lock
from typing import Any, Callable, Protocol
from app.ai.tools.confirmation import ConfirmationGrant
from app.ai.tools.contracts import ToolResult
from app.ai.tools.permissions import ToolPermissionPolicy

from app.ai.tools.registry import (
    ToolRegistry,
    ToolValidationError,
)


logger = logging.getLogger(__name__)


# ============================================================
# Execution exceptions
# ============================================================

class ToolExecutionError(RuntimeError):
    """A controlled tool failure with explicit retry semantics."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_execution_failed",
        retryable: bool = False,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class ToolCancelledError(RuntimeError):
    """Cooperative cancellation requested."""


class ToolDeadlineExceeded(TimeoutError):
    """The execution deadline has expired."""


# ============================================================
# Cancellation
# ============================================================

class CancellationToken:
    """
    Thread-safe cooperative cancellation.

    Can observe both a runtime-owned event and an existing
    application cancellation event.
    """

    def __init__(
        self,
        parent_event: Any | None = None,
    ) -> None:

        if (
            parent_event is not None
            and not callable(getattr(parent_event, "is_set", None))
        ):
            raise TypeError(
                "parent_event must expose is_set()."
            )

        self._event = Event()
        self._parent_event = parent_event

        self._lock = Lock()
        self._reason = "cancelled"

    def cancel(
        self,
        reason: str = "cancelled",
    ) -> None:

        with self._lock:
            self._reason = (
                reason[:200]
                if isinstance(reason, str)
                else "cancelled"
            )

        self._event.set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or bool(
            self._parent_event is not None
            and self._parent_event.is_set()
        )

    def checkpoint(
        self,
        deadline: float,
    ) -> None:

        if self.cancelled:
            raise ToolCancelledError(
                self.reason
            )

        if time.monotonic() >= deadline:
            raise ToolDeadlineExceeded(
                "Tool execution deadline exceeded."
            )

    def wait(
        self,
        seconds: float,
        deadline: float,
    ) -> None:
        """Wait without ignoring cancellation or execution deadlines."""

        target = min(
            time.monotonic() + max(0.0, seconds),
            deadline,
        )

        while True:
            self.checkpoint(deadline)

            remaining = target - time.monotonic()

            if remaining <= 0:
                return

            self._event.wait(
                min(remaining, 0.05)
            )


# ============================================================
# Per-invocation context
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    tool_name: str
    session_id: str

    call_id: str
    trace_id: str
    idempotency_key: str

    attempt: int
    deadline: float

    cancellation: CancellationToken

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0,
            self.deadline - time.monotonic(),
        )

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation.cancelled

    def checkpoint(self) -> None:
        """Handlers must call this between long-running operations."""

        self.cancellation.checkpoint(
            self.deadline
        )

    def wait(self, seconds: float) -> None:
        """Cancellation-aware waiting for tool handlers."""

        self.cancellation.wait(
            seconds,
            self.deadline,
        )


# ============================================================
# Global execution policy
# ============================================================

@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Application-wide hard limits for registered tools."""

    max_concurrent: int = 4
    max_concurrent_per_session: int = 2

    max_retries: int = 2

    retry_base_delay: float = 0.25
    retry_max_delay: float = 2.0

    idempotency_ttl_seconds: int = 86400
    max_idempotency_records: int = 4096

    def __post_init__(self) -> None:

        for name in (
            "max_concurrent",
            "max_concurrent_per_session",
            "max_retries",
            "idempotency_ttl_seconds",
            "max_idempotency_records",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer."
                )

        if not 1 <= self.max_concurrent <= 64:
            raise ValueError("Invalid global concurrency limit.")

        if not 1 <= self.max_concurrent_per_session <= self.max_concurrent:
            raise ValueError("Invalid per-session concurrency limit.")

        if not 0 <= self.max_retries <= 3:
            raise ValueError("Invalid retry limit.")

        if (
            not 0 < self.idempotency_ttl_seconds <= 604800
            or self.max_idempotency_records < 64
        ):
            raise ValueError("Invalid idempotency storage limits.")

        for name in (
            "retry_base_delay",
            "retry_max_delay",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                or value > 30
            ):
                raise ValueError(
                    f"Invalid execution policy value: {name}."
                )

        if self.retry_base_delay > self.retry_max_delay:
            raise ValueError(
                "Retry base delay cannot exceed maximum delay."
            )


# ============================================================
# Idempotency storage interface
# ============================================================

class IdempotencyStore(Protocol):
    """Storage contract; a persistent backend can replace memory."""

    def reserve(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> tuple[str, dict[str, Any] | None]:
        ...

    def complete(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:
        ...

    def mark_indeterminate(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:
        ...

    def abandon(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> None:
        ...


@dataclass(slots=True)
class _IdempotencyEntry:
    fingerprint: str
    state: str

    created_at: float
    expires_at: float | None

    result: dict[str, Any] | None = None


class InMemoryIdempotencyStore:
    """
    Bounded, process-local idempotency ledger.

    Completed outcomes expire normally.
    Indeterminate outcomes are never automatically evicted.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 86400,
        max_entries: int = 4096,
    ) -> None:

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._entries: OrderedDict[
            str,
            _IdempotencyEntry,
        ] = OrderedDict()

        self._lock = Lock()

    @staticmethod
    def _storage_key(
        session_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> str:

        raw = "\0".join((
            session_id,
            tool_name,
            idempotency_key,
        ))

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def _purge_locked(self) -> None:

        now = time.monotonic()

        expired = [
            key
            for key, entry in self._entries.items()
            if (
                entry.state == "completed"
                and entry.expires_at is not None
                and entry.expires_at <= now
            )
        ]

        for key in expired:
            self._entries.pop(key, None)

    def _make_capacity_locked(self) -> bool:

        self._purge_locked()

        while len(self._entries) >= self.max_entries:

            removable = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if entry.state == "completed"
                ),
                None,
            )

            if removable is None:
                return False

            self._entries.pop(removable, None)

        return True

    def reserve(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> tuple[str, dict[str, Any] | None]:

        storage_key = self._storage_key(
            session_id,
            tool_name,
            idempotency_key,
        )

        with self._lock:
            self._purge_locked()

            existing = self._entries.get(storage_key)

            if existing is not None:

                self._entries.move_to_end(storage_key)

                if existing.fingerprint != fingerprint:
                    return "conflict", None

                if existing.state == "running":
                    return "in_progress", None

                if existing.state == "indeterminate":
                    return (
                        "indeterminate",
                        deepcopy(existing.result),
                    )

                if existing.state == "completed":
                    return (
                        "completed",
                        deepcopy(existing.result),
                    )

            if not self._make_capacity_locked():
                return "capacity", None

            self._entries[storage_key] = _IdempotencyEntry(
                fingerprint=fingerprint,
                state="running",
                created_at=time.monotonic(),
                expires_at=None,
            )

            return "new", None

    def complete(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:

        key = self._storage_key(
            session_id,
            tool_name,
            idempotency_key,
        )

        with self._lock:
            entry = self._entries.get(key)

            if entry is None or entry.state != "running":
                raise RuntimeError(
                    "Idempotency reservation is no longer active."
                )

            entry.state = "completed"
            entry.result = deepcopy(result)
            entry.expires_at = (
                time.monotonic() + self.ttl_seconds
            )

    def mark_indeterminate(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:

        key = self._storage_key(
            session_id,
            tool_name,
            idempotency_key,
        )

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                raise RuntimeError(
                    "Idempotency reservation was lost."
                )

            entry.state = "indeterminate"
            entry.result = deepcopy(result)
            entry.expires_at = None

    def abandon(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> None:

        key = self._storage_key(
            session_id,
            tool_name,
            idempotency_key,
        )

        with self._lock:
            entry = self._entries.get(key)

            if entry is not None and entry.state == "running":
                self._entries.pop(key, None)

    def forget_indeterminate(
        self,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
        *,
        verified_reconciled: bool = False,
    ) -> bool:
        """
        Release a blocked record only after external reconciliation.

        This method must be called by trusted application code,
        never directly by a model-generated tool request.
        """

        if verified_reconciled is not True:
            return False

        key = self._storage_key(
            session_id,
            tool_name,
            idempotency_key,
        )

        with self._lock:
            entry = self._entries.get(key)

            if entry is None or entry.state != "indeterminate":
                return False

            self._entries.pop(key, None)
            return True

    @property
    def size(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._entries)


# ============================================================
# Bounded concurrency gate
# ============================================================

class _ExecutionGate:

    def __init__(
        self,
        maximum: int,
        per_session: int,
    ) -> None:

        self.maximum = maximum
        self.per_session = per_session

        self._condition = Condition()

        self._active_total = 0
        self._active_by_session: dict[str, int] = {}

    def acquire(
        self,
        session_id: str,
        context: ToolExecutionContext,
    ) -> None:

        with self._condition:

            while True:

                context.checkpoint()

                session_active = self._active_by_session.get(
                    session_id,
                    0,
                )

                available = (
                    self._active_total < self.maximum
                    and session_active < self.per_session
                )

                if available:
                    self._active_total += 1

                    self._active_by_session[session_id] = (
                        session_active + 1
                    )

                    return

                remaining = context.remaining_seconds

                if remaining <= 0:
                    raise ToolDeadlineExceeded(
                        "Timed out waiting for an execution slot."
                    )

                self._condition.wait(
                    min(0.1, remaining)
                )

    def release(self, session_id: str) -> None:

        with self._condition:

            session_active = self._active_by_session.get(
                session_id,
                0,
            )

            if self._active_total <= 0 or session_active <= 0:
                raise RuntimeError(
                    "Execution concurrency counter is inconsistent."
                )

            self._active_total -= 1

            if session_active == 1:
                self._active_by_session.pop(
                    session_id,
                    None,
                )
            else:
                self._active_by_session[session_id] = (
                    session_active - 1
                )

            self._condition.notify_all()


# ============================================================
# Tool execution runtime
# ============================================================

ToolHandler = Callable[
    [dict[str, Any], ToolExecutionContext],
    dict[str, Any] | ToolResult,
]


class ToolExecutionRuntime:
    """
    Shared synchronous execution boundary.

    Handler signature:
        handler(validated_arguments, execution_context)

    This runtime does not dynamically import code or execute
    model-provided functions.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_policy: ToolPermissionPolicy,
        *,
        handlers: dict[str, ToolHandler] | None = None,
        policy: ExecutionPolicy | None = None,
        idempotency_store: IdempotencyStore | None = None,
    ) -> None:

        if not isinstance(registry, ToolRegistry):
            raise TypeError("A valid ToolRegistry is required.")

        if not isinstance(
            permission_policy,
            ToolPermissionPolicy,
        ):
            raise TypeError(
                "A valid ToolPermissionPolicy is required."
            )

        self.registry = registry
        self.permission_policy = permission_policy

        self.policy = policy or ExecutionPolicy()

        self.idempotency_store = (
            idempotency_store
            or InMemoryIdempotencyStore(
                ttl_seconds=self.policy.idempotency_ttl_seconds,
                max_entries=self.policy.max_idempotency_records,
            )
        )

        self._handlers: dict[str, ToolHandler] = {}

        self._gate = _ExecutionGate(
            self.policy.max_concurrent,
            self.policy.max_concurrent_per_session,
        )

        for name, handler in (handlers or {}).items():
            self.register_handler(name, handler)

    def register_handler(
        self,
        tool_name: str,
        handler: ToolHandler,
    ) -> None:
        """Register trusted application code for an existing tool."""

        if self.registry.get(tool_name) is None:
            raise ValueError(
                "Cannot register a handler for an unknown tool."
            )

        if not callable(handler):
            raise TypeError("Tool handler must be callable.")

        if inspect.iscoroutinefunction(handler):
            raise TypeError(
                "Register a synchronous handler or an explicit adapter."
            )

        if tool_name in self._handlers:
            raise ValueError(
                f"Handler already registered for {tool_name!r}."
            )

        self._handlers[tool_name] = handler

    @staticmethod
    def _validate_identity(
        value: str,
        field_name: str,
        *,
        minimum: int = 1,
        maximum: int = 128,
    ) -> str:

        if (
            not isinstance(value, str)
            or not minimum <= len(value) <= maximum
            or value != value.strip()
            or "\x00" in value
        ):
            raise ValueError(
                f"Invalid {field_name}."
            )

        return value

    @staticmethod
    def _fingerprint(
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:

        serialized = json.dumps(
            {
                "tool_name": tool_name,
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _result(
        tool_name: str,
        status: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        duration_ms: int | None = None,
        trace_id: str | None = None,
        call_id: str | None = None,
    ) -> dict[str, Any]:

        return ToolResult(
            tool_name=tool_name,
            status=status,
            message=message,
            data=data or {},
            error_code=error_code,
            retryable=retryable,
            duration_ms=duration_ms,
            trace_id=trace_id,
            call_id=call_id,
        ).to_legacy_dict()

    def _decorate_result(
        self,
        tool_name: str,
        result: dict[str, Any],
        *,
        attempts: int | None,
        duration_ms: int,
        trace_id: str,
        call_id: str,
        replayed: bool = False,
    ) -> dict[str, Any]:

        normalized = ToolResult.from_mapping(
            tool_name,
            result,
        )

        data = deepcopy(normalized.data)

        if attempts is not None:
            data["attempt_count"] = attempts

        if replayed:
            data["idempotency_replayed"] = True

        return ToolResult(
            tool_name=tool_name,
            status=normalized.status,
            message=normalized.message,
            data=data,
            error_code=normalized.error_code,
            retryable=normalized.retryable,
            duration_ms=duration_ms,
            trace_id=trace_id,
            call_id=call_id,
        ).to_legacy_dict()

    def _complete(
        self,
        result: dict[str, Any],
        *,
        session_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> dict[str, Any]:

        try:
            self.idempotency_store.complete(
                session_id,
                tool_name,
                idempotency_key,
                result,
            )

        except Exception as exc:
            logger.error(
                "Tool idempotency completion failed (%s).",
                type(exc).__name__,
            )

            unknown = self._result(
                tool_name,
                "error",
                (
                    "The operation returned, but its recovery record "
                    "could not be stored. Verify the result before retrying."
                ),
                error_code="idempotency_commit_failed",
                retryable=False,
                trace_id=result.get("trace_id"),
                call_id=result.get("call_id"),
            )

            try:
                self.idempotency_store.mark_indeterminate(
                    session_id,
                    tool_name,
                    idempotency_key,
                    unknown,
                )
            except Exception:
                logger.critical(
                    "Tool outcome and idempotency state could not be persisted."
                )

            return unknown

        return result

    def _indeterminate(
        self,
        *,
        tool_name: str,
        session_id: str,
        idempotency_key: str,
        message: str,
        code: str,
        trace_id: str,
        call_id: str,
        duration_ms: int,
        attempts: int,
    ) -> dict[str, Any]:

        result = self._result(
            tool_name,
            "error",
            message,
            data={
                "outcome_verified": False,
                "manual_reconciliation_required": True,
                "attempt_count": attempts,
            },
            error_code=code,
            retryable=False,
            duration_ms=duration_ms,
            trace_id=trace_id,
            call_id=call_id,
        )

        try:
            self.idempotency_store.mark_indeterminate(
                session_id,
                tool_name,
                idempotency_key,
                result,
            )
        except Exception as exc:
            logger.critical(
                "Could not persist an indeterminate tool outcome (%s).",
                type(exc).__name__,
            )

        return result

    def execute(
        self,
        tool_name: str,
        raw_arguments: Any,
        *,
        session_id: str,
        interface: str,
        idempotency_key: str | None = None,
        call_id: str | None = None,
        trace_id: str | None = None,
        cancel_event: Any | None = None,
        confirmation_grant: ConfirmationGrant | None = None,
    ) -> dict[str, Any]:
        """Validate, authorize and reliably execute one registered tool."""

        started = time.monotonic()

        if (
            not isinstance(tool_name, str)
            or self.registry.get(tool_name) is None
        ):
            safe_name = (
                tool_name
                if (
                    isinstance(tool_name, str)
                    and re.fullmatch(
                        r"[a-z][a-z0-9_]{0,63}",
                        tool_name,
                    )
                )
                else "unknown_tool"
            )

            return self._result(
                safe_name,
                "not_found",
                "The requested tool is not registered.",
                error_code="unknown_tool",
            )

        spec = self.registry.get(tool_name)
        assert spec is not None

        handler = self._handlers.get(tool_name)

        if handler is None:
            return self._result(
                tool_name,
                "error",
                "No trusted handler is registered for this tool.",
                error_code="handler_not_registered",
            )

        try:
            session_id = self._validate_identity(
                session_id,
                "session_id",
            )

            selected_call_id = (
                self._validate_identity(call_id, "call_id")
                if call_id is not None
                else uuid.uuid4().hex
            )

            selected_trace_id = (
                self._validate_identity(trace_id, "trace_id")
                if trace_id is not None
                else uuid.uuid4().hex
            )

        except ValueError:
            return self._result(
                tool_name,
                "error",
                "The execution identity is invalid.",
                error_code="invalid_execution_identity",
            )

        cancellation = CancellationToken(cancel_event)

        if cancellation.cancelled:
            return self._result(
                tool_name,
                "cancelled",
                "Execution was cancelled before it started.",
                error_code="cancelled_before_execution",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        # ----------------------------------------------------
        # Permission boundary
        # ----------------------------------------------------

        decision = self.permission_policy.authorize(
            tool_name,
            interface=interface,
        )

        if not decision.allowed:
            return self._result(
                tool_name,
                "denied",
                decision.message,
                error_code=decision.code,
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        # ----------------------------------------------------
        # Argument boundary
        # ----------------------------------------------------

        try:
            arguments = self.registry.validate_arguments(
                tool_name,
                raw_arguments,
            )

            if not isinstance(arguments, dict):
                raise ToolValidationError(
                    "Validated arguments must be an object."
                )

            fingerprint = self._fingerprint(
                tool_name,
                arguments,
            )


        except (ToolValidationError, TypeError, ValueError) as exc:
            return self._result(
                tool_name,
                "error",
                "Tool arguments did not pass validation.",
                error_code="invalid_tool_arguments",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        # Verify the exact one-time approval received from the router.
        confirmation_digest = hashlib.sha256(
            json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        if decision.requires_confirmation and confirmation_grant is None:
            return self._result(
                tool_name,
                "needs_confirmation",
                "Explicit approval is required.",
                error_code="confirmation_required",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if confirmation_grant is not None:

            if not isinstance(
                confirmation_grant,
                ConfirmationGrant,
            ):
                return self._result(
                    tool_name,
                    "denied",
                    "The approval object is invalid.",
                    error_code="invalid_confirmation_grant",
                    trace_id=selected_trace_id,
                    call_id=selected_call_id,
                )

            approved_age = (
                time.time() - confirmation_grant.approved_at
            )

            valid_grant = (
                confirmation_grant.tool_name == tool_name
                and confirmation_grant.session_id == session_id
                and hmac.compare_digest(
                    confirmation_grant.arguments_digest,
                    confirmation_digest,
                )
                and 0 <= approved_age <= 120
            )

            if not valid_grant:
                return self._result(
                    tool_name,
                    "denied",
                    "Approval is invalid, expired or does not match the action.",
                    error_code="invalid_confirmation_grant",
                    trace_id=selected_trace_id,
                    call_id=selected_call_id,
                )
            
        # ----------------------------------------------------
        # Idempotency boundary
        # ----------------------------------------------------

        if spec.permission == "mutate" and idempotency_key is None:
            return self._result(
                tool_name,
                "error",
                (
                    "A unique idempotency key is required "
                    "for mutating operations."
                ),
                error_code="idempotency_key_required",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if idempotency_key is None:
            idempotency_key = selected_call_id

        try:
            idempotency_key = self._validate_identity(
                idempotency_key,
                "idempotency_key",
            )

            reservation, previous = (
                self.idempotency_store.reserve(
                    session_id,
                    tool_name,
                    idempotency_key,
                    fingerprint,
                )
            )

        except Exception as exc:
            logger.error(
                "Tool idempotency reservation failed (%s).",
                type(exc).__name__,
            )

            return self._result(
                tool_name,
                "error",
                "The duplicate-prevention service is unavailable.",
                error_code="idempotency_store_unavailable",
                retryable=False,
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if reservation == "completed" and previous is not None:
            return self._decorate_result(
                tool_name,
                previous,
                attempts=None,
                duration_ms=int(
                    (time.monotonic() - started) * 1000
                ),
                trace_id=selected_trace_id,
                call_id=selected_call_id,
                replayed=True,
            )

        if reservation == "in_progress":
            return self._result(
                tool_name,
                "error",
                "An identical tool operation is already running.",
                error_code="duplicate_operation_in_progress",
                retryable=True,
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if reservation == "conflict":
            return self._result(
                tool_name,
                "denied",
                (
                    "The idempotency key was already used with "
                    "different arguments."
                ),
                error_code="idempotency_key_conflict",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if reservation == "indeterminate":
            return self._result(
                tool_name,
                "error",
                (
                    "The previous operation has an uncertain outcome. "
                    "Verify its actual state before retrying."
                ),
                data={
                    "manual_reconciliation_required": True,
                },
                error_code="previous_outcome_indeterminate",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        if reservation != "new":
            return self._result(
                tool_name,
                "error",
                "Execution capacity is currently unavailable.",
                error_code="idempotency_capacity_reached",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        # ----------------------------------------------------
        # Deadline and retry policy
        # ----------------------------------------------------

        timeout = min(
            spec.metadata.timeout_seconds,
            600,
        )

        deadline = time.monotonic() + timeout

        can_retry = (
            spec.permission == "read"
            or spec.metadata.idempotent
        )

        retries = (
            min(
                spec.metadata.max_retries,
                self.policy.max_retries,
            )
            if can_retry
            else 0
        )

        max_attempts = retries + 1
        acquired = False
        attempts = 0

        try:
            # ------------------------------------------------
            # Concurrency control
            # ------------------------------------------------

            context = ToolExecutionContext(
                tool_name=tool_name,
                session_id=session_id,
                call_id=selected_call_id,
                trace_id=selected_trace_id,
                idempotency_key=idempotency_key,
                attempt=1,
                deadline=deadline,
                cancellation=cancellation,
            )

            try:
                self._gate.acquire(
                    session_id,
                    context,
                )
                acquired = True

            except (ToolCancelledError, ToolDeadlineExceeded):
                self.idempotency_store.abandon(
                    session_id,
                    tool_name,
                    idempotency_key,
                )

                return self._result(
                    tool_name,
                    (
                        "cancelled"
                        if cancellation.cancelled
                        else "error"
                    ),
                    "Execution stopped before the handler started.",
                    error_code=(
                        "cancelled"
                        if cancellation.cancelled
                        else "execution_deadline_exceeded"
                    ),
                    trace_id=selected_trace_id,
                    call_id=selected_call_id,
                )

            # ------------------------------------------------
            # Bounded execution and retry loop
            # ------------------------------------------------

            while attempts < max_attempts:

                attempts += 1
                handler_started = False

                context = ToolExecutionContext(
                    tool_name=tool_name,
                    session_id=session_id,
                    call_id=selected_call_id,
                    trace_id=selected_trace_id,
                    idempotency_key=idempotency_key,
                    attempt=attempts,
                    deadline=deadline,
                    cancellation=cancellation,
                )

                try:
                    context.checkpoint()

                    handler_started = True

                    raw_result = handler(
                        deepcopy(arguments),
                        context,
                    )

                    # Async handlers must use an explicit adapter.
                    if inspect.isawaitable(raw_result):
                        close = getattr(raw_result, "close", None)

                        if callable(close):
                            close()

                        raise ToolExecutionError(
                            "Async handlers require an explicit adapter.",
                            code="async_handler_adapter_required",
                        )

                    # A synchronous Python call cannot be forcibly
                    # killed safely. If it ignored the deadline and
                    # returned late, never assume its state.
                    if time.monotonic() >= deadline:

                        if spec.permission == "mutate":
                            return self._indeterminate(
                                tool_name=tool_name,
                                session_id=session_id,
                                idempotency_key=idempotency_key,
                                message=(
                                    "The tool returned after its deadline. "
                                    "Its final state must be verified before "
                                    "another write is attempted."
                                ),
                                code="execution_outcome_unknown",
                                trace_id=selected_trace_id,
                                call_id=selected_call_id,
                                duration_ms=int(
                                    (time.monotonic() - started) * 1000
                                ),
                                attempts=attempts,
                            )

                        result = self._result(
                            tool_name,
                            "error",
                            "The read operation exceeded its deadline.",
                            error_code="execution_deadline_exceeded",
                            trace_id=selected_trace_id,
                            call_id=selected_call_id,
                            duration_ms=int(
                                (time.monotonic() - started) * 1000
                            ),
                        )

                        return self._complete(
                            result,
                            session_id=session_id,
                            tool_name=tool_name,
                            idempotency_key=idempotency_key,
                        )

                    normalized = self.registry.validate_result(
                        tool_name,
                        (
                            raw_result.to_legacy_dict()
                            if isinstance(raw_result, ToolResult)
                            else raw_result
                        ),
                    )

                except ToolCancelledError:

                    if handler_started and spec.permission == "mutate":
                        return self._indeterminate(
                            tool_name=tool_name,
                            session_id=session_id,
                            idempotency_key=idempotency_key,
                            message=(
                                "Cancellation was requested during a "
                                "mutating operation. Its final state is "
                                "unknown and must be reconciled."
                            ),
                            code="execution_outcome_unknown",
                            trace_id=selected_trace_id,
                            call_id=selected_call_id,
                            duration_ms=int(
                                (time.monotonic() - started) * 1000
                            ),
                            attempts=attempts,
                        )

                    result = self._result(
                        tool_name,
                        "cancelled",
                        "The operation was cancelled.",
                        error_code="execution_cancelled",
                        trace_id=selected_trace_id,
                        call_id=selected_call_id,
                        duration_ms=int(
                            (time.monotonic() - started) * 1000
                        ),
                    )

                    return self._complete(
                        result,
                        session_id=session_id,
                        tool_name=tool_name,
                        idempotency_key=idempotency_key,
                    )

                except ToolDeadlineExceeded:

                    if handler_started and spec.permission == "mutate":
                        return self._indeterminate(
                            tool_name=tool_name,
                            session_id=session_id,
                            idempotency_key=idempotency_key,
                            message=(
                                "The execution deadline expired during a "
                                "write. Verify the actual result before retrying."
                            ),
                            code="execution_outcome_unknown",
                            trace_id=selected_trace_id,
                            call_id=selected_call_id,
                            duration_ms=int(
                                (time.monotonic() - started) * 1000
                            ),
                            attempts=attempts,
                        )

                    result = self._result(
                        tool_name,
                        "error",
                        "The operation exceeded its execution deadline.",
                        error_code="execution_deadline_exceeded",
                        trace_id=selected_trace_id,
                        call_id=selected_call_id,
                        duration_ms=int(
                            (time.monotonic() - started) * 1000
                        ),
                    )

                    return self._complete(
                        result,
                        session_id=session_id,
                        tool_name=tool_name,
                        idempotency_key=idempotency_key,
                    )

                except ToolExecutionError as exc:

                    if (
                        spec.permission == "mutate"
                        and exc.outcome_unknown
                    ):
                        return self._indeterminate(
                            tool_name=tool_name,
                            session_id=session_id,
                            idempotency_key=idempotency_key,
                            message=(
                                "The handler reported an uncertain "
                                "mutation outcome. Manual verification is required."
                            ),
                            code="execution_outcome_unknown",
                            trace_id=selected_trace_id,
                            call_id=selected_call_id,
                            duration_ms=int(
                                (time.monotonic() - started) * 1000
                            ),
                            attempts=attempts,
                        )

                    normalized = self._result(
                        tool_name,
                        "error",
                        str(exc)[:1000] or "Tool execution failed.",
                        error_code=exc.code,
                        retryable=exc.retryable,
                    )

                except Exception as exc:

                    logger.error(
                        "Tool '%s' raised %s.",
                        tool_name,
                        type(exc).__name__,
                    )

                    if spec.permission == "mutate":
                        return self._indeterminate(
                            tool_name=tool_name,
                            session_id=session_id,
                            idempotency_key=idempotency_key,
                            message=(
                                "A write handler failed unexpectedly. "
                                "Its final state must be verified before retrying."
                            ),
                            code="execution_outcome_unknown",
                            trace_id=selected_trace_id,
                            call_id=selected_call_id,
                            duration_ms=int(
                                (time.monotonic() - started) * 1000
                            ),
                            attempts=attempts,
                        )

                    normalized = self._result(
                        tool_name,
                        "error",
                        "The tool execution failed.",
                        error_code="tool_execution_failed",
                        retryable=False,
                    )

                retryable = (
                    normalized.get("status") in {"error", "failed"}
                    and normalized.get("retryable") is True
                )

                if retryable and attempts < max_attempts:

                    delay = min(
                        self.policy.retry_base_delay * (2 ** (attempts - 1)),
                        self.policy.retry_max_delay,
                    )

                    delay *= random.uniform(0.8, 1.2)

                    try:
                        context.wait(delay)

                    except (ToolCancelledError, ToolDeadlineExceeded):
                        result = self._result(
                            tool_name,
                            (
                                "cancelled"
                                if cancellation.cancelled
                                else "error"
                            ),
                            "Retry was stopped before another attempt.",
                            error_code=(
                                "retry_cancelled"
                                if cancellation.cancelled
                                else "retry_deadline_exceeded"
                            ),
                        )

                        return self._complete(
                            result,
                            session_id=session_id,
                            tool_name=tool_name,
                            idempotency_key=idempotency_key,
                        )

                    continue

                final_result = self._decorate_result(
                    tool_name,
                    normalized,
                    attempts=attempts,
                    duration_ms=int(
                        (time.monotonic() - started) * 1000
                    ),
                    trace_id=selected_trace_id,
                    call_id=selected_call_id,
                )

                return self._complete(
                    final_result,
                    session_id=session_id,
                    tool_name=tool_name,
                    idempotency_key=idempotency_key,
                )

            raise RuntimeError(
                "Execution retry loop ended unexpectedly."
            )

        except Exception as exc:

            logger.error(
                "Tool execution runtime failed (%s).",
                type(exc).__name__,
            )

            # A reservation must not remain falsely marked as running.
            # Conservatively block a possible duplicate mutation.
            if spec.permission == "mutate":
                return self._indeterminate(
                    tool_name=tool_name,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    message=(
                        "The execution runtime failed. Verify the operation "
                        "before attempting it again."
                    ),
                    code="execution_recovery_required",
                    trace_id=selected_trace_id,
                    call_id=selected_call_id,
                    duration_ms=int(
                        (time.monotonic() - started) * 1000
                    ),
                    attempts=attempts,
                )

            self.idempotency_store.abandon(
                session_id,
                tool_name,
                idempotency_key,
            )

            return self._result(
                tool_name,
                "error",
                "The execution runtime could not complete the request.",
                error_code="execution_runtime_failed",
                trace_id=selected_trace_id,
                call_id=selected_call_id,
            )

        finally:
            if acquired:
                self._gate.release(session_id)

    async def execute_async(
        self,
        tool_name: str,
        raw_arguments: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Async application adapter for the synchronous runtime.

        Cancelling the awaiting coroutine requests cooperative
        cancellation; it does not forcibly terminate its worker thread.
        """

        cancel_event = kwargs.pop(
            "cancel_event",
            None,
        )

        if cancel_event is None:
            cancel_event = Event()

        task = asyncio.create_task(
            asyncio.to_thread(
                self.execute,
                tool_name,
                raw_arguments,
                cancel_event=cancel_event,
                **kwargs,
            )
        )

        try:
            return await task

        except asyncio.CancelledError:
            cancel_event.set()
            raise