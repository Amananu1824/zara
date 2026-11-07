"""Unified action routing for Terminal, API, Browser, Desktop and Voice."""

from __future__ import annotations

import logging
import secrets
import inspect
import time
import uuid

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from app.ai.tools.confirmation import (
    ToolConfirmationManager,
)
from app.ai.tools.contracts import ToolResult
from app.ai.tools.permissions import ToolPermissionPolicy
from app.ai.tools.registry import ToolRegistry
from app.ai.tools.audit import ToolAuditError, ToolAuditLogger

from app.ai.tools.execution import (
    ToolExecutionError,
    ToolExecutionRuntime,
    ToolResult
)

from app.ai.tools.verification import (
    ToolOutcomeVerifier,
    VerificationReport,
)

logger = logging.getLogger(__name__)

SUPPORTED_INTERFACES = frozenset({
    "terminal",
    "api",
    "browser",
    "desktop",
    "voice",
    "internal",
})

SUPPORTED_SOURCES = frozenset({
    "model",
    "terminal_command",
    "api_request",
    "browser_action",
    "desktop_action",
    "voice_action",
    "internal",
})


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """Trusted application envelope around a requested tool operation."""

    tool_name: str
    arguments: Any

    session_id: str
    interface: str

    source: str = "model"
    user_message: str = ""

    call_id: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None

    cancel_event: Any | None = None


@dataclass(frozen=True, slots=True)
class _PendingAction:
    confirmation_id: str
    secret_token: str

    tool_name: str
    arguments: dict[str, Any]

    session_id: str
    interface: str

    user_message: str

    call_id: str
    trace_id: str
    idempotency_key: str

    summary: str
    risk_level: str

    expires_at: float


ConfirmationPreview = Callable[
    [dict[str, Any]],
    str,
]


class UnifiedActionRouter:
    """
    The single application-level boundary for tool actions.

    Responsibilities:
      - Resolve canonical registered tools.
      - Enforce trusted interface identity.
      - Apply centralized authorization.
      - Preserve legacy database-tool execution.
      - Dispatch future handlers through ExecutionRuntime.
      - Manage action-bound confirmation.
      - Produce consistent execution traces.

    The router never imports or executes model-provided code.
    """

    MAX_PENDING_CONFIRMATIONS = 128
    MAX_PENDING_PER_SESSION = 5
    MAX_TRACE_EVENTS = 500

    MAX_USER_MESSAGE_LENGTH = 12000
    MAX_PREVIEW_LENGTH = 500

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        permission_policy: ToolPermissionPolicy,
        execution_runtime: ToolExecutionRuntime,
        confirmation_manager: ToolConfirmationManager,
        database_service: Any | None = None,
        legacy_tool_names: set[str] | frozenset[str] | None = None,
        trace_callback: Callable[
            [dict[str, Any]],
            None,
        ] | None = None,
        audit_logger: ToolAuditLogger | None = None,
        result_verifier: ToolOutcomeVerifier | None = None,
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

        if not isinstance(
            execution_runtime,
            ToolExecutionRuntime,
        ):
            raise TypeError(
                "A valid ToolExecutionRuntime is required."
            )

        if not isinstance(
            confirmation_manager,
            ToolConfirmationManager,
        ):
            raise TypeError(
                "A valid ToolConfirmationManager is required."
            )

        if execution_runtime.registry is not registry:
            raise ValueError(
                "Router and execution runtime must share one registry."
            )

        if execution_runtime.permission_policy is not permission_policy:
            raise ValueError(
                "Router and execution runtime must share one permission policy."
            )

        if (
            trace_callback is not None
            and not callable(trace_callback)
        ):
            raise TypeError("trace_callback must be callable.")

        self.registry = registry
        self.permission_policy = permission_policy

        self.execution_runtime = execution_runtime
        self.confirmation_manager = confirmation_manager

        self.database_service = database_service

        self.legacy_tool_names = frozenset(
            legacy_tool_names or ()
        )

        unknown_legacy_tools = (
            self.legacy_tool_names - set(registry.names)
        )

        if unknown_legacy_tools:
            raise ValueError(
                "Legacy tool list contains unregistered tools."
            )

        self._generic_tools: set[str] = set()
        self._request_guards: dict[
            str,
            Callable[[ActionRequest, dict[str, Any]], bool],
        ] = {}

        self._confirmation_previews: dict[
            str,
            ConfirmationPreview,
        ] = {}

        self._pending: dict[
            str,
            _PendingAction,
        ] = {}

        self._lock = RLock()

        self._trace_events: deque[
            dict[str, Any]
        ] = deque(maxlen=self.MAX_TRACE_EVENTS)

        self._trace_callback = trace_callback
        self.audit_logger = audit_logger

        self.result_verifier = (
            result_verifier
            or ToolOutcomeVerifier()
        )

    # ========================================================
    # Registration and discovery
    # ========================================================

    def register_request_guard(
        self,
        category: str,
        guard: Callable[
            [ActionRequest, dict[str, Any]],
            bool,
        ],
    ) -> None:
        """
        Attach a trusted original-message guard to a tool category.

        Model arguments cannot override this guard.
        """

        if (
            not isinstance(category, str)
            or not category
            or len(category) > 64
            or not category.replace("_", "").isalnum()
        ):
            raise ValueError("Invalid tool guard category.")

        if not callable(guard):
            raise TypeError("Request guard must be callable.")

        if category in self._request_guards:
            raise ValueError(
                f"A request guard already exists for {category!r}."
            )

        self._request_guards[category] = guard
        
    def register_handler(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        *,
        confirmation_preview: ConfirmationPreview | None = None,
        result_verifier: Callable[..., Any] | None = None,
    ) -> None:
        """Register trusted code, preview and postcondition verifier."""

        spec = self.registry.get(tool_name)

        if spec is None:
            raise ValueError(
                f"Tool {tool_name!r} is not registered."
            )

        if tool_name in self.legacy_tool_names:
            raise ValueError(
                "Legacy database handlers cannot be replaced here."
            )

        if tool_name in self._generic_tools:
            raise ValueError(
                f"Handler already registered for {tool_name!r}."
            )

        if not callable(handler):
            raise TypeError("Tool handler must be callable.")

        if inspect.iscoroutinefunction(handler):
            raise TypeError(
                "Register a synchronous handler or an explicit adapter."
            )

        if (
            confirmation_preview is not None
            and not callable(confirmation_preview)
        ):
            raise TypeError(
                "Confirmation preview must be callable."
            )

        requires_confirmation = (
            spec.confirmation in {
                "required",
                "conditional",
            }
            or spec.metadata.risk_level in {
                "medium",
                "high",
                "critical",
            }
        )

        if (
            requires_confirmation
            and confirmation_preview is None
        ):
            raise ValueError(
                "Risky tools require a trusted confirmation preview."
            )

        verifier_available = (
            result_verifier is not None
            or self.result_verifier.has_verifier(tool_name)
        )

        if (
            spec.permission == "mutate"
            and not verifier_available
        ):
            raise ValueError(
                "Mutating tools require an independent result verifier."
            )

        if result_verifier is not None:
            self.result_verifier.register(
                tool_name,
                result_verifier,
            )

        def verified_handler(
            arguments: dict[str, Any],
            context: Any,
        ) -> dict[str, Any]:

            raw_result = handler(
                arguments,
                context,
            )

            if isinstance(raw_result, ToolResult):
                raw_result = raw_result.to_legacy_dict()

            normalized = self.registry.validate_result(
                tool_name,
                raw_result,
            )

            if normalized["status"] != "success":
                return normalized

            report = self.result_verifier.verify(
                tool_name,
                arguments,
                normalized,
                permission=spec.permission,
            )

            if (
                report.verified is False
                or (
                    spec.permission == "mutate"
                    and report.verified is not True
                )
            ):

                raise ToolExecutionError(
                    "Independent tool-result verification failed.",
                    code=report.error_code or "result_verification_failed",
                    retryable=False,
                    outcome_unknown=(
                        spec.permission == "mutate"
                    ),
                )

            normalized["verification"] = report.to_dict()

            return normalized

        self.execution_runtime.register_handler(
            tool_name,
            verified_handler,
        )

        if confirmation_preview is not None:
            self._confirmation_previews[
                tool_name
            ] = confirmation_preview

        self._generic_tools.add(tool_name)

    def definitions(
        self,
        *,
        interface: str,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Expose only implemented and authorized tools."""

        if (
            not isinstance(interface, str)
            or interface not in SUPPORTED_INTERFACES
        ):
            raise ValueError("Unsupported action interface.")

        if (
            category is not None
            and (
                not isinstance(category, str)
                or not category
            )
        ):
            raise ValueError("Invalid tool category.")

        available: list[dict[str, Any]] = []

        for spec in self.registry.specifications(
            interface=interface,
            category=category,
        ):
            decision = self.permission_policy.authorize(
                spec.name,
                interface=interface,
            )

            if not decision.allowed:
                continue

            is_legacy = (
                spec.name in self.legacy_tool_names
                and self.database_service is not None
            )

            has_handler = (
                spec.name in self._generic_tools
            )

            if is_legacy or has_handler:
                available.append(
                    spec.definition
                )

        return available

    # ========================================================
    # Standard results
    # ========================================================

    @staticmethod
    def _result(
        tool_name: str,
        status: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        error_code: str | None = None,
        trace_id: str | None = None,
        call_id: str | None = None,
    ) -> dict[str, Any]:

        return ToolResult(
            tool_name=tool_name,
            status=status,
            message=message,
            data=data or {},
            error_code=error_code,
            trace_id=trace_id,
            call_id=call_id,
        ).to_legacy_dict()

    def _write_audit_event(
        self,
        event: dict[str, Any],
        *,
        required: bool,
    ) -> bool:
        """Persist an action event before sensitive execution."""

        if self.audit_logger is None:
            return not required

        try:
            self.audit_logger.record_event(event)
            return True

        except ToolAuditError as exc:
            logger.error(
                "Required tool audit write failed (%s).",
                type(exc).__name__,
            )
            return not required

        except Exception as exc:
            logger.error(
                "Unexpected audit integration failure (%s).",
                type(exc).__name__,
            )
            return not required
        
    @staticmethod
    def _safe_identity(
        value: str | None,
        field_name: str,
    ) -> str:

        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 128
            or value != value.strip()
            or "\x00" in value
        ):
            raise ValueError(
                f"Invalid {field_name}."
            )

        return value

    def _validate_request(
        self,
        request: ActionRequest,
    ) -> None:
        if (
            not isinstance(request.interface, str)
            or request.interface not in SUPPORTED_INTERFACES
        ):
            raise ValueError("Unsupported action interface.")

        if (
            not isinstance(request.source, str)
            or request.source not in SUPPORTED_SOURCES
        ):
            raise ValueError("Unsupported action source.")

        self._safe_identity(
            request.tool_name,
            "tool_name",
        )

        self._safe_identity(
            request.session_id,
            "session_id",
        )

        if request.interface not in SUPPORTED_INTERFACES:
            raise ValueError("Unsupported interface.")

        if request.source not in SUPPORTED_SOURCES:
            raise ValueError("Unsupported action source.")

        if (
            not isinstance(request.user_message, str)
            or len(request.user_message) > self.MAX_USER_MESSAGE_LENGTH
        ):
            raise ValueError("Invalid user message.")

        for value, field_name in (
            (request.call_id, "call_id"),
            (request.trace_id, "trace_id"),
            (request.idempotency_key, "idempotency_key"),
        ):
            if value is not None:
                self._safe_identity(
                    value,
                    field_name,
                )

        if (
            request.cancel_event is not None
            and not callable(
                getattr(request.cancel_event, "is_set", None)
            )
        ):
            raise ValueError("Invalid cancellation event.")

    # ========================================================
    # Execution tracing
    # ========================================================

    def _record_trace(
        self,
        *,
        request: ActionRequest,
        status: str,
        started_at: float,
        error_code: str | None = None,
        risk_level: str = "unknown",
        confirmation_required: bool = False,
    ) -> dict[str, Any]:

        event = {
            "tool": request.tool_name,
            "session_id": request.session_id,
            "interface": request.interface,
            "source": request.source,
            "call_id": request.call_id,
            "trace_id": request.trace_id,
            "status": status,
            "risk_level": risk_level,
            "confirmation_required": confirmation_required,
            "duration_ms": max(
                0,
                int(
                    (time.monotonic() - started_at) * 1000
                ),
            ),
            "error_code": error_code,
            "timestamp": time.time(),
        }

        if self.audit_logger is not None:
            try:
                self.audit_logger.record_event({
                    **event,
                    "event_type": "action_result",
                })

            except ToolAuditError as exc:
                logger.error(
                    "Action result audit write failed (%s).",
                    type(exc).__name__,
                )

        with self._lock:
            self._trace_events.append(event)

        if self._trace_callback is not None:
            try:
                self._trace_callback(dict(event))
            except Exception:
                logger.exception(
                    "Action trace callback failed."
                )

        return event

    def trace_snapshot(self) -> list[dict[str, Any]]:
        """Return a defensive copy of recent execution metadata."""

        with self._lock:
            return [
                dict(item)
                for item in self._trace_events
            ]

    def acceptance_report(self) -> dict[str, Any]:
        """Return static tool-system acceptance without executing tools."""

        from app.ai.tools.acceptance import (
            ToolAcceptanceEvaluator,
        )

        return ToolAcceptanceEvaluator().evaluate(self)
    
    # ========================================================
    # Confirmation lifecycle
    # ========================================================

    def _purge_pending_locked(self) -> int:

        now = time.monotonic()

        expired = [
            key
            for key, pending in self._pending.items()
            if pending.expires_at <= now
        ]

        for key in expired:
            pending = self._pending.pop(key)

            self.confirmation_manager.cancel_confirmation(
                token=pending.secret_token,
                session_id=pending.session_id,
            )

        return len(expired)

    def _create_confirmation(
        self,
        request: ActionRequest,
        arguments: dict[str, Any],
        *,
        risk_level: str,
        summary: str,
    ) -> dict[str, Any]:

        if (
            not isinstance(summary, str)
            or not summary.strip()
            or len(summary) > self.MAX_PREVIEW_LENGTH
        ):
            return self._result(
                request.tool_name,
                "failed",
                "A safe confirmation preview could not be created.",
                error_code="invalid_confirmation_preview",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

        with self._lock:
            self._purge_pending_locked()

            if len(self._pending) >= self.MAX_PENDING_CONFIRMATIONS:
                return self._result(
                    request.tool_name,
                    "failed",
                    "Confirmation capacity is temporarily full.",
                    error_code="confirmation_capacity_reached",
                    trace_id=request.trace_id,
                    call_id=request.call_id,
                )

            session_count = sum(
                item.session_id == request.session_id
                for item in self._pending.values()
            )

            if session_count >= self.MAX_PENDING_PER_SESSION:
                return self._result(
                    request.tool_name,
                    "failed",
                    "Too many pending actions in this session.",
                    error_code="session_confirmation_limit",
                    trace_id=request.trace_id,
                    call_id=request.call_id,
                )

            idempotency_key = (
                request.idempotency_key
                or request.call_id
                or uuid.uuid4().hex
            )

            call_id = (
                request.call_id
                or uuid.uuid4().hex
            )

            trace_id = (
                request.trace_id
                or uuid.uuid4().hex
            )

            try:
                challenge = (
                    self.confirmation_manager.request_confirmation(
                        tool_name=request.tool_name,
                        session_id=request.session_id,
                        arguments=arguments,
                        summary=summary.strip(),
                        risk_level=risk_level,
                        ttl_seconds=120,
                    )
                )

            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Confirmation challenge rejected (%s).",
                    type(exc).__name__,
                )

                return self._result(
                    request.tool_name,
                    "failed",
                    "The confirmation request could not be created.",
                    error_code="confirmation_creation_failed",
                    trace_id=trace_id,
                    call_id=call_id,
                )

            confirmation_id = secrets.token_urlsafe(18)

            pending = _PendingAction(
                confirmation_id=confirmation_id,
                secret_token=challenge.token,
                tool_name=request.tool_name,
                arguments=arguments,
                session_id=request.session_id,
                interface=request.interface,
                user_message=request.user_message,
                call_id=call_id,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                summary=summary.strip(),
                risk_level=risk_level,
                expires_at=(
                    time.monotonic()
                    + challenge.expires_in_seconds
                ),
            )

            self._pending[confirmation_id] = pending

        return self._result(
            request.tool_name,
            "needs_confirmation",
            "Explicit approval is required before this action executes.",
            data={
                "confirmation_id": confirmation_id,
                "summary": pending.summary,
                "risk_level": risk_level,
                "expires_in_seconds": challenge.expires_in_seconds,
                "outcome": "not_executed",
            },
            error_code="confirmation_required",
            trace_id=trace_id,
            call_id=call_id,
        )

    def confirm_pending(
        self,
        *,
        session_id: str,
        confirmation_id: str,
        interface: str,
        approved: bool,
        cancel_event: Any | None = None,
    ) -> dict[str, Any]:
        """
        Approve or reject one pending action.

        The trusted interface must call this method only after
        receiving the user's explicit confirmation.
        """

        if (
            not isinstance(session_id, str)
            or not isinstance(confirmation_id, str)
            or interface not in SUPPORTED_INTERFACES
            or not isinstance(approved, bool)
        ):
            return self._result(
                "action_confirmation",
                "denied",
                "Invalid confirmation request.",
                error_code="invalid_confirmation_request",
            )

        with self._lock:
            self._purge_pending_locked()

            pending = self._pending.get(
                confirmation_id
            )

        if pending is None:
            return self._result(
                "action_confirmation",
                "not_found",
                "The requested confirmation is no longer available.",
                error_code="confirmation_not_found",
            )

        if (
            pending.session_id != session_id
            or pending.interface != interface
        ):
            return self._result(
                pending.tool_name,
                "denied",
                "This confirmation belongs to another session or interface.",
                error_code="confirmation_scope_mismatch",
            )

        if not approved:
            return self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

        if (
            cancel_event is not None
            and cancel_event.is_set()
        ):
            return self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

        spec = self.registry.get(pending.tool_name)

        if spec is None:
            self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

            return self._result(
                pending.tool_name,
                "not_found",
                "The registered tool is no longer available.",
                error_code="tool_unavailable",
            )

        decision = self.permission_policy.authorize(
            pending.tool_name,
            interface=interface,
        )

        if not decision.allowed:
            self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

            return self._result(
                pending.tool_name,
                "denied",
                decision.message,
                error_code=decision.code,
                trace_id=pending.trace_id,
                call_id=pending.call_id,
            )

        if not self._write_audit_event(
            {
                "event_type": "confirmation_approved",
                "tool": pending.tool_name,
                "session_id": session_id,
                "interface": interface,
                "source": "explicit_user_confirmation",
                "call_id": pending.call_id,
                "trace_id": pending.trace_id,
                "status": "approved",
                "risk_level": pending.risk_level,
                "permission": "mutate",
                "confirmation_required": True,
            },
            required=True,
        ):
            return self._result(
                pending.tool_name,
                "failed",
                "Approval could not be recorded. No action was executed.",
                error_code="audit_unavailable",
                trace_id=pending.trace_id,
                call_id=pending.call_id,
            )
        
        try:
            validated_arguments = self.registry.validate_arguments(
                pending.tool_name,
                pending.arguments,
            )

            grant = self.confirmation_manager.consume_confirmation(
                token=pending.secret_token,
                tool_name=pending.tool_name,
                session_id=session_id,
                arguments=validated_arguments,
            )

        except (TypeError, ValueError) as exc:
            logger.warning(
                "Action confirmation validation failed (%s).",
                type(exc).__name__,
            )

            self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

            return self._result(
                pending.tool_name,
                "denied",
                "The confirmation could not be validated.",
                error_code="confirmation_validation_failed",
                trace_id=pending.trace_id,
                call_id=pending.call_id,
            )

        # A consumed ticket cannot be used again.
        with self._lock:
            self._pending.pop(
                confirmation_id,
                None,
            )

        started = time.monotonic()

        result = self.execution_runtime.execute(
            pending.tool_name,
            validated_arguments,
            session_id=session_id,
            interface=interface,
            idempotency_key=pending.idempotency_key,
            call_id=pending.call_id,
            trace_id=pending.trace_id,
            cancel_event=cancel_event,
            confirmation_grant=grant,
        )

        self._record_trace(
            request=ActionRequest(
                tool_name=pending.tool_name,
                arguments=validated_arguments,
                session_id=session_id,
                interface=interface,
                source="internal",
                call_id=pending.call_id,
                trace_id=pending.trace_id,
            ),
            status=str(result.get("status", "error")),
            started_at=started,
            error_code=result.get("error_code"),
            risk_level=pending.risk_level,
        )

        return result

    def cancel_pending(
        self,
        *,
        session_id: str,
        confirmation_id: str,
        interface: str,
    ) -> dict[str, Any]:

        with self._lock:
            pending = self._pending.get(
                confirmation_id
            )

            if pending is None:
                return self._result(
                    "action_confirmation",
                    "not_found",
                    "The confirmation is no longer available.",
                    error_code="confirmation_not_found",
                )

            if (
                pending.session_id != session_id
                or pending.interface != interface
            ):
                return self._result(
                    pending.tool_name,
                    "denied",
                    "Confirmation ownership does not match.",
                    error_code="confirmation_scope_mismatch",
                )

            self._pending.pop(
                confirmation_id,
                None,
            )

        self.confirmation_manager.cancel_confirmation(
            token=pending.secret_token,
            session_id=session_id,
        )

        self._write_audit_event(
            {
                "event_type": "action_cancelled",
                "tool": pending.tool_name,
                "session_id": session_id,
                "interface": interface,
                "source": "user_cancellation",
                "call_id": pending.call_id,
                "trace_id": pending.trace_id,
                "status": "cancelled",
                "risk_level": pending.risk_level,
                "permission": "mutate",
                "confirmation_required": True,
            },
            required=False,
        )

        return self._result(
            pending.tool_name,
            "cancelled",
            "The action was cancelled. No operation was executed.",
            data={"outcome": "not_executed"},
            error_code="action_cancelled",
            trace_id=pending.trace_id,
            call_id=pending.call_id,
        )

    def confirm_latest_pending(
        self,
        *,
        session_id: str,
        interface: str,
        cancel_event: Any | None = None,
    ) -> dict[str, Any] | None:

        with self._lock:
            self._purge_pending_locked()

            pending = [
                item
                for item in self._pending.values()
                if (
                    item.session_id == session_id
                    and item.interface == interface
                )
            ]

        if not pending:
            return None

        if len(pending) != 1:
            return self._result(
                "action_confirmation",
                "needs_confirmation",
                (
                    "Multiple actions are awaiting approval. "
                    "Use the relevant confirmation reference."
                ),
                data={
                    "pending_confirmation_ids": [
                        item.confirmation_id
                        for item in pending
                    ],
                },
                error_code="ambiguous_pending_confirmation",
            )

        return self.confirm_pending(
            session_id=session_id,
            confirmation_id=pending[0].confirmation_id,
            interface=interface,
            approved=True,
            cancel_event=cancel_event,
        )

    def cancel_latest_pending(
        self,
        *,
        session_id: str,
        interface: str,
    ) -> dict[str, Any] | None:

        with self._lock:
            self._purge_pending_locked()

            pending_ids = [
                item.confirmation_id
                for item in self._pending.values()
                if (
                    item.session_id == session_id
                    and item.interface == interface
                )
            ]

        if not pending_ids:
            return None

        cancelled = 0

        for confirmation_id in pending_ids:
            result = self.cancel_pending(
                session_id=session_id,
                confirmation_id=confirmation_id,
                interface=interface,
            )

            if result["status"] == "cancelled":
                cancelled += 1

        return self._result(
            "action_confirmation",
            "cancelled",
            f"Cancelled {cancelled} pending action(s).",
            data={
                "cancelled_count": cancelled,
                "outcome": "not_executed",
            },
        )

    def cancel_session(
        self,
        session_id: str,
    ) -> int:
        """Atomically remove pending confirmations for one session."""

        if not isinstance(session_id, str):
            return 0

        with self._lock:
            matching = [
                item
                for item in self._pending.values()
                if item.session_id == session_id
            ]

            for item in matching:
                self._pending.pop(
                    item.confirmation_id,
                    None,
                )

        for item in matching:
            self.confirmation_manager.cancel_confirmation(
                token=item.secret_token,
                session_id=session_id,
            )

        self.confirmation_manager.cancel_session(
            session_id
        )

        return len(matching)

    # ========================================================
    # Unified dispatch
    # ========================================================

    def dispatch(
        self,
        request: ActionRequest,
    ) -> dict[str, Any]:
        """
        Dispatch a tool request through the common application boundary.

        Existing database tools retain their established service and
        confirmation workflow. New registered tools use ExecutionRuntime.
        """

        started = time.monotonic()

        try:
            self._validate_request(request)

        except (TypeError, ValueError) as exc:
            return self._result(
                "action_router",
                "failed",
                "The action request is invalid.",
                error_code="invalid_action_request",
            )

        spec = self.registry.get(
            request.tool_name
        )

        if spec is None:
            result = self._result(
                request.tool_name,
                "not_found",
                "The requested tool is not registered.",
                error_code="unknown_tool",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

            self._record_trace(
                request=request,
                status="not_found",
                started_at=started,
                error_code="unknown_tool",
            )

            return result

        # Validate every model-originated argument before preview or execution.
        try:
            arguments = self.registry.validate_arguments(
                request.tool_name,
                request.arguments,
            )

        except (TypeError, ValueError):
            result = self._result(
                request.tool_name,
                "failed",
                "Tool arguments failed schema validation.",
                error_code="invalid_tool_arguments",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

            self._record_trace(
                request=request,
                status="failed",
                started_at=started,
                error_code="invalid_tool_arguments",
                risk_level=spec.metadata.risk_level,
            )

            return result

        request_guard = self._request_guards.get(
            spec.metadata.category
        )

        if request_guard is not None:
            try:
                intent_allowed = (
                    request_guard(
                        request,
                        arguments,
                    )
                    is True
                )

            except Exception as exc:
                logger.warning(
                    "Action request guard failed (%s).",
                    type(exc).__name__,
                )
                intent_allowed = False

            if not intent_allowed:
                result = self._result(
                    request.tool_name,
                    "denied",
                    (
                        "User message mein is exact system action ki "
                        "clear permission identify nahi hui. "
                        "Koi system operation execute nahi hua."
                    ),
                    error_code="system_action_intent_not_explicit",
                    trace_id=request.trace_id,
                    call_id=request.call_id,
                )

                self._record_trace(
                    request=request,
                    status="denied",
                    started_at=started,
                    error_code="system_action_intent_not_explicit",
                    risk_level=spec.metadata.risk_level,
                )

                return result
            
        decision = self.permission_policy.authorize(
            request.tool_name,
            interface=request.interface,
        )

        if not decision.allowed:
            result = self._result(
                request.tool_name,
                "denied",
                decision.message,
                error_code=decision.code,
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

            self._record_trace(
                request=request,
                status="denied",
                started_at=started,
                error_code=decision.code,
                risk_level=spec.metadata.risk_level,
            )

            return result

        # ----------------------------------------------------
        # T5.5: mandatory pre-execution audit
        # ----------------------------------------------------

        audit_required = (
            spec.permission == "mutate"
            or decision.risk_level in {
                "medium",
                "high",
                "critical",
            }
        )

        audit_event = {
            "event_type": "action_authorized",
            "tool": request.tool_name,
            "session_id": request.session_id,
            "interface": request.interface,
            "source": request.source,
            "call_id": request.call_id,
            "trace_id": request.trace_id,
            "status": "authorized",
            "risk_level": decision.risk_level,
            "permission": spec.permission,
            "confirmation_required": decision.requires_confirmation,
        }

        if not self._write_audit_event(
            audit_event,
            required=audit_required,
        ):
            return self._result(
                request.tool_name,
                "failed",
                (
                    "Audit storage unavailable. "
                    "The requested operation was not started."
                ),
                error_code="audit_unavailable",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )
        
        # ----------------------------------------------------
        # Existing database service — preserve old behavior.
        # ----------------------------------------------------

        if request.tool_name in self.legacy_tool_names:

            if self.database_service is None:
                result = self._result(
                    request.tool_name,
                    "failed",
                    "The database tool service is unavailable.",
                    error_code="database_service_unavailable",
                    trace_id=request.trace_id,
                    call_id=request.call_id,
                )

            else:
                try:
                    result = self.database_service.execute_tool(
                        request.tool_name,
                        arguments,
                        request.user_message,
                    )

                    if not isinstance(result, dict):
                        raise TypeError(
                            "Database service returned a non-object."
                        )

                    result = self.registry.validate_result(
                        request.tool_name,
                        result,
                    )

                except Exception as exc:
                    logger.error(
                        "Legacy action dispatch failed (%s).",
                        type(exc).__name__,
                    )

                    result = self._result(
                        request.tool_name,
                        "error",
                        (
                            "The existing application tool could not "
                            "verify its execution result."
                        ),
                        error_code="legacy_tool_execution_failed",
                    )

            self._record_trace(
                request=request,
                status=str(result.get("status", "error")),
                started_at=started,
                error_code=result.get("error_code"),
                risk_level=spec.metadata.risk_level,
                confirmation_required=(
                    result.get("status") == "needs_confirmation"
                ),
            )

            return result

        # ----------------------------------------------------
        # New application tool handler.
        # ----------------------------------------------------

        if request.tool_name not in self._generic_tools:
            result = self._result(
                request.tool_name,
                "failed",
                "No trusted application handler is registered.",
                error_code="handler_not_registered",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

            self._record_trace(
                request=request,
                status="failed",
                started_at=started,
                error_code="handler_not_registered",
                risk_level=spec.metadata.risk_level,
            )

            return result

        if request.cancel_event is not None and request.cancel_event.is_set():
            return self._result(
                request.tool_name,
                "cancelled",
                "The action was cancelled before execution.",
                error_code="cancelled_before_dispatch",
                trace_id=request.trace_id,
                call_id=request.call_id,
            )

        if decision.requires_confirmation:

            preview_handler = self._confirmation_previews.get(
                request.tool_name
            )

            if preview_handler is None:
                result = self._result(
                    request.tool_name,
                    "failed",
                    (
                        "This tool cannot be executed because "
                        "a trusted confirmation preview is missing."
                    ),
                    error_code="confirmation_preview_required",
                )

                self._record_trace(
                    request=request,
                    status="failed",
                    started_at=started,
                    error_code="confirmation_preview_required",
                    risk_level=decision.risk_level,
                    confirmation_required=True,
                )

                return result

            try:
                summary = preview_handler(
                    dict(arguments)
                )

            except Exception as exc:
                logger.warning(
                    "Confirmation preview generation failed (%s).",
                    type(exc).__name__,
                )

                summary = None

            result = self._create_confirmation(
                request,
                arguments,
                risk_level=decision.risk_level,
                summary=summary,
            )

            self._record_trace(
                request=request,
                status=str(result["status"]),
                started_at=started,
                error_code=result.get("error_code"),
                risk_level=decision.risk_level,
                confirmation_required=True,
            )

            return result

        call_id = (
            request.call_id
            or uuid.uuid4().hex
        )

        trace_id = (
            request.trace_id
            or uuid.uuid4().hex
        )

        idempotency_key = (
            request.idempotency_key
            or request.call_id
            or uuid.uuid4().hex
        )

        result = self.execution_runtime.execute(
            request.tool_name,
            arguments,
            session_id=request.session_id,
            interface=request.interface,
            idempotency_key=idempotency_key,
            call_id=call_id,
            trace_id=trace_id,
            cancel_event=request.cancel_event,
        )

        self._record_trace(
            request=ActionRequest(
                tool_name=request.tool_name,
                arguments=arguments,
                session_id=request.session_id,
                interface=request.interface,
                source=request.source,
                call_id=call_id,
                trace_id=trace_id,
            ),
            status=str(result.get("status", "error")),
            started_at=started,
            error_code=result.get("error_code"),
            risk_level=spec.metadata.risk_level,
        )

        return result

    # ========================================================
    # User-facing rendering
    # ========================================================

    def render_result(
        self,
        result: dict[str, Any],
        *,
        tool_name: str | None = None,
    ) -> str:
        """Render a safe user-facing result without exposing secret tokens."""

        if not isinstance(result, dict):
            return "Action ka result valid nahi hai."

        status = result.get("status")

        if (
            tool_name in self.legacy_tool_names
            and self.database_service is not None
        ):
            return self.database_service.render_result(
                result
            )

        if status == "needs_confirmation":
            data = result.get("data", {})

            if not isinstance(data, dict):
                data = {}

            summary = str(
                data.get("summary") or "Requested action"
            )

            confirmation_id = data.get(
                "confirmation_id"
            )

            return (
                f"{summary}\n"
                f"Risk: {data.get('risk_level', 'unknown')}\n"
                f"Confirmation reference: {confirmation_id}\n"
                "Action abhi execute nahi hua hai. Approve karne ke liye "
                "'confirm' aur cancel karne ke liye 'cancel' kahein."
            )

        message = result.get("message")

        if isinstance(message, str) and message.strip():
            return message.strip()

        return "Action ka verified result available nahi hai."