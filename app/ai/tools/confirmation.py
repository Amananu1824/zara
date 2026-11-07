"""Short-lived, action-bound confirmation manager for Zara."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time

from dataclasses import dataclass
from threading import RLock
from typing import Any


class ConfirmationError(ValueError):
    """An approval is invalid, expired or bound to another action."""


@dataclass(frozen=True, slots=True)
class ConfirmationChallenge:
    """Internal approval request created by a trusted tool handler."""

    token: str
    tool_name: str
    session_id: str
    summary: str
    risk_level: str
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class ConfirmationGrant:
    """One-time authorization for one exact tool invocation."""

    tool_name: str
    session_id: str
    arguments_digest: str
    approved_at: float


@dataclass(frozen=True, slots=True)
class _PendingConfirmation:
    tool_name: str
    session_id: str
    arguments_digest: str
    risk_level: str
    expires_at: float


class ToolConfirmationManager:
    """
    In-memory, one-time confirmation tickets.

    Approval is bound to:
      - exact registered tool name
      - exact normalized arguments
      - originating session
      - short expiry period

    The model must never approve or consume its own confirmation.
    """

    MAX_PENDING = 256
    MAX_PENDING_PER_SESSION = 8
    MAX_ARGUMENT_BYTES = 16384
    MAX_SUMMARY_LENGTH = 500

    MIN_TTL_SECONDS = 15
    MAX_TTL_SECONDS = 300

    TOOL_NAME_PATTERN = re.compile(
        r"[a-z][a-z0-9_]{0,63}"
    )

    def __init__(self) -> None:
        self._pending: dict[
            str,
            _PendingConfirmation,
        ] = {}

        self._lock = RLock()

    @classmethod
    def _validate_identity(
        cls,
        tool_name: str,
        session_id: str,
    ) -> None:

        if (
            not isinstance(tool_name, str)
            or not cls.TOOL_NAME_PATTERN.fullmatch(tool_name)
        ):
            raise ConfirmationError(
                "Invalid confirmation tool identity."
            )

        if (
            not isinstance(session_id, str)
            or not session_id.strip()
            or len(session_id) > 128
        ):
            raise ConfirmationError(
                "Invalid confirmation session."
            )

    @classmethod
    def _arguments_digest(
        cls,
        arguments: dict[str, Any],
    ) -> str:
        """Hash canonical JSON without storing raw arguments."""

        if not isinstance(arguments, dict):
            raise ConfirmationError(
                "Confirmation arguments must be an object."
            )

        try:
            serialized = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")

        except (TypeError, ValueError, RecursionError) as exc:
            raise ConfirmationError(
                "Arguments cannot be safely fingerprinted."
            ) from exc

        if len(serialized) > cls.MAX_ARGUMENT_BYTES:
            raise ConfirmationError(
                "Confirmation arguments exceed the size limit."
            )

        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _token_digest(token: str) -> str:
        return hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

    def _purge_expired_locked(self) -> int:
        now = time.monotonic()

        expired = [
            key
            for key, pending in self._pending.items()
            if pending.expires_at <= now
        ]

        for key in expired:
            self._pending.pop(key, None)

        return len(expired)

    def purge_expired(self) -> int:
        """Remove expired confirmation tickets."""

        with self._lock:
            return self._purge_expired_locked()

    def request_confirmation(
        self,
        *,
        tool_name: str,
        session_id: str,
        arguments: dict[str, Any],
        summary: str,
        risk_level: str,
        ttl_seconds: int = 120,
    ) -> ConfirmationChallenge:
        """Create a short-lived ticket for one exact action."""

        self._validate_identity(
            tool_name,
            session_id,
        )

        if (
            not isinstance(summary, str)
            or not summary.strip()
            or len(summary) > self.MAX_SUMMARY_LENGTH
            or "\x00" in summary
        ):
            raise ConfirmationError(
                "Invalid confirmation summary."
            )

        if risk_level not in {
            "low",
            "medium",
            "high",
            "critical",
        }:
            raise ConfirmationError(
                "Invalid confirmation risk level."
            )

        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or not (
                self.MIN_TTL_SECONDS
                <= ttl_seconds
                <= self.MAX_TTL_SECONDS
            )
        ):
            raise ConfirmationError(
                "Confirmation lifetime is outside allowed limits."
            )

        args_digest = self._arguments_digest(arguments)

        token = secrets.token_urlsafe(32)
        token_digest = self._token_digest(token)

        with self._lock:
            self._purge_expired_locked()

            if len(self._pending) >= self.MAX_PENDING:
                raise ConfirmationError(
                    "Confirmation capacity is temporarily full."
                )

            session_count = sum(
                pending.session_id == session_id
                for pending in self._pending.values()
            )

            if session_count >= self.MAX_PENDING_PER_SESSION:
                raise ConfirmationError(
                    "Too many pending actions in this session."
                )

            self._pending[token_digest] = _PendingConfirmation(
                tool_name=tool_name,
                session_id=session_id,
                arguments_digest=args_digest,
                risk_level=risk_level,
                expires_at=(
                    time.monotonic() + ttl_seconds
                ),
            )

        return ConfirmationChallenge(
            token=token,
            tool_name=tool_name,
            session_id=session_id,
            summary=summary.strip(),
            risk_level=risk_level,
            expires_in_seconds=ttl_seconds,
        )

    def consume_confirmation(
        self,
        *,
        token: str,
        tool_name: str,
        session_id: str,
        arguments: dict[str, Any],
    ) -> ConfirmationGrant:
        """
        Consume a ticket exactly once.

        Call this only after the original interface has received
        the user's explicit approval.
        """

        self._validate_identity(
            tool_name,
            session_id,
        )

        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 128
        ):
            raise ConfirmationError(
                "Invalid or expired confirmation."
            )

        token_digest = self._token_digest(token)
        args_digest = self._arguments_digest(arguments)

        with self._lock:
            self._purge_expired_locked()

            # Pop first: even a mismatched approval cannot be replayed.
            pending = self._pending.pop(
                token_digest,
                None,
            )

        if pending is None:
            raise ConfirmationError(
                "Confirmation is invalid, expired or already used."
            )

        if not secrets.compare_digest(
            pending.tool_name,
            tool_name,
        ):
            raise ConfirmationError(
                "Confirmation belongs to another tool."
            )

        if not secrets.compare_digest(
            pending.session_id,
            session_id,
        ):
            raise ConfirmationError(
                "Confirmation belongs to another session."
            )

        if not secrets.compare_digest(
            pending.arguments_digest,
            args_digest,
        ):
            raise ConfirmationError(
                "The approved action parameters have changed."
            )

        return ConfirmationGrant(
            tool_name=tool_name,
            session_id=session_id,
            arguments_digest=args_digest,
            approved_at=time.time(),
        )

    def cancel_confirmation(
        self,
        *,
        token: str,
        session_id: str,
    ) -> bool:
        """Cancel one pending confirmation belonging to a session."""

        if (
            not isinstance(token, str)
            or not isinstance(session_id, str)
        ):
            return False

        token_digest = self._token_digest(token)

        with self._lock:
            pending = self._pending.get(token_digest)

            if pending is None:
                return False

            if not secrets.compare_digest(
                pending.session_id,
                session_id,
            ):
                return False

            self._pending.pop(token_digest, None)
            return True

    def cancel_session(
        self,
        session_id: str,
    ) -> int:
        """Invalidate all pending actions when a session ends."""

        if not isinstance(session_id, str):
            return 0

        with self._lock:
            matching = [
                key
                for key, pending in self._pending.items()
                if pending.session_id == session_id
            ]

            for key in matching:
                self._pending.pop(key, None)

            return len(matching)

    @property
    def pending_count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._pending)