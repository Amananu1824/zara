"""Shared, versioned contracts for Zara tool execution."""

from __future__ import annotations

import json
import re

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


ToolStatus = Literal[
    "success",
    "error",
    "failed",
    "denied",
    "not_found",
    "needs_confirmation",
    "unchanged",
    "cancelled",
]


TOOL_RESULT_STATUSES = frozenset({
    "success",
    "error",
    "failed",
    "denied",
    "not_found",
    "needs_confirmation",
    "unchanged",
    "cancelled",
})


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Canonical result returned by an application tool.

    `to_dict()` provides the new structured contract.
    `to_legacy_dict()` preserves compatibility with existing
    database-tool consumers.
    """

    tool_name: str
    status: ToolStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    error_code: str | None = None
    retryable: bool = False

    duration_ms: int | None = None
    trace_id: str | None = None
    call_id: str | None = None

    schema_version: int = 1

    MAX_DATA_BYTES = 32768
    MAX_MESSAGE_LENGTH = 4000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tool_name, str)
            or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}",
                self.tool_name,
            )
        ):
            raise ValueError("Invalid ToolResult tool_name.")

        if (
            not isinstance(self.status, str)
            or self.status not in TOOL_RESULT_STATUSES
        ):
            raise ValueError("Unsupported ToolResult status.")

        if not isinstance(self.message, str):
            raise TypeError("ToolResult message must be text.")

        message = self.message.strip()

        if not message:
            raise ValueError("ToolResult message cannot be empty.")

        if len(message) > self.MAX_MESSAGE_LENGTH:
            raise ValueError("ToolResult message is too long.")

        if not isinstance(self.data, dict):
            raise TypeError("ToolResult data must be an object.")

        if (
            self.error_code is not None
            and (
                not isinstance(self.error_code, str)
                or len(self.error_code) > 128
            )
        ):
            raise TypeError("Invalid ToolResult error_code.")

        if not isinstance(self.retryable, bool):
            raise TypeError("ToolResult retryable must be boolean.")

        if (
            self.duration_ms is not None
            and (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, int)
                or self.duration_ms < 0
            )
        ):
            raise ValueError("Invalid ToolResult duration_ms.")

        for value, field_name in (
            (self.trace_id, "trace_id"),
            (self.call_id, "call_id"),
        ):
            if (
                value is not None
                and (
                    not isinstance(value, str)
                    or len(value) > 128
                )
            ):
                raise ValueError(
                    f"Invalid ToolResult {field_name}."
                )

        if self.schema_version != 1:
            raise ValueError("Unsupported ToolResult version.")

        try:
            serialized = json.dumps(
                self.data,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                "ToolResult data must be JSON-compatible."
            ) from exc

        if len(serialized) > self.MAX_DATA_BYTES:
            raise ValueError("ToolResult data exceeds its size limit.")

        object.__setattr__(self, "message", message)
        object.__setattr__(self, "data", deepcopy(self.data))

    @classmethod
    def from_mapping(
        cls,
        tool_name: str,
        result: dict[str, Any],
    ) -> "ToolResult":
        """Convert an existing tool dictionary into ToolResult."""

        if not isinstance(result, dict):
            raise TypeError("Tool result must be an object.")

        supplied_name = result.get("tool_name")

        if supplied_name is not None and supplied_name != tool_name:
            raise ValueError("Tool result name mismatch.")

        version = result.get("schema_version", 1)

        if isinstance(version, bool) or version != 1:
            raise ValueError("Unsupported tool result schema version.")

        raw_status = result.get("status")

        if not isinstance(raw_status, str):
            raise ValueError("Tool result status is missing.")

        raw_message = result.get("message")

        if raw_message is None:
            raw_message = (
                f"Tool '{tool_name}' returned status '{raw_status}'."
            )

        if not isinstance(raw_message, str):
            raise TypeError("Tool result message must be text.")

        nested_data = result.get("data", {})

        if nested_data is None:
            nested_data = {}

        if not isinstance(nested_data, dict):
            raise TypeError("Tool result data must be an object.")

        reserved = {
            "schema_version",
            "tool_name",
            "status",
            "message",
            "data",
            "error_code",
            "retryable",
            "duration_ms",
            "trace_id",
            "call_id",
        }

        data = deepcopy(nested_data)

        for key, value in result.items():
            if key in reserved:
                continue

            if key in data and data[key] != value:
                raise ValueError(
                    "Conflicting tool result data fields."
                )

            data.setdefault(key, deepcopy(value))

        raw_duration = result.get(
            "duration_ms",
            data.get("duration_ms"),
        )

        return cls(
            tool_name=tool_name,
            status=raw_status,
            message=raw_message,
            data=data,
            error_code=result.get("error_code"),
            retryable=result.get("retryable", False),
            duration_ms=raw_duration,
            trace_id=result.get("trace_id"),
            call_id=result.get("call_id"),
            schema_version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical nested result contract."""

        return {
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "status": self.status,
            "message": self.message,
            "data": deepcopy(self.data),
            "error_code": self.error_code,
            "retryable": self.retryable,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "call_id": self.call_id,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """
        Return a backwards-compatible flattened result.

        Existing database handlers expose fields such as
        record, records, operation and outcome at the top level.
        """

        result = deepcopy(self.data)

        result.update({
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "status": self.status,
            "message": self.message,
            "error_code": self.error_code,
            "retryable": self.retryable,
        })

        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        if self.trace_id is not None:
            result["trace_id"] = self.trace_id

        if self.call_id is not None:
            result["call_id"] = self.call_id

        return result