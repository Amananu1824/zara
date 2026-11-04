from dataclasses import dataclass, field
from typing import Any, Literal


TaskStatus = Literal[
    "success",
    "failed",
    "not_found",
    "needs_confirmation",
    "unchanged",
    "denied",
    "cancelled",
]

_ALLOWED_STATUSES = frozenset({
    "success",
    "failed",
    "not_found",
    "needs_confirmation",
    "unchanged",
    "denied",
    "cancelled",
})

_DATABASE_STATUS_MAP: dict[str, TaskStatus] = {
    "success": "success",
    "error": "failed",
    "failed": "failed",
    "not_found": "not_found",
    "needs_confirmation": "needs_confirmation",
    "unchanged": "unchanged",
    "denied": "denied",
    "cancelled": "cancelled",
}


def normalize_task_status(status: object) -> TaskStatus:
    """Convert database-operation statuses to the shared task contract."""
    if not isinstance(status, str):
        return "failed"

    return _DATABASE_STATUS_MAP.get(
        status.strip().casefold(),
        "failed",
    )


@dataclass(frozen=True, slots=True)
class TaskResult:
    operation: str
    status: TaskStatus
    system_message: str
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError(
                "TaskResult operation must be a non-empty string."
            )

        if (
            not isinstance(self.status, str)
            or self.status not in _ALLOWED_STATUSES
        ):
            raise ValueError(
                f"Unsupported task status: {self.status!r}"
            )

        if not isinstance(self.system_message, str):
            raise TypeError(
                "TaskResult system_message must be a string."
            )

        if not isinstance(self.data, dict):
            raise TypeError(
                "TaskResult data must be a dictionary."
            )

        if (
            self.error_code is not None
            and not isinstance(self.error_code, str)
        ):
            raise TypeError(
                "TaskResult error_code must be a string or None."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operation": self.operation,
            "status": self.status,
            "success": self.status == "success",
            "system_message": self.system_message,
            "data": dict(self.data),
            "error_code": self.error_code,
        }