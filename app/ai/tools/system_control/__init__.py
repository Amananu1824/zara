"""System Control A1 public module."""

from app.ai.tools.system_control.definitions import (
    SYSTEM_CONTROL_MUTATION_SCOPES,
    SYSTEM_CONTROL_READ_SCOPES,
    SYSTEM_CONTROL_TOOL_NAMES,
    register_system_control_tools,
)

from app.ai.tools.system_control.intent import (
    detect_system_control_intent,
)

from app.ai.tools.system_control.service import (
    SystemControlError,
    SystemControlService,
)


__all__ = [
    "SYSTEM_CONTROL_MUTATION_SCOPES",
    "SYSTEM_CONTROL_READ_SCOPES",
    "SYSTEM_CONTROL_TOOL_NAMES",
    "SystemControlError",
    "SystemControlService",
    "detect_system_control_intent",
    "register_system_control_tools",
]