"""Registered contracts for Zara's System Control tools."""

from __future__ import annotations

from threading import RLock
from typing import Any

from app.ai.tools.registry import ToolMetadata, ToolRegistry


SYSTEM_CONTROL_INTERFACES = (
    "terminal",
    "api",
    "browser",
    "desktop",
    "voice",
    "internal",
)

SYSTEM_CONTROL_READ_SCOPES = frozenset({
    "system.read",
})

SYSTEM_CONTROL_MUTATION_SCOPES = frozenset({
    "system.audio.write",
    "system.display.write",
    "system.lock",
    "system.power",
    "system.network.wake",
})

SYSTEM_CONTROL_TOOL_NAMES = (
    "system_info",
    "get_current_time",
    "get_current_date",
    "battery_status",
    "storage_monitor",
    "process_monitor",
    "volume_control",
    "brightness_control",
    "power_off",
    "restart_pc",
    "lock_pc",
    "wake_on_lan",
)

_REGISTRATION_LOCK = RLock()


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build one strict function-tool definition."""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


SYSTEM_CONTROL_DEFINITIONS = [
    _tool(
        "system_info",
        (
            "Read the current local computer's OS, architecture, processor, "
            "CPU core counts and installed RAM. Use only when the user asks "
            "for their own PC/system specifications or current system details. "
            "A casual mention of a PC, CPU or operating system is not a request."
        ),
    ),

    _tool(
        "get_current_time",
        (
            "Read the current local system time and timezone. Use when the "
            "user asks what the current local time is. Do not use for time "
            "zones in other cities, time calculations, fictional examples "
            "or general discussion about time."
        ),
    ),

    _tool(
        "get_current_date",
        (
            "Read the current local system date. Use when the user asks "
            "for today's/current local date, not when the date is merely "
            "mentioned in a sentence or example."
        ),
    ),

    _tool(
        "battery_status",
        (
            "Read the local device battery percentage, charging state and "
            "estimated remaining time. Use only when the user asks for the "
            "current battery status of this device. Do not infer a request "
            "from a passing mention of a battery."
        ),
    ),

    _tool(
        "storage_monitor",
        (
            "Read storage capacity, used space, free space and percentage "
            "for local mounted drives. Do not read file contents, delete "
            "files or modify partitions."
        ),
    ),

    _tool(
        "process_monitor",
        (
            "Read a limited list of currently running process names, PIDs, "
            "statuses and memory usage. Never terminate, suspend, modify "
            "or inject into a process. Do not return command lines, user "
            "names, environment variables or process memory contents."
        ),
        properties={
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 15,
            },
        },
    ),

    _tool(
        "volume_control",
        (
            "Change the current default audio output volume. Supported "
            "actions are set, increase, decrease, mute and unmute. "
            "Use only when the current user message clearly requests an "
            "audio-volume change. A discussion or complaint about sound "
            "without a requested change is not permission."
        ),
        properties={
            "action": {
                "type": "string",
                "enum": [
                    "set",
                    "increase",
                    "decrease",
                    "mute",
                    "unmute",
                ],
            },
            "level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "step": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
            },
        },
        required=["action"],
    ),

    _tool(
        "brightness_control",
        (
            "Change the brightness of a supported local display. Supported "
            "actions are set, increase and decrease. Use only when the user "
            "clearly requests a brightness change. Do not infer permission "
            "from a general discussion about screens or brightness."
        ),
        properties={
            "action": {
                "type": "string",
                "enum": [
                    "set",
                    "increase",
                    "decrease",
                ],
            },
            "level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "step": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
            },
            "display": {
                "type": "integer",
                "minimum": 0,
                "maximum": 8,
            },
        },
        required=["action"],
    ),

    _tool(
        "power_off",
        (
            "Request shutdown of this local Windows PC. This is a high-risk "
            "action and requires explicit confirmation of the exact shutdown "
            "request. Never call for hypothetical questions, examples, "
            "translations, future reminders or a casual mention of shutdown."
        ),
    ),

    _tool(
        "restart_pc",
        (
            "Request restart of this local Windows PC. This is a high-risk "
            "action and requires explicit confirmation of the exact restart "
            "request. Never call for hypothetical questions, examples, "
            "translations, future reminders or a casual mention of restart."
        ),
    ),

    _tool(
        "lock_pc",
        (
            "Request locking of the current local Windows interactive session. "
            "Requires explicit confirmation. Do not use when the user is "
            "merely discussing screen locks or account security."
        ),
    ),

    _tool(
        "wake_on_lan",
        (
            "Send a limited number of Wake-on-LAN magic packets to the "
            "explicitly supplied target MAC address over a local/private "
            "broadcast address. Requires confirmation. This sends packets; "
            "it cannot guarantee that the target computer wakes."
        ),
        properties={
            "mac_address": {
                "type": "string",
                "minLength": 12,
                "maxLength": 17,
                "pattern": (
                    r"^(?:[0-9A-Fa-f]{2}[:-]){5}"
                    r"[0-9A-Fa-f]{2}$|^[0-9A-Fa-f]{12}$"
                ),
            },
            "broadcast_address": {
                "type": "string",
                "minLength": 7,
                "maxLength": 15,
            },
        },
        required=["mac_address"],
    ),
]


def _metadata(
    *,
    provider: str,
    risk: str,
    scopes: tuple[str, ...],
    timeout: float,
    idempotent: bool,
) -> ToolMetadata:
    return ToolMetadata(
        category="system_control",
        version="1.0.0",
        provider=provider,
        capabilities=("system_control",),
        supported_interfaces=SYSTEM_CONTROL_INTERFACES,
        risk_level=risk,
        required_scopes=scopes,
        timeout_seconds=timeout,
        max_retries=0,
        idempotent=idempotent,
    )


SYSTEM_CONTROL_POLICIES = {
    "system_info": (
        "read",
        "none",
        _metadata(
            provider="psutil",
            risk="low",
            scopes=("system.read",),
            timeout=10,
            idempotent=True,
        ),
    ),

    "get_current_time": (
        "read",
        "none",
        _metadata(
            provider="python_stdlib",
            risk="low",
            scopes=("system.read",),
            timeout=3,
            idempotent=False,
        ),
    ),

    "get_current_date": (
        "read",
        "none",
        _metadata(
            provider="python_stdlib",
            risk="low",
            scopes=("system.read",),
            timeout=3,
            idempotent=True,
        ),
    ),

    "battery_status": (
        "read",
        "none",
        _metadata(
            provider="psutil",
            risk="low",
            scopes=("system.read",),
            timeout=5,
            idempotent=True,
        ),
    ),

    "storage_monitor": (
        "read",
        "none",
        _metadata(
            provider="psutil",
            risk="low",
            scopes=("system.read",),
            timeout=10,
            idempotent=True,
        ),
    ),

    "process_monitor": (
        "read",
        "none",
        _metadata(
            provider="psutil",
            risk="low",
            scopes=("system.read",),
            timeout=15,
            idempotent=True,
        ),
    ),

    "volume_control": (
        "mutate",
        "none",
        _metadata(
            provider="pycaw",
            risk="low",
            scopes=("system.audio.write",),
            timeout=10,
            idempotent=False,
        ),
    ),

    "brightness_control": (
        "mutate",
        "none",
        _metadata(
            provider="screen_brightness_control",
            risk="low",
            scopes=("system.display.write",),
            timeout=20,
            idempotent=False,
        ),
    ),

    "power_off": (
        "mutate",
        "required",
        _metadata(
            provider="windows_native",
            risk="high",
            scopes=("system.power",),
            timeout=10,
            idempotent=False,
        ),
    ),

    "restart_pc": (
        "mutate",
        "required",
        _metadata(
            provider="windows_native",
            risk="high",
            scopes=("system.power",),
            timeout=10,
            idempotent=False,
        ),
    ),

    "lock_pc": (
        "mutate",
        "required",
        _metadata(
            provider="windows_native",
            risk="medium",
            scopes=("system.lock",),
            timeout=5,
            idempotent=False,
        ),
    ),

    "wake_on_lan": (
        "mutate",
        "required",
        _metadata(
            provider="python_stdlib",
            risk="medium",
            scopes=("system.network.wake",),
            timeout=5,
            idempotent=False,
        ),
    ),
}


def register_system_control_tools(
    registry: ToolRegistry,
) -> None:
    """
    Register the 12 A1 tools into the existing shared registry.

    Safe to call for each ChatEngine instance. Existing matching
    definitions are reused; conflicting registrations fail closed.
    """

    if not isinstance(registry, ToolRegistry):
        raise TypeError("A valid ToolRegistry is required.")

    with _REGISTRATION_LOCK:
        for definition in SYSTEM_CONTROL_DEFINITIONS:
            name = definition["function"]["name"]

            permission, confirmation, metadata = (
                SYSTEM_CONTROL_POLICIES[name]
            )

            existing = registry.get(name)

            if existing is not None:
                if (
                    existing.definition != definition
                    or existing.permission != permission
                    or existing.confirmation != confirmation
                    or existing.metadata.category != "system_control"
                    or existing.metadata != metadata
                ):
                    raise RuntimeError(
                        f"Conflicting registration for system tool: {name}"
                    )

                continue

            registry.register(
                definition,
                permission=permission,
                confirmation=confirmation,
                metadata=metadata,
            )