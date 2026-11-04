"""Deterministic application-owned permissions for Zara tools."""
from __future__ import annotations

from dataclasses import dataclass

from app.ai.tools.registry import ToolRegistry, ToolSpec


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    code: str
    message: str


class ToolPermissionPolicy:
    """Model output cannot grant or change application permissions."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        tools_enabled: bool | None = None,
        mutations_enabled: bool | None = None,
        allow_tools: bool | None = None,
        allow_mutations: bool | None = None,
    ) -> None:
        self.registry = registry

        # Old argument names remain supported.
        self.tools_enabled = bool(
            tools_enabled
            if tools_enabled is not None
            else allow_tools
            if allow_tools is not None
            else False
        )

        self.mutations_enabled = bool(
            mutations_enabled
            if mutations_enabled is not None
            else allow_mutations
            if allow_mutations is not None
            else False
        )

    @property
    def allow_tools(self) -> bool:
        return self.tools_enabled

    @property
    def allow_mutations(self) -> bool:
        return self.mutations_enabled

    def authorize(
        self,
        tool_name: str | ToolSpec | None,
    ) -> PermissionDecision:
        if isinstance(tool_name, ToolSpec):
            spec = tool_name
        elif self.registry is not None:
            spec = self.registry.get(tool_name)
        else:
            spec = None

        if spec is None:
            return PermissionDecision(
                False,
                "unknown_tool",
                "This tool is not registered.",
            )

        if not self.tools_enabled:
            return PermissionDecision(
                False,
                "tools_disabled",
                "Tools are disabled for this interface.",
            )

        if spec.permission == "mutate" and not self.mutations_enabled:
            return PermissionDecision(
                False,
                "mutations_disabled",
                "Database changes are disabled for this interface.",
            )

        return PermissionDecision(
            True,
            "allowed",
            "Tool access allowed.",
        )

    def authorize_mutation(self) -> PermissionDecision:
        if not self.tools_enabled:
            return PermissionDecision(
                False,
                "tools_disabled",
                "Tools are disabled for this interface.",
            )

        if not self.mutations_enabled:
            return PermissionDecision(
                False,
                "mutations_disabled",
                "Database changes are disabled for this interface.",
            )

        return PermissionDecision(
            True,
            "allowed",
            "Mutation permission granted.",
        )