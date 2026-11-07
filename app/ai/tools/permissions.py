"""Deterministic, application-owned tool authorization."""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Literal

from app.ai.tools.registry import (
    ToolRegistry,
    ToolSpec,
)


RiskLevel = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


RISK_LEVELS = (
    "low",
    "medium",
    "high",
    "critical",
)

SUPPORTED_INTERFACES = frozenset({
    "terminal",
    "api",
    "browser",
    "desktop",
    "voice",
    "internal",
})


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """
    Result of a deterministic permission evaluation.

    `allowed` means the registered tool may enter its authorized
    execution flow.

    `requires_confirmation` means side effects must remain paused
    until the trusted application confirms the exact action.
    """

    allowed: bool
    code: str
    message: str

    risk_level: str = "unknown"
    requires_confirmation: bool = False

    confirmation_mode: str = "none"

    required_scopes: tuple[str, ...] = ()

    tool_name: str | None = None
    interface: str | None = None


class ToolPermissionPolicy:
    """
    Central policy for tool access, interfaces and permissions.

    Model output cannot grant permissions, add scopes or change
    a tool's risk classification.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        tools_enabled: bool | None = None,
        mutations_enabled: bool | None = None,
        allow_tools: bool | None = None,
        allow_mutations: bool | None = None,
        allowed_interfaces: set[str] | None = None,
        default_interface: str | None = None,
        granted_scopes: set[str] | None = None,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        allowed_categories: set[str] | None = None,
        denied_categories: set[str] | None = None,
        category_mutation_overrides: dict[str, bool] | None = None,        
        critical_tools_enabled: bool = False,
    ) -> None:

        self.registry = registry

        # Backwards-compatible argument names.
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

        if category_mutation_overrides is None:
            category_mutation_overrides = {}

        if not isinstance(category_mutation_overrides, dict):
            raise TypeError(
                "Category mutation overrides must be a mapping."
            )

        for category, enabled in category_mutation_overrides.items():
            if (
                not isinstance(category, str)
                or not category
                or type(enabled) is not bool
            ):
                raise ValueError(
                    "Invalid category mutation override."
                )

        self.category_mutation_overrides = dict(
            category_mutation_overrides
        )

        self._validate_string_set(
            granted_scopes or set(),
            "granted_scopes",
            r"[a-z][a-z0-9_.:-]{0,127}",
        )

        self.granted_scopes = frozenset(
            granted_scopes or ()
        )

        self.allowed_tools = self._optional_names(
            allowed_tools,
            "allowed_tools",
        )

        self.denied_tools = self._normalise_names(
            denied_tools or set(),
            "denied_tools",
        )

        self.allowed_categories = self._optional_names(
            allowed_categories,
            "allowed_categories",
        )

        self.denied_categories = self._normalise_names(
            denied_categories or set(),
            "denied_categories",
        )

        if (
            self.allowed_tools is not None
            and self.allowed_tools & self.denied_tools
        ):
            raise ValueError(
                "A tool cannot be both explicitly allowed and denied."
            )

        if (
            self.allowed_categories is not None
            and self.allowed_categories & self.denied_categories
        ):
            raise ValueError(
                "A category cannot be both allowed and denied."
            )

        if allowed_interfaces is None:
            self.allowed_interfaces = None
        else:
            interfaces = frozenset(allowed_interfaces)

            if not interfaces <= SUPPORTED_INTERFACES:
                raise ValueError(
                    "Unsupported interface permission."
                )

            self.allowed_interfaces = interfaces

        if default_interface is not None:
            if default_interface not in SUPPORTED_INTERFACES:
                raise ValueError(
                    "Unsupported default interface."
                )

            self.default_interface = default_interface
        else:
            self.default_interface = None

        if not isinstance(critical_tools_enabled, bool):
            raise TypeError(
                "critical_tools_enabled must be boolean."
            )

        self.critical_tools_enabled = critical_tools_enabled

    @staticmethod
    def _validate_string_set(
        values: set[str] | frozenset[str],
        field_name: str,
        pattern: str,
    ) -> None:
        if not isinstance(values, (set, frozenset)):
            raise TypeError(
                f"{field_name} must be a set."
            )

        for value in values:
            if (
                not isinstance(value, str)
                or not re.fullmatch(pattern, value)
                or "*" in value
            ):
                raise ValueError(
                    f"Invalid entry in {field_name}."
                )

    @classmethod
    def _normalise_names(
        cls,
        values: set[str],
        field_name: str,
    ) -> frozenset[str]:
        cls._validate_string_set(
            values,
            field_name,
            r"[a-z][a-z0-9_.-]{0,63}",
        )

        return frozenset(values)

    @classmethod
    def _optional_names(
        cls,
        values: set[str] | None,
        field_name: str,
    ) -> frozenset[str] | None:
        if values is None:
            return None

        return cls._normalise_names(
            values,
            field_name,
        )

    @property
    def allow_tools(self) -> bool:
        return self.tools_enabled

    @property
    def allow_mutations(self) -> bool:
        return self.mutations_enabled

    def _decision(
        self,
        *,
        allowed: bool,
        code: str,
        message: str,
        tool_name: str | None = None,
        interface: str | None = None,
        risk_level: str = "unknown",
        requires_confirmation: bool = False,
        confirmation_mode: str = "none",
        required_scopes: tuple[str, ...] = (),
    ) -> PermissionDecision:

        return PermissionDecision(
            allowed=allowed,
            code=code,
            message=message,
            tool_name=tool_name,
            interface=interface,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            confirmation_mode=confirmation_mode,
            required_scopes=required_scopes,
        )

    def authorize(
        self,
        tool_name: str | ToolSpec | None,
        *,
        interface: str | None = None,
    ) -> PermissionDecision:
        """
        Evaluate one registered tool.

        `interface` must be supplied by trusted application code,
        never copied from model-generated arguments.

        A confirmation-required decision is not approval to execute.
        The caller must pause the action and obtain a matching approval.
        """

        if isinstance(tool_name, ToolSpec):
            requested_name = tool_name.name
        else:
            requested_name = tool_name

        if (
            not isinstance(requested_name, str)
            or not requested_name
        ):
            return self._decision(
                allowed=False,
                code="unknown_tool",
                message="This tool is not registered.",
            )

        # Always resolve the canonical registered entry.
        spec = (
            self.registry.get(requested_name)
            if self.registry is not None
            else None
        )

        if spec is None:
            return self._decision(
                allowed=False,
                code="unknown_tool",
                message="This tool is not registered.",
                tool_name=requested_name,
            )

        if not self.tools_enabled:
            return self._decision(
                allowed=False,
                code="tools_disabled",
                message="Tools are disabled for this interface.",
                tool_name=spec.name,
            )

        if spec.name in self.denied_tools:
            return self._decision(
                allowed=False,
                code="tool_denied",
                message="This tool is explicitly blocked.",
                tool_name=spec.name,
            )

        if (
            self.allowed_tools is not None
            and spec.name not in self.allowed_tools
        ):
            return self._decision(
                allowed=False,
                code="tool_not_allowlisted",
                message="This tool is not in the allowed tool list.",
                tool_name=spec.name,
            )

        category = spec.category

        if category in self.denied_categories:
            return self._decision(
                allowed=False,
                code="category_denied",
                message="This tool category is blocked.",
                tool_name=spec.name,
            )

        if (
            self.allowed_categories is not None
            and category not in self.allowed_categories
        ):
            return self._decision(
                allowed=False,
                code="category_not_allowlisted",
                message="This category is not enabled.",
                tool_name=spec.name,
            )

        selected_interface = (
            interface
            if interface is not None
            else self.default_interface
        )

        if selected_interface is not None:
            if selected_interface not in SUPPORTED_INTERFACES:
                return self._decision(
                    allowed=False,
                    code="invalid_interface",
                    message="The requested interface is not supported.",
                    tool_name=spec.name,
                )

            if (
                self.allowed_interfaces is not None
                and selected_interface not in self.allowed_interfaces
            ):
                return self._decision(
                    allowed=False,
                    code="interface_not_allowed",
                    message="This interface is not authorized.",
                    tool_name=spec.name,
                    interface=selected_interface,
                )

            if (
                selected_interface
                not in spec.supported_interfaces
            ):
                return self._decision(
                    allowed=False,
                    code="tool_interface_mismatch",
                    message="This tool does not support this interface.",
                    tool_name=spec.name,
                    interface=selected_interface,
                )

        if spec.permission not in {"read", "mutate"}:
            return self._decision(
                allowed=False,
                code="invalid_permission_classification",
                message="The tool has an invalid permission classification.",
                tool_name=spec.name,
            )

        category_mutations_enabled = (
            self.category_mutation_overrides.get(
                spec.category,
                self.mutations_enabled,
            )
        )

        if (
            spec.permission == "mutate"
            and not category_mutations_enabled
        ):
            return self._decision(
                allowed=False,
                code="mutations_disabled",
                message="Mutating operations are disabled.",
                tool_name=spec.name,
                interface=selected_interface,
            )

        metadata = spec.metadata

        risk_level = getattr(
            metadata,
            "risk_level",
            "high",
        )

        if risk_level not in RISK_LEVELS:
            return self._decision(
                allowed=False,
                code="invalid_risk_classification",
                message="Tool risk classification is invalid.",
                tool_name=spec.name,
                interface=selected_interface,
            )

        if (
            risk_level == "critical"
            and not self.critical_tools_enabled
        ):
            return self._decision(
                allowed=False,
                code="critical_action_blocked",
                message=(
                    "Critical-risk tools are disabled by application policy."
                ),
                tool_name=spec.name,
                interface=selected_interface,
                risk_level=risk_level,
            )

        required_scopes = tuple(
            getattr(metadata, "required_scopes", ())
        )

        missing_scopes = (
            set(required_scopes) - self.granted_scopes
        )

        if missing_scopes:
            return self._decision(
                allowed=False,
                code="missing_permission_scope",
                message=(
                    "The tool does not have all required permission scopes."
                ),
                tool_name=spec.name,
                interface=selected_interface,
                risk_level=risk_level,
                required_scopes=required_scopes,
            )

        confirmation_mode = spec.confirmation

        requires_confirmation = (
            confirmation_mode in {
                "required",
                "conditional",
            }
            or risk_level in {
                "medium",
                "high",
                "critical",
            }
        )

        if requires_confirmation:
            return self._decision(
                allowed=True,
                code="confirmation_required",
                message=(
                    "Tool access is authorized, but execution must wait "
                    "for confirmation of the exact action."
                ),
                tool_name=spec.name,
                interface=selected_interface,
                risk_level=risk_level,
                requires_confirmation=True,
                confirmation_mode=confirmation_mode,
                required_scopes=required_scopes,
            )

        return self._decision(
            allowed=True,
            code="allowed",
            message="Tool access allowed.",
            tool_name=spec.name,
            interface=selected_interface,
            risk_level=risk_level,
            confirmation_mode=confirmation_mode,
            required_scopes=required_scopes,
        )

    def authorize_mutation(self) -> PermissionDecision:
        """
        Backwards-compatible global mutation gate.

        Existing DatabaseToolService performs its own preview and
        confirmation workflow after this capability check.
        """

        if not self.tools_enabled:
            return self._decision(
                allowed=False,
                code="tools_disabled",
                message="Tools are disabled for this interface.",
            )

        if not self.mutations_enabled:
            return self._decision(
                allowed=False,
                code="mutations_disabled",
                message="Mutating operations are disabled.",
            )

        return self._decision(
            allowed=True,
            code="allowed",
            message="Mutation capability is enabled.",
        )