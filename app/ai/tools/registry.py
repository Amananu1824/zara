"""Strict registry and schema validation for Zara tools."""
from __future__ import annotations

import json
import math
import re

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.ai.tools.contracts import (
    TOOL_RESULT_STATUSES,
    ToolResult,
)


class ToolValidationError(ValueError):
    """Invalid tool request or tool result."""


class DuplicateJSONKeyError(ToolValidationError):
    """Repeated key in model-generated JSON."""


# Backward compatibility with the earlier registry.
ToolRegistryError = ToolValidationError


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(
                "Tool arguments contain duplicate JSON keys."
            )
        result[key] = value

    return result


def _reject_constant(value: str) -> None:
    raise ToolValidationError(
        f"Invalid JSON numeric constant: {value}"
    )


def _json_size(value: Any) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ToolValidationError(
            "Tool data must contain valid JSON-compatible values."
        ) from exc


SUPPORTED_SCHEMA_TYPES = frozenset({
    "object",
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "null",
})

SUPPORTED_SCHEMA_KEYS = frozenset({
    "type",
    "description",
    "title",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "enum",
    "default",
})


def _validate_schema(
    schema: dict[str, Any],
    path: str = "schema",
    depth: int = 0,
) -> None:
    """Validate a tool's JSON Schema before registration."""

    if depth > 8:
        raise ToolValidationError(
            "Registered schema is nested too deeply."
        )

    if not isinstance(schema, dict):
        raise ToolValidationError(
            f"{path} must be an object."
        )

    unsupported = set(schema) - SUPPORTED_SCHEMA_KEYS

    if unsupported:
        raise ToolValidationError(
            f"{path} contains unsupported schema keywords: "
            + ", ".join(sorted(unsupported))
        )

    expected = schema.get("type")

    if (
        not isinstance(expected, str)
        or expected not in SUPPORTED_SCHEMA_TYPES
    ):
        raise ToolValidationError(
            f"{path} has an unsupported or missing type."
        )

    for key in ("description", "title"):
        value = schema.get(key)

        if value is not None and not isinstance(value, str):
            raise ToolValidationError(
                f"{path}.{key} must be text."
            )

    if "default" in schema:
        _json_size(schema["default"])

    if "enum" in schema:
        choices = schema["enum"]

        if not isinstance(choices, list) or not choices:
            raise ToolValidationError(
                f"{path}.enum must be a non-empty list."
            )

        _json_size(choices)

    if expected == "object":
        properties = schema.get("properties")
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            raise ToolValidationError(
                f"{path}.properties must be an object."
            )

        if schema.get("additionalProperties") is not False:
            raise ToolValidationError(
                f"{path} must explicitly disable additionalProperties."
            )

        if not isinstance(required, list):
            raise ToolValidationError(
                f"{path}.required must be a list."
            )

        if (
            any(not isinstance(item, str) for item in required)
            or len(required) != len(set(required))
        ):
            raise ToolValidationError(
                f"{path}.required contains invalid fields."
            )

        unknown_required = set(required) - set(properties)

        if unknown_required:
            raise ToolValidationError(
                f"{path}.required references unknown properties."
            )

        for key, property_schema in properties.items():
            if not isinstance(key, str) or not key:
                raise ToolValidationError(
                    f"{path} contains an invalid property name."
                )

            _validate_schema(
                property_schema,
                f"{path}.{key}",
                depth + 1,
            )

        minimum = schema.get("minProperties", 0)
        maximum = schema.get("maxProperties", 10000)

        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < minimum
        ):
            raise ToolValidationError(
                f"{path} has invalid property limits."
            )

    elif expected == "string":
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength", 10000)

        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < minimum
        ):
            raise ToolValidationError(
                f"{path} has invalid string length limits."
            )

        pattern = schema.get("pattern")

        if pattern is not None:
            if not isinstance(pattern, str):
                raise ToolValidationError(
                    f"{path}.pattern must be text."
                )

            try:
                re.compile(pattern)
            except re.error as exc:
                raise ToolValidationError(
                    f"{path} contains an invalid regex pattern."
                ) from exc

    elif expected in {"integer", "number"}:
        for key in (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
        ):
            value = schema.get(key)

            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (
                    isinstance(value, float)
                    and not math.isfinite(value)
                )
            ):
                raise ToolValidationError(
                    f"{path}.{key} must be a finite number."
                )

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ToolValidationError(
                f"{path} has inconsistent numeric limits."
            )

        if (
            "exclusiveMinimum" in schema
            and "exclusiveMaximum" in schema
            and schema["exclusiveMinimum"] >= schema["exclusiveMaximum"]
        ):
            raise ToolValidationError(
                f"{path} has inconsistent exclusive limits."
            )

    elif expected == "array":
        minimum = schema.get("minItems", 0)
        maximum = schema.get("maxItems", 100)

        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < minimum
        ):
            raise ToolValidationError(
                f"{path} has invalid array limits."
            )

        if not isinstance(schema.get("uniqueItems", False), bool):
            raise ToolValidationError(
                f"{path}.uniqueItems must be boolean."
            )

        item_schema = schema.get("items")

        if not isinstance(item_schema, dict):
            raise ToolValidationError(
                f"{path}.items is required."
            )

        _validate_schema(
            item_schema,
            f"{path}.items",
            depth + 1,
        )


def _validate_value(
    name: str,
    value: Any,
    schema: dict[str, Any],
    depth: int = 0,
) -> Any:
    """Validate and normalize a value against a registered schema."""

    if depth > 8:
        raise ToolValidationError(
            "Tool arguments are nested too deeply."
        )

    expected = schema.get("type")

    if expected == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(
                f"{name} must be an object."
            )

        if any(not isinstance(key, str) for key in value):
            raise ToolValidationError(
                f"{name} contains invalid property names."
            )

        properties = schema["properties"]
        required = schema.get("required", [])

        unknown = set(value) - set(properties)

        if unknown:
            raise ToolValidationError(
                "Unexpected argument(s): "
                + ", ".join(sorted(unknown))
            )

        missing = [
            key for key in required
            if key not in value
        ]

        if missing:
            raise ToolValidationError(
                "Missing required argument(s): "
                + ", ".join(missing)
            )

        if len(value) < schema.get("minProperties", 0):
            raise ToolValidationError(
                f"{name} contains too few properties."
            )

        if len(value) > schema.get("maxProperties", 10000):
            raise ToolValidationError(
                f"{name} contains too many properties."
            )

        return {
            key: _validate_value(
                key,
                item,
                properties[key],
                depth + 1,
            )
            for key, item in value.items()
        }

    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(
                f"{name} must be text."
            )

        if len(value) < schema.get("minLength", 0):
            raise ToolValidationError(
                f"{name} is too short."
            )

        if len(value) > schema.get("maxLength", 10000):
            raise ToolValidationError(
                f"{name} is too long."
            )

        if "\x00" in value:
            raise ToolValidationError(
                f"{name} contains an invalid character."
            )

        pattern = schema.get("pattern")

        if pattern is not None and not re.search(pattern, value):
            raise ToolValidationError(
                f"{name} does not match the required format."
            )

    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(
                f"{name} must be an integer."
            )

    elif expected == "number":
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ToolValidationError(
                f"{name} must be a number."
            )

    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(
                f"{name} must be a boolean."
            )

    elif expected == "null":
        if value is not None:
            raise ToolValidationError(
                f"{name} must be null."
            )

    elif expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(
                f"{name} must be a list."
            )

        if len(value) < schema.get("minItems", 0):
            raise ToolValidationError(
                f"{name} has too few items."
            )

        if len(value) > schema.get("maxItems", 100):
            raise ToolValidationError(
                f"{name} has too many items."
            )

        item_schema = schema["items"]

        value = [
            _validate_value(
                f"{name}[{index}]",
                item,
                item_schema,
                depth + 1,
            )
            for index, item in enumerate(value)
        ]

        if schema.get("uniqueItems", False):
            serialized = [
                json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                for item in value
            ]

            if len(serialized) != len(set(serialized)):
                raise ToolValidationError(
                    f"{name} contains duplicate items."
                )

    else:
        raise ToolValidationError(
            f"{name} has an unsupported schema type."
        )

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    exclusive_minimum = schema.get("exclusiveMinimum")
    exclusive_maximum = schema.get("exclusiveMaximum")

    if expected in {"integer", "number"}:
        if minimum is not None and value < minimum:
            raise ToolValidationError(
                f"{name} is below the minimum."
            )

        if maximum is not None and value > maximum:
            raise ToolValidationError(
                f"{name} exceeds the maximum."
            )

        if (
            exclusive_minimum is not None
            and value <= exclusive_minimum
        ):
            raise ToolValidationError(
                f"{name} must be above the exclusive minimum."
            )

        if (
            exclusive_maximum is not None
            and value >= exclusive_maximum
        ):
            raise ToolValidationError(
                f"{name} must be below the exclusive maximum."
            )

    if "enum" in schema:
        matched = any(
            (
                expected == "number"
                and not isinstance(choice, bool)
                and isinstance(choice, (int, float))
                and value == choice
            )
            or (
                type(value) is type(choice)
                and value == choice
            )
            for choice in schema["enum"]
        )

        if not matched:
            raise ToolValidationError(
                f"{name} is not an allowed value."
            )

    return value

@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Application-owned descriptive and security metadata."""

    category: str = "general"
    version: str = "1.0.0"
    provider: str = "native"

    capabilities: tuple[str, ...] = ()

    supported_interfaces: tuple[str, ...] = (
        "terminal",
        "api",
    )

    # Unknown/new tools default to high risk.
    risk_level: str = "high"

    # Granted by application configuration, never by the model.
    required_scopes: tuple[str, ...] = ()

    timeout_seconds: float = 30.0
    max_retries: int = 0
    idempotent: bool = False

    output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        slug_pattern = r"[a-z][a-z0-9_.-]{0,63}"

        if (
            not isinstance(self.category, str)
            or not re.fullmatch(slug_pattern, self.category)
        ):
            raise ValueError("Invalid tool category.")

        if (
            not isinstance(self.provider, str)
            or not re.fullmatch(slug_pattern, self.provider)
        ):
            raise ValueError("Invalid tool provider.")

        if (
            not isinstance(self.version, str)
            or not re.fullmatch(
                r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?",
                self.version,
            )
        ):
            raise ValueError(
                "Tool version must use semantic versioning."
            )

        allowed_risks = {
            "low",
            "medium",
            "high",
            "critical",
        }

        if self.risk_level not in allowed_risks:
            raise ValueError("Invalid tool risk level.")

        if not isinstance(self.capabilities, tuple):
            raise TypeError(
                "Tool capabilities must be a tuple."
            )

        if not isinstance(self.supported_interfaces, tuple):
            raise TypeError(
                "Supported interfaces must be a tuple."
            )

        if not isinstance(self.required_scopes, tuple):
            raise TypeError(
                "Required scopes must be a tuple."
            )

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0.1 <= self.timeout_seconds <= 600
        ):
            raise ValueError(
                "Tool timeout must be between 0.1 and 600 seconds."
            )

        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= 5
        ):
            raise ValueError(
                "Tool max_retries must be between 0 and 5."
            )

        if not isinstance(self.idempotent, bool):
            raise TypeError(
                "Tool idempotent must be boolean."
            )
        
        if (
            len(self.capabilities)
            != len(set(self.capabilities))
        ):
            raise ValueError("Duplicate tool capabilities.")

        if (
            len(self.supported_interfaces)
            != len(set(self.supported_interfaces))
        ):
            raise ValueError(
                "Duplicate supported interfaces."
            )

        supported_interfaces = {
            "terminal",
            "api",
            "browser",
            "desktop",
            "voice",
            "internal",
        }

        for item in self.capabilities:
            if (
                not isinstance(item, str)
                or not re.fullmatch(slug_pattern, item)
            ):
                raise ValueError(
                    "Invalid tool capability."
                )

        for item in self.supported_interfaces:
            if item not in supported_interfaces:
                raise ValueError(
                    f"Unsupported interface: {item!r}"
                )

        scope_pattern = r"[a-z][a-z0-9_.:-]{0,127}"

        for scope in self.required_scopes:
            if (
                not isinstance(scope, str)
                or not re.fullmatch(scope_pattern, scope)
            ):
                raise ValueError(
                    "Invalid required permission scope."
                )

        if (
            len(self.required_scopes)
            != len(set(self.required_scopes))
        ):
            raise ValueError(
                "Duplicate required permission scopes."
            )

        if (
            self.output_schema is not None
            and not isinstance(self.output_schema, dict)
        ):
            raise TypeError(
                "Tool output schema must be an object."
            )

@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable registered tool specification."""

    name: str
    definition: dict[str, Any]
    permission: str
    confirmation: str = "none"

    metadata: ToolMetadata = field(
        default_factory=ToolMetadata
    )

    @property
    def side_effect(self) -> bool:
        return self.permission == "mutate"

    @property
    def confirmation_required(self) -> bool:
        return self.confirmation == "required"

    @property
    def category(self) -> str:
        return self.metadata.category

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self.metadata.capabilities

    @property
    def supported_interfaces(self) -> tuple[str, ...]:
        return self.metadata.supported_interfaces

    @property
    def input_schema(self) -> dict[str, Any]:
        return deepcopy(
            self.definition["function"]["parameters"]
        )

    @property
    def output_schema(self) -> dict[str, Any] | None:
        return deepcopy(self.metadata.output_schema)


class ToolRegistry:
    """
    Central application-owned registry.

    Only trusted application code may register tools.
    Model-generated output can request a registered tool,
    but cannot create or modify registry entries.
    """

    ALLOWED_RESULTS = TOOL_RESULT_STATUSES

    MAX_ARGUMENT_BYTES = 8192
    MAX_RESULT_BYTES = 32768

    _ALLOWED_PERMISSIONS = frozenset({
        "read",
        "mutate",
    })

    _ALLOWED_CONFIRMATIONS = frozenset({
        "none",
        "required",
        "conditional",
    })

    def __init__(
        self,
        definitions: list[dict[str, Any]],
        *,
        read_only_tools: set[str],
        mutating_tools: set[str],
        confirmation_tools: set[str] | None = None,
        conditional_confirmation_tools: set[str] | None = None,
        metadata_by_name: dict[
            str,
            ToolMetadata | dict[str, Any],
        ] | None = None,
        default_metadata: ToolMetadata | dict[str, Any] | None = None,
    ) -> None:

        self._specs: dict[str, ToolSpec] = {}

        read_only_tools = set(read_only_tools)
        mutating_tools = set(mutating_tools)

        confirmation_tools = set(confirmation_tools or ())
        conditional_confirmation_tools = set(
            conditional_confirmation_tools or ()
        )

        if read_only_tools & mutating_tools:
            raise ValueError(
                "A tool cannot be both read-only and mutating."
            )

        if confirmation_tools & conditional_confirmation_tools:
            raise ValueError(
                "Confirmation classifications cannot overlap."
            )

        if not (
            confirmation_tools | conditional_confirmation_tools
        ) <= mutating_tools:
            raise ValueError(
                "Confirmation is only valid for mutating tools."
            )

        if metadata_by_name is None:
            metadata_by_name = {}

        if not isinstance(metadata_by_name, dict):
            raise TypeError("metadata_by_name must be a dictionary.")

        if any(not isinstance(key, str) for key in metadata_by_name):
            raise ValueError("Metadata keys must be tool names.")

        self._default_metadata = self._coerce_metadata(
            default_metadata
        )

        for raw in definitions:
            if not isinstance(raw, dict):
                raise ValueError(
                    "Tool definition must be an object."
                )

            function = raw.get("function")

            if not isinstance(function, dict):
                raise ValueError(
                    "Tool definition is missing its function object."
                )

            name = function.get("name")

            if not isinstance(name, str):
                raise ValueError("Tool definition has an invalid name.")

            if name in mutating_tools:
                permission = "mutate"
            elif name in read_only_tools:
                permission = "read"
            else:
                raise ValueError(
                    f"Tool {name!r} has no explicit permission classification."
                )

            confirmation = (
                "required"
                if name in confirmation_tools
                else "conditional"
                if name in conditional_confirmation_tools
                else "none"
            )

            self.register(
                raw,
                permission=permission,
                confirmation=confirmation,
                metadata=metadata_by_name.get(name),
            )

        registered = set(self._specs)

        if (read_only_tools | mutating_tools) - registered:
            raise ValueError(
                "Permission classification references an unknown tool."
            )

        if (confirmation_tools | conditional_confirmation_tools) - registered:
            raise ValueError(
                "Confirmation policy references an unknown tool."
            )

        if set(metadata_by_name) - registered:
            raise ValueError(
                "Metadata references an unknown tool."
            )

    @staticmethod
    def _coerce_metadata(
        value: ToolMetadata | dict[str, Any] | None,
    ) -> ToolMetadata:
        """Validate and copy registration metadata."""

        if value is None:
            value = ToolMetadata()

        if isinstance(value, ToolMetadata):
            raw = {
                "category": value.category,
                "version": value.version,
                "provider": value.provider,
                "capabilities": value.capabilities,
                "supported_interfaces": value.supported_interfaces,
                "risk_level": value.risk_level,
                "required_scopes": value.required_scopes,
                "output_schema": value.output_schema,
                "timeout_seconds": value.timeout_seconds,
                "max_retries": value.max_retries,
                "idempotent": value.idempotent,
            }

        elif isinstance(value, dict):
            raw = deepcopy(value)

        else:
            raise TypeError("Invalid tool metadata.")

        allowed = set(ToolMetadata.__dataclass_fields__)
        unknown = set(raw) - allowed

        if unknown:
            raise ValueError(
                "Unknown tool metadata field(s): "
                + ", ".join(sorted(unknown))
            )

        capabilities = raw.get("capabilities", ())
        interfaces = raw.get(
            "supported_interfaces",
            ("terminal", "api"),
        )
        required_scopes = raw.get(
            "required_scopes",
            (),
        )

        for value_to_check, field_name in (
            (capabilities, "capabilities"),
            (interfaces, "supported_interfaces"),
            (required_scopes, "required_scopes"),
        ):
            if (
                isinstance(value_to_check, str)
                or not isinstance(value_to_check, (list, tuple))
            ):
                raise ValueError(
                    f"Invalid tool metadata: {field_name}."
                )

        output_schema = raw.get("output_schema")

        if output_schema is not None:
            output_schema = deepcopy(output_schema)

            _validate_schema(
                output_schema,
                "output_schema",
            )

            if output_schema.get("type") != "object":
                raise ToolValidationError(
                    "Tool output schema must have object as its root type."
                )

        return ToolMetadata(
            category=raw.get("category", "general"),
            version=raw.get("version", "1.0.0"),
            provider=raw.get("provider", "native"),
            capabilities=tuple(capabilities),
            supported_interfaces=tuple(interfaces),
            risk_level=raw.get("risk_level", "high"),
            required_scopes=tuple(required_scopes),
            output_schema=output_schema,
            timeout_seconds=raw.get("timeout_seconds", 30.0),
            max_retries=raw.get("max_retries", 0),
            idempotent=raw.get("idempotent", False),
        )

    def register(
        self,
        definition: dict[str, Any],
        *,
        permission: str,
        confirmation: str = "none",
        metadata: ToolMetadata | dict[str, Any] | None = None,
    ) -> ToolSpec:
        """
        Register one trusted application tool.

        Duplicate names are rejected. Replacing an existing contract
        requires an explicit application-level code change.
        """

        if not isinstance(definition, dict):
            raise ToolValidationError(
                "Tool definition must be an object."
            )

        definition = deepcopy(definition)

        if definition.get("type") != "function":
            raise ToolValidationError(
                "Only function tools can be registered."
            )

        function = definition.get("function")

        if not isinstance(function, dict):
            raise ToolValidationError(
                "Tool definition is missing its function object."
            )

        name = function.get("name")
        description = function.get("description")
        schema = function.get("parameters")

        if (
            not isinstance(name, str)
            or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}",
                name,
            )
        ):
            raise ToolValidationError(
                "Tool name must use lowercase snake_case."
            )

        if (
            not isinstance(description, str)
            or not description.strip()
            or len(description) > 2048
        ):
            raise ToolValidationError(
                f"Tool {name!r} has an invalid description."
            )

        if name in self._specs:
            raise ToolValidationError(
                f"Duplicate tool registration: {name}"
            )

        if permission not in self._ALLOWED_PERMISSIONS:
            raise ToolValidationError(
                f"Tool {name!r} has an invalid permission classification."
            )

        if confirmation not in self._ALLOWED_CONFIRMATIONS:
            raise ToolValidationError(
                f"Tool {name!r} has an invalid confirmation policy."
            )

        if confirmation != "none" and permission != "mutate":
            raise ToolValidationError(
                "Confirmation policies apply only to mutating tools."
            )

        if (
            not isinstance(schema, dict)
            or schema.get("type") != "object"
        ):
            raise ToolValidationError(
                f"Tool {name!r} requires an object input schema."
            )

        _validate_schema(schema, f"{name}.input")

        tool_metadata = self._coerce_metadata(
            metadata
            if metadata is not None
            else self._default_metadata
        )

        spec = ToolSpec(
            name=name,
            definition=definition,
            permission=permission,
            confirmation=confirmation,
            metadata=tool_metadata,
        )

        self._specs[name] = spec

        return self.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        if not isinstance(name, str):
            return None

        spec = self._specs.get(name)

        if spec is None:
            return None

        return ToolSpec(
            name=spec.name,
            definition=deepcopy(spec.definition),
            permission=spec.permission,
            confirmation=spec.confirmation,
            metadata=self._coerce_metadata(spec.metadata),
        )

    def specifications(
        self,
        *,
        category: str | None = None,
        interface: str | None = None,
        capabilities: set[str] | None = None,
    ) -> tuple[ToolSpec, ...]:
        """Discover registered tools by their metadata."""

        required_capabilities = set(capabilities or ())
        matches: list[ToolSpec] = []

        for name in self.names:
            spec = self.get(name)

            if spec is None:
                continue

            if category is not None and spec.category != category:
                continue

            if (
                interface is not None
                and interface not in spec.supported_interfaces
            ):
                continue

            if not required_capabilities <= set(spec.capabilities):
                continue

            matches.append(spec)

        return tuple(matches)

    def is_mutating(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.permission == "mutate")

    def definitions(
        self,
        *,
        category: str | None = None,
        interface: str | None = None,
        capabilities: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return model-facing function definitions."""

        return [
            deepcopy(spec.definition)
            for spec in self.specifications(
                category=category,
                interface=interface,
                capabilities=capabilities,
            )
        ]

    def validate_arguments(
        self,
        name: str,
        raw_arguments: Any,
    ) -> dict[str, Any]:
        """Strictly parse and validate model-generated arguments."""

        spec = self.get(name)

        if spec is None:
            raise ToolValidationError("Unknown tool name.")

        if isinstance(raw_arguments, str):
            try:
                raw_size = len(
                    raw_arguments.encode("utf-8")
                )
            except UnicodeError as exc:
                raise ToolValidationError(
                    "Tool arguments contain invalid text."
                ) from exc

            if raw_size > self.MAX_ARGUMENT_BYTES:
                raise ToolValidationError(
                    "Tool arguments exceed the allowed size."
                )

            try:
                raw_arguments = json.loads(
                    raw_arguments,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_constant,
                )

            except ToolValidationError:
                raise

            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ToolValidationError(
                    "Tool arguments contain invalid JSON."
                ) from exc

        if not isinstance(raw_arguments, dict):
            raise ToolValidationError(
                "Tool arguments must be a JSON object."
            )

        if _json_size(raw_arguments) > self.MAX_ARGUMENT_BYTES:
            raise ToolValidationError(
                "Tool arguments exceed the allowed size."
            )

        validated = _validate_value(
            "arguments",
            raw_arguments,
            spec.definition["function"]["parameters"],
        )

        if not isinstance(validated, dict):
            raise ToolValidationError(
                "Validated arguments must be an object."
            )

        return validated

    def validate_result(
        self,
        name: str,
        result: Any,
    ) -> dict[str, Any]:
        """Validate a handler result against the shared ToolResult contract."""

        spec = self.get(name)

        if spec is None:
            raise ToolValidationError(
                "Cannot validate a result for an unknown tool."
            )

        if not isinstance(result, dict):
            raise ToolValidationError(
                "Tool handler returned a non-object result."
            )

        status = result.get("status")

        if (
            not isinstance(status, str)
            or status not in self.ALLOWED_RESULTS
        ):
            raise ToolValidationError(
                "Tool handler returned an unsupported status."
            )

        if spec.confirmation == "required" and status == "success":
            raise ToolValidationError(
                "A confirmation-required tool returned success directly."
            )

        if (
            spec.confirmation == "conditional"
            and status == "success"
            and result.get("operation") != "profile_created"
        ):
            raise ToolValidationError(
                "Conditional-confirmation tool returned an unverified success."
            )

        try:
            normalized = ToolResult.from_mapping(
                name,
                result,
            )

        except (TypeError, ValueError, RecursionError) as exc:
            raise ToolValidationError(
                "Tool handler returned an invalid standard ToolResult."
            ) from exc

        if spec.metadata.output_schema is not None:
            _validate_value(
                "result.data",
                normalized.data,
                spec.metadata.output_schema,
            )

        legacy_result = normalized.to_legacy_dict()

        if _json_size(legacy_result) > self.MAX_RESULT_BYTES:
            raise ToolValidationError(
                "Tool result exceeds the allowed size."
            )

        return legacy_result