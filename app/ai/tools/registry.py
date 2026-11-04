"""Strict registry and schema validation for Zara tools."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


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


def _validate_value(
    name: str,
    value: Any,
    schema: dict[str, Any],
    depth: int = 0,
) -> Any:
    if depth > 8:
        raise ToolValidationError(
            "Tool arguments are nested too deeply."
        )

    expected = schema.get("type")

    if expected == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{name} must be an object.")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            raise ToolValidationError("Registered schema is invalid.")

        if not isinstance(required, list):
            raise ToolValidationError("Registered required fields are invalid.")

        unknown = set(value) - set(properties)

        if unknown and schema.get("additionalProperties", False) is False:
            raise ToolValidationError(
                "Unexpected argument(s): "
                + ", ".join(sorted(map(str, unknown)))
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

        return {
            key: _validate_value(
                key,
                item,
                properties[key],
                depth + 1,
            )
            for key, item in value.items()
            if key in properties
        }

    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{name} must be text.")

        value = value.strip()

        if len(value) < schema.get("minLength", 0):
            raise ToolValidationError(f"{name} is too short.")

        if len(value) > schema.get("maxLength", 10000):
            raise ToolValidationError(f"{name} is too long.")

        if "\x00" in value:
            raise ToolValidationError(f"{name} contains an invalid character.")

    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(f"{name} must be an integer.")

        if value < schema.get("minimum", -1_000_000_000):
            raise ToolValidationError(f"{name} is below the minimum.")

        if value > schema.get("maximum", 1_000_000_000):
            raise ToolValidationError(f"{name} exceeds the maximum.")

    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolValidationError(f"{name} must be a number.")

        if value < schema.get("minimum", float("-inf")):
            raise ToolValidationError(f"{name} is below the minimum.")

        if value > schema.get("maximum", float("inf")):
            raise ToolValidationError(f"{name} exceeds the maximum.")

    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{name} must be a boolean.")

    elif expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{name} must be a list.")

        if len(value) < schema.get("minItems", 0):
            raise ToolValidationError(f"{name} has too few items.")

        if len(value) > schema.get("maxItems", 100):
            raise ToolValidationError(f"{name} has too many items.")

        item_schema = schema.get("items", {})

        value = [
            _validate_value(
                f"{name}[{index}]",
                item,
                item_schema,
                depth + 1,
            )
            for index, item in enumerate(value)
        ]

    elif expected is not None:
        raise ToolValidationError(
            f"Unsupported registered schema type: {expected}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{name} is not an allowed value.")

    return value


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    definition: dict[str, Any]
    permission: str
    confirmation: str = "none"

    @property
    def side_effect(self) -> bool:
        return self.permission == "mutate"

    @property
    def confirmation_required(self) -> bool:
        return self.confirmation == "required"


class ToolRegistry:
    """Application-owned tool definitions and their validation contracts."""

    ALLOWED_RESULTS = frozenset({
        "success",
        "error",
        "failed",
        "denied",
        "not_found",
        "needs_confirmation",
        "unchanged",
        "cancelled",
    })

    MAX_ARGUMENT_BYTES = 8192
    MAX_RESULT_BYTES = 32768

    def __init__(
        self,
        definitions: list[dict[str, Any]],
        *,
        read_only_tools: set[str],
        mutating_tools: set[str],
        confirmation_tools: set[str] | None = None,
        conditional_confirmation_tools: set[str] | None = None,
    ) -> None:
        self._specs: dict[str, ToolSpec] = {}

        confirmation_tools = confirmation_tools or set()
        conditional_confirmation_tools = (
            conditional_confirmation_tools or set()
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

        for raw in definitions:
            definition = deepcopy(raw)
            function = definition.get("function")

            if not isinstance(function, dict):
                raise ValueError(
                    "Tool definition is missing its function object."
                )

            name = function.get("name")
            schema = function.get("parameters")

            if (
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 64
                or not name.isascii()
                or not name.replace("_", "").isalnum()
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
                or not isinstance(schema.get("properties"), dict)
                or schema.get("additionalProperties", False) is not False
            ):
                raise ValueError(
                    "Tool definition has an invalid name or schema."
                )

            if name in self._specs:
                raise ValueError(f"Duplicate tool registration: {name}")

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

            self._specs[name] = ToolSpec(
                name=name,
                definition=definition,
                permission=permission,
                confirmation=confirmation,
            )

        classified = read_only_tools | mutating_tools

        if classified - set(self._specs):
            raise ValueError(
                "Permission classification references an unknown tool."
            )

        if confirmation_tools - set(self._specs):
            raise ValueError(
                "Confirmation policy references an unknown tool."
            )

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
        )

    def is_mutating(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.permission == "mutate")

    def definitions(self) -> list[dict[str, Any]]:
        return [
            deepcopy(spec.definition)
            for spec in self._specs.values()
        ]

    def validate_arguments(
        self,
        name: str,
        raw_arguments: Any,
    ) -> dict[str, Any]:
        spec = self.get(name)

        if spec is None:
            raise ToolValidationError("Unknown tool name.")

        if isinstance(raw_arguments, str):
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

        if status not in self.ALLOWED_RESULTS:
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

        message = result.get("message")

        if message is not None and not isinstance(message, str):
            raise ToolValidationError(
                "Tool result message must be text."
            )

        if _json_size(result) > self.MAX_RESULT_BYTES:
            raise ToolValidationError(
                "Tool result exceeds the allowed size."
            )

        return json.loads(
            json.dumps(result, ensure_ascii=False, default=str)
        )