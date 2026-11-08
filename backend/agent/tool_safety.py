"""Argument validation and authorization checks for agent tools."""

from collections.abc import Mapping
import re
from typing import Any


class ToolValidationError(ValueError):
    """Raised when generated tool arguments do not satisfy their schema."""


class ToolAuthorizationError(PermissionError):
    """Raised when a requested tool action lacks explicit user authorization."""


_SAVE_INTENT_PATTERN = re.compile(
    r"^\s*(?:(?:ollie[,:\s]+)?(?:please\s+)?"
    r"(?:remember|save|store|keep\s+(?:this|that)\s+in\s+mind|"
    r"don't\s+forget|do\s+not\s+forget)\b|"
    r"(?:can|could|would)\s+you\s+(?:please\s+)?(?:remember|save|store)\b|"
    r"i\s+(?:want|need|would\s+like)\s+you\s+to\s+(?:remember|save|store)\b)",
    re.IGNORECASE,
)


def validate_tool_arguments(
    arguments: object,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an object against the supported, deliberately small JSON Schema subset."""
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be a JSON object.")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ToolValidationError("Tool schema is invalid.")
    if any(not isinstance(key, str) for key in arguments):
        raise ToolValidationError("Tool argument names must be strings.")

    unknown = set(arguments) - set(properties)
    if unknown and schema.get("additionalProperties") is False:
        raise ToolValidationError("Tool arguments contain unsupported fields.")
    missing = set(required) - set(arguments)
    if missing:
        raise ToolValidationError("A required tool argument is missing.")

    for name, value in arguments.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, Mapping):
            if schema.get("additionalProperties") is False:
                raise ToolValidationError("Tool arguments contain unsupported fields.")
            continue
        expected = field_schema.get("type")
        if not _matches_type(value, expected):
            raise ToolValidationError(f"Tool argument {name!r} has an invalid type.")
        if isinstance(value, str):
            if not value.strip():
                raise ToolValidationError(f"Tool argument {name!r} cannot be blank.")
            if len(value) > field_schema.get("maxLength", len(value)):
                raise ToolValidationError(f"Tool argument {name!r} is too long.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in field_schema and value < field_schema["minimum"]:
                raise ToolValidationError(f"Tool argument {name!r} is below its minimum.")
            if "maximum" in field_schema and value > field_schema["maximum"]:
                raise ToolValidationError(f"Tool argument {name!r} exceeds its maximum.")
        if "enum" in field_schema and value not in field_schema["enum"]:
            raise ToolValidationError(f"Tool argument {name!r} is not an allowed value.")

    return dict(arguments)


def authorize_tool_call(name: str, user_message: str) -> None:
    """Require explicit, current-message intent before writing a long-term memory."""
    if name == "save_user_memory" and not _SAVE_INTENT_PATTERN.search(user_message):
        raise ToolAuthorizationError(
            "The user did not explicitly ask Ollie to save a memory."
        )


def _matches_type(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False
