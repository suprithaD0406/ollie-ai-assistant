"""Definitions and discovery for the finite set of Ollie tools."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


ToolHandler = Callable[[dict[str, Any]], Any]


class ToolRegistryError(ValueError):
    """Raised when a tool definition is invalid or duplicated."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable tool and the JSON schema used to describe its arguments."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolRegistryError("Tool name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolRegistryError("Tool description must be a non-empty string.")
        if not isinstance(self.input_schema, Mapping):
            raise ToolRegistryError("Tool input_schema must be a mapping.")
        if not callable(self.handler):
            raise ToolRegistryError("Tool handler must be callable.")
        if self.input_schema.get("type") != "object":
            raise ToolRegistryError("Tool input_schema must describe an object.")


class ToolRegistry:
    """Register tools by unique name and expose their public descriptions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Register a tool, rejecting duplicate names."""
        if not isinstance(tool, ToolSpec):
            raise TypeError("tool must be a ToolSpec instance.")
        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        """Return a registered tool, or None when the name is unknown."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """Return registered tools in deterministic registration order."""
        return list(self._tools.values())

    def descriptions(self) -> list[dict[str, Any]]:
        """Return the names, descriptions, and schemas used for model discovery."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in self._tools.values()
        ]
