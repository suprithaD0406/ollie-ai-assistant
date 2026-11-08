"""Safe serialization of tool outputs for the local language model."""

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Structured success or sanitized failure from a tool invocation."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


class ToolResultHandler:
    """Convert results to compact, predictable model context."""

    @staticmethod
    def for_model(result: ToolExecutionResult) -> str:
        """Serialize a tool result without leaking exception messages."""
        payload: dict[str, object] = {
            "tool": result.tool_name,
            "ok": result.success,
        }
        if result.success:
            payload["result"] = result.data
        else:
            payload["error"] = result.error or "Tool execution failed."
        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return json.dumps(
                {
                    "tool": result.tool_name,
                    "ok": False,
                    "error": "Tool returned an invalid result.",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
