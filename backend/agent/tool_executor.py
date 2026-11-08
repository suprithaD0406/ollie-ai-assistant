"""Validated, bounded execution of registered tools."""

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging
from typing import Any

from agent.tool_registry import ToolRegistry
from agent.tool_result_handler import ToolExecutionResult
from agent.tool_safety import (
    ToolAuthorizationError,
    ToolValidationError,
    authorize_tool_call,
    validate_tool_arguments,
)


logger = logging.getLogger(__name__)
_TOOL_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ollie-tool")
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


class ToolExecutor:
    """Enforce registry, schema, authorization, and timeout policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a positive number.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number.")
        self.registry = registry
        self.timeout_seconds = float(timeout_seconds)

    def execute(
        self,
        name: str,
        arguments: object,
        *,
        user_message: str,
    ) -> ToolExecutionResult:
        """Validate and execute one known tool, returning only safe error messages."""
        tool = self.registry.get(name) if isinstance(name, str) else None
        if tool is None:
            return ToolExecutionResult(
                tool_name=name if isinstance(name, str) else "unknown",
                success=False,
                error="Unknown tool.",
            )
        try:
            validated = validate_tool_arguments(arguments, tool.input_schema)
            authorize_tool_call(name, user_message)
        except (ToolValidationError, ToolAuthorizationError) as exc:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=str(exc),
            )

        future: Future[Any] = _TOOL_POOL.submit(tool.handler, validated)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            logger.warning("Tool %s exceeded its execution timeout.", name)
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error="Tool execution timed out.",
            )
        except Exception as exc:
            logger.warning("Tool %s failed (%s).", name, type(exc).__name__)
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error="Tool execution failed.",
            )
        return ToolExecutionResult(tool_name=name, success=True, data=result)
