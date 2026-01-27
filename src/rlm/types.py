from __future__ import annotations

from typing import Any, Literal, NotRequired, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable


# OpenResponses Protocol Types
class Input(TypedDict):
    """OpenResponses standard input message.

    Represents a single message in the conversation history.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class Item(TypedDict, total=False):
    """OpenResponses standard item (optional fields).

    Used for tool calls or other structured content.
    """

    type: str
    content: Any


class Output(TypedDict):
    """OpenResponses standard output.

    Standardized response format from LLM calls.
    """

    content: str
    metadata: dict[str, Any]
    sub_results: NotRequired[list[Output]]  # Optional, for nested results


class LLMCaller(Protocol):
    """Protocol for LLM backend.

    Any callable matching this signature can be used as an LLM backend.
    This enables dependency injection and testability.
    """

    def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Call LLM with inputs and return output.

        Args:
            inputs: List of Input messages (role, content)
            context: Metadata (mode, schema, etc.)

        Returns:
            Output dict with content and metadata
        """
        ...


class AsyncLLMCaller(Protocol):
    """Async protocol for LLM backends.

    Any async callable matching this signature can be used as an async LLM
    backend. This enables dependency injection for async engines.

    Example:
        >>> async def my_async_llm(inputs: list[Input], context: dict[str, Any]) -> Output:
        ...     return {"content": "result", "metadata": {}}
    """

    def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Awaitable[Output]:
        """Call LLM asynchronously and return output.

        Args:
            inputs: List of Input messages (role, content)
            context: Metadata (mode, schema, etc.)

        Returns:
            Awaitable Output dict with content and metadata
        """
        ...


class AsyncToolCaller(Protocol):
    """Async protocol for tool-calling backends.

    Mirrors AsyncLLMCaller to enable tool pipelines that return Output.

    Example:
        >>> async def my_tool_call(inputs: list[Input], context: dict[str, Any]) -> Output:
        ...     return {"content": "tool result", "metadata": {}}
    """

    def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Awaitable[Output]:
        """Call tool backend asynchronously and return output.

        Args:
            inputs: List of Input messages (role, content)
            context: Metadata (mode, schema, etc.)

        Returns:
            Awaitable Output dict with content and metadata
        """
        ...


# Internal Control Flow Types
class SubTask(TypedDict):
    """Sub-task with agent assignment for multi-agent routing.

    Used within PlannerDecision to specify task decomposition
    with optional agent assignment for specialized execution.

    Fields:
        description: Required task description
        assigned_agent: Optional agent name (None means use router_model)
    """

    description: str
    assigned_agent: NotRequired[str | None]


class PlannerDecision(TypedDict):
    """Schema for planner LLM output.

    Used to decide whether to execute task atomically
    or decompose into sub-tasks with agent assignments.
    """

    thoughts: str
    decision: Literal["EXECUTE", "RECURSE"]
    sub_tasks: list[SubTask]  # Enhanced to support agent assignment


# Trace Object (MIMIR Compatible)
class TraceObject(TypedDict):
    """MIMIR-compatible execution trace.

    Emitted for each step in the recursion tree.
    Enables observability and debugging.
    """

    trace_id: str
    parent_id: str | None
    root_id: str
    depth: int
    input: str
    output: str
    metadata: dict[str, Any]


__all__ = [
    "Input",
    "Item",
    "Output",
    "LLMCaller",
    "AsyncLLMCaller",
    "AsyncToolCaller",
    "SubTask",
    "PlannerDecision",
    "TraceObject",
]
