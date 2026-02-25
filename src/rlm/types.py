from __future__ import annotations

from typing import Any, Literal, NotRequired, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable


class UsageInfo(TypedDict, total=False):
    """Token usage reported by an LLM backend."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    cost_usd: float


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


class ToolCall(TypedDict):
    """Tool call request from LLM.

    Represents a request from the LLM to execute an external tool.
    Used in tool calling workflows where the LLM can request function execution.

    Fields:
        name: Tool identifier (must match registered tool name)
        arguments: Dict of parameter values matching tool's JSON Schema
    """

    name: str
    arguments: dict[str, Any]


class Output(TypedDict):
    """OpenResponses standard output.

    Standardized response format from LLM calls.
    Extended to support tool calling workflows via optional tool_calls field.
    """

    content: str
    metadata: dict[str, Any]
    usage: NotRequired[UsageInfo]  # Optional, token usage from backend
    sub_results: NotRequired[list[Output]]  # Optional, for nested results
    tool_calls: NotRequired[list[ToolCall]]  # Optional, for tool calling


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


class AsyncStreamingLLMCaller(Protocol):
    """Async protocol for streaming LLM backends.

    Extends AsyncLLMCaller with token-level streaming support.
    LLMs that support streaming can yield tokens as they're generated
    for sub-500ms time-to-first-token (TTFT).

    The stream() method returns an AsyncGenerator that yields token strings
    as they arrive from the LLM. Implementations should buffer partial tokens
    and yield complete words or phrases for better UX.

    Example:
        >>> class StreamingLLM:
        ...     async def __call__(self, inputs, context) -> Output:
        ...         return {"content": "full response", "metadata": {}}
        ...
        ...     async def stream(self, inputs, context) -> AsyncGenerator[str, None]:
        ...         for token in ["Hello", " ", "world", "!"]:
        ...             yield token
    """

    def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Awaitable[Output]:
        """Call LLM asynchronously and return full output (batch mode).

        Args:
            inputs: List of Input messages (role, content)
            context: Metadata (mode, schema, etc.)

        Returns:
            Awaitable Output dict with content and metadata
        """
        ...

    def stream(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> "AsyncGenerator[str, None]":
        """Stream tokens from LLM as they're generated.

        Args:
            inputs: List of Input messages (role, content)
            context: Metadata (mode, schema, etc.)

        Yields:
            Token strings as they arrive from LLM

        Note:
            Must yield complete Output metadata as final yield or via separate channel.
            Implementations should catch exceptions and handle partial results gracefully.
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


# REPL Execution Types
class REPLIteration(TypedDict):
    """Record of a single REPL iteration (code + output).

    Stored in Output metadata for observability.
    """

    code: str
    output: str
    error: str | None
    iteration: int


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
    "UsageInfo",
    "Input",
    "Item",
    "Output",
    "ToolCall",
    "LLMCaller",
    "AsyncLLMCaller",
    "AsyncStreamingLLMCaller",
    "AsyncToolCaller",
    "REPLIteration",
    "SubTask",
    "PlannerDecision",
    "TraceObject",
]
