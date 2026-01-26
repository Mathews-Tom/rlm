from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict


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


# Internal Control Flow Types
class PlannerDecision(TypedDict):
    """Schema for planner LLM output.

    Used to decide whether to execute task atomically
    or decompose into sub-tasks.
    """

    thoughts: str
    decision: Literal["EXECUTE", "RECURSE"]
    sub_tasks: list[str]


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


# Custom Exceptions
class RLMError(Exception):
    """Base exception for py-rlm."""

    pass


class RecursionDepthError(RLMError):
    """Raised when max_depth exceeded."""

    pass


class MaxStepsError(RLMError):
    """Raised when max_steps exceeded."""

    pass


class InvalidJSONError(RLMError):
    """Raised when JSON parsing fails."""

    pass


class ExecutionError(RLMError):
    """Raised when LLM call fails."""

    pass
