"""Streaming event protocol for progressive results via AsyncGenerator.

This module defines the StreamEvent dataclass and event type system used by
StreamingEngine to emit real-time updates during task execution. Events are
JSON-serializable for transport over SSE (Server-Sent Events) or WebSocket.

Event Types:
    - plan: Emitted after planning decision (decompose vs execute)
    - token: Emitted during LLM token generation (real-time streaming)
    - result: Emitted when a sub-task or leaf task completes
    - error: Emitted when an error occurs during execution

Example:
    >>> from rlm.streaming import StreamEvent
    >>> event = StreamEvent(
    ...     type="token",
    ...     data={"content": "Hello"},
    ...     metadata={"depth": 0, "task_id": "task-123", "timestamp": "2026-01-27T16:30:00Z"}
    ... )
    >>> json_str = event.to_json()
    >>> print(json_str)
    '{"type": "token", "data": {"content": "Hello"}, "metadata": {...}}'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

EventType = Literal["plan", "token", "result", "error"]


@dataclass
class StreamEvent:
    """Event emitted during streaming execution.

    Attributes:
        type: Event type (plan, token, result, error)
        data: Event payload (varies by type)
        metadata: Execution metadata (depth, task_id, timestamp)

    Example:
        >>> event = StreamEvent(
        ...     type="plan",
        ...     data={"decision": "decompose", "sub_tasks": ["task1", "task2"]},
        ...     metadata={"depth": 0, "task_id": "root", "timestamp": "2026-01-27T16:30:00Z"}
        ... )
        >>> event.type
        'plan'
    """

    type: EventType
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event structure after initialization.

        Ensures required metadata fields are present and adds timestamp if missing.

        Raises:
            ValueError: If type is not a valid EventType
            ValueError: If required metadata fields are missing
        """
        # Validate event type
        valid_types: tuple[str, ...] = ("plan", "token", "result", "error")
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid event type '{self.type}'. Must be one of: {valid_types}"
            )

        # Ensure required metadata fields
        if "task_id" not in self.metadata:
            raise ValueError("Metadata must include 'task_id'")

        if "depth" not in self.metadata:
            raise ValueError("Metadata must include 'depth'")

        # Add timestamp if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        """Serialize event to JSON string for transport.

        Returns:
            JSON string representation of event

        Example:
            >>> event = StreamEvent(
            ...     type="token",
            ...     data={"content": "Hello"},
            ...     metadata={"depth": 0, "task_id": "task-123"}
            ... )
            >>> json_str = event.to_json()
            >>> isinstance(json_str, str)
            True
        """
        return json.dumps(
            {"type": self.type, "data": self.data, "metadata": self.metadata},
            separators=(",", ":"),  # Compact JSON
        )

    @classmethod
    def from_json(cls, json_str: str) -> StreamEvent:
        """Deserialize event from JSON string.

        Args:
            json_str: JSON string to deserialize

        Returns:
            StreamEvent instance

        Raises:
            ValueError: If JSON is invalid or missing required fields

        Example:
            >>> json_str = '{"type":"token","data":{"content":"Hi"},"metadata":{"depth":0,"task_id":"t1"}}'
            >>> event = StreamEvent.from_json(json_str)
            >>> event.type
            'token'
        """
        try:
            data = json.loads(json_str)
            return cls(
                type=data["type"], data=data["data"], metadata=data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid StreamEvent JSON: {e}") from e

    @classmethod
    def plan_event(
        cls, decision: str, sub_tasks: list[str] | None, task_id: str, depth: int
    ) -> StreamEvent:
        """Create a plan event after planning decision.

        Args:
            decision: Planning decision ("decompose" or "execute")
            sub_tasks: List of sub-task descriptions (None for execute)
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="plan"

        Example:
            >>> event = StreamEvent.plan_event(
            ...     decision="decompose",
            ...     sub_tasks=["task1", "task2"],
            ...     task_id="root",
            ...     depth=0
            ... )
            >>> event.type
            'plan'
            >>> event.data["decision"]
            'decompose'
        """
        return cls(
            type="plan",
            data={
                "decision": decision,
                "sub_tasks": sub_tasks,
            },
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def token_event(cls, content: str, task_id: str, depth: int) -> StreamEvent:
        """Create a token event during LLM generation.

        Args:
            content: Token or partial text content
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="token"

        Example:
            >>> event = StreamEvent.token_event(
            ...     content="Hello",
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'token'
            >>> event.data["content"]
            'Hello'
        """
        return cls(
            type="token",
            data={"content": content},
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def result_event(
        cls, content: str, metadata: dict[str, Any], task_id: str, depth: int
    ) -> StreamEvent:
        """Create a result event when task completes.

        Args:
            content: Final result content
            metadata: Result metadata (tokens, tool_calls, etc.)
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="result"

        Example:
            >>> event = StreamEvent.result_event(
            ...     content="Task complete",
            ...     metadata={"tokens": 150},
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'result'
            >>> event.data["content"]
            'Task complete'
        """
        return cls(
            type="result",
            data={"content": content, "result_metadata": metadata},
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def error_event(
        cls, error: str, error_type: str, task_id: str, depth: int
    ) -> StreamEvent:
        """Create an error event when execution fails.

        Args:
            error: Error message
            error_type: Error type/class name
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="error"

        Example:
            >>> event = StreamEvent.error_event(
            ...     error="Connection timeout",
            ...     error_type="TimeoutError",
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'error'
            >>> event.data["error"]
            'Connection timeout'
        """
        return cls(
            type="error",
            data={
                "error": error,
                "error_type": error_type,
            },
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )


__all__ = [
    "EventType",
    "StreamEvent",
]
