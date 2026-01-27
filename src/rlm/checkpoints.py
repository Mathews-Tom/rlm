"""Checkpoint data model for execution state persistence.

Provides serialization and validation for resumable execution. Checkpoints
enable fault-tolerant long-running tasks by capturing execution state at
strategic points.

Example:
    >>> from rlm.checkpoints import Checkpoint
    >>> from rlm.memory import RLMContext, SharedMemory
    >>>
    >>> context = RLMContext(
    ...     task_id="task-1",
    ...     parent_id=None,
    ...     depth=0,
    ...     breadcrumbs=(),
    ...     memory_ref=SharedMemory(),
    ...     active_agent=None,
    ... )
    >>>
    >>> checkpoint = Checkpoint(
    ...     checkpoint_id="ckpt-123",
    ...     task="Analyze dataset",
    ...     context=context,
    ...     completed_steps=["load_data", "clean_data"],
    ...     pending_steps=["analyze", "report"],
    ...     results={"rows_loaded": 1000},
    ... )
    >>>
    >>> # Serialize to JSON
    >>> json_str = checkpoint.to_json()
    >>>
    >>> # Deserialize from JSON
    >>> restored = Checkpoint.from_json(json_str)
    >>> assert restored.checkpoint_id == checkpoint.checkpoint_id
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rlm.memory import RLMContext, SharedMemory


@dataclass
class Checkpoint:
    """Execution state snapshot for resumable tasks.

    Captures all state needed to resume execution after interruption.
    Serializes to JSON for storage in checkpoint stores.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        task: Original task description
        context: Execution context with depth and memory
        completed_steps: List of step descriptions already executed
        pending_steps: List of step descriptions remaining
        results: Dict of partial results collected so far
        timestamp: ISO 8601 timestamp of checkpoint creation

    Example:
        >>> checkpoint = Checkpoint(
        ...     checkpoint_id="ckpt-abc",
        ...     task="Process documents",
        ...     context=my_context,
        ...     completed_steps=["load", "parse"],
        ...     pending_steps=["analyze"],
        ...     results={"docs_loaded": 10},
        ... )
        >>> json_data = checkpoint.to_json()
    """

    checkpoint_id: str
    task: str
    context: RLMContext
    completed_steps: list[str]
    pending_steps: list[str]
    results: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate checkpoint fields after initialization.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id is required")
        if not self.task:
            raise ValueError("task is required")
        if not isinstance(self.completed_steps, list):
            raise ValueError("completed_steps must be a list")
        if not isinstance(self.pending_steps, list):
            raise ValueError("pending_steps must be a list")
        if not isinstance(self.results, dict):
            raise ValueError("results must be a dict")

        # Validate timestamp format
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {self.timestamp}") from e

    def to_json(self) -> str:
        """Serialize checkpoint to JSON string.

        Returns:
            JSON string representation of checkpoint

        Example:
            >>> checkpoint = Checkpoint(...)
            >>> json_str = checkpoint.to_json()
            >>> print(json_str)
            {"checkpoint_id": "ckpt-123", "task": "...", ...}
        """
        data = {
            "checkpoint_id": self.checkpoint_id,
            "task": self.task,
            "context": _serialize_context(self.context),
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "results": self.results,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> Checkpoint:
        """Deserialize checkpoint from JSON string.

        Args:
            json_str: JSON string representation of checkpoint

        Returns:
            Checkpoint instance

        Raises:
            ValueError: If JSON is invalid or missing required fields

        Example:
            >>> json_str = '{"checkpoint_id": "ckpt-123", ...}'
            >>> checkpoint = Checkpoint.from_json(json_str)
            >>> print(checkpoint.checkpoint_id)
            ckpt-123
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        # Validate required fields
        required_fields = [
            "checkpoint_id",
            "task",
            "context",
            "completed_steps",
            "pending_steps",
            "results",
            "timestamp",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        # Deserialize context
        context = _deserialize_context(data["context"])

        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            task=data["task"],
            context=context,
            completed_steps=data["completed_steps"],
            pending_steps=data["pending_steps"],
            results=data["results"],
            timestamp=data["timestamp"],
        )

    @staticmethod
    def create(
        task: str,
        context: RLMContext,
        completed_steps: list[str] | None = None,
        pending_steps: list[str] | None = None,
        results: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Factory method to create checkpoint with auto-generated ID.

        Args:
            task: Task description
            context: Execution context
            completed_steps: Steps already executed (default: empty list)
            pending_steps: Steps remaining (default: empty list)
            results: Partial results collected (default: empty dict)

        Returns:
            New Checkpoint instance with auto-generated checkpoint_id

        Example:
            >>> checkpoint = Checkpoint.create(
            ...     task="Process data",
            ...     context=my_context,
            ...     completed_steps=["load"],
            ...     pending_steps=["analyze", "report"],
            ... )
            >>> print(checkpoint.checkpoint_id)
            ckpt-a1b2c3d4-...
        """
        checkpoint_id = f"ckpt-{uuid.uuid4()}"
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            task=task,
            context=context,
            completed_steps=completed_steps or [],
            pending_steps=pending_steps or [],
            results=results or {},
        )


def _serialize_context(context: RLMContext) -> dict[str, Any]:
    """Serialize RLMContext to dict.

    Args:
        context: RLMContext to serialize

    Returns:
        Dict representation of context
    """
    return {
        "task_id": context.task_id,
        "parent_id": context.parent_id,
        "depth": context.depth,
        "breadcrumbs": list(context.breadcrumbs),
        "memory": _serialize_memory(context.memory_ref),
        "active_agent": context.active_agent,
    }


def _deserialize_context(data: dict[str, Any]) -> RLMContext:
    """Deserialize RLMContext from dict.

    Args:
        data: Dict representation of context

    Returns:
        RLMContext instance
    """
    memory_ref = _deserialize_memory(data["memory"])
    return RLMContext(
        task_id=data["task_id"],
        parent_id=data["parent_id"],
        depth=data["depth"],
        breadcrumbs=tuple(data["breadcrumbs"]),
        memory_ref=memory_ref,
        active_agent=data["active_agent"],
    )


def _serialize_memory(memory: SharedMemory) -> dict[str, Any]:
    """Serialize SharedMemory to dict.

    Args:
        memory: SharedMemory to serialize

    Returns:
        Dict representation of memory
    """
    return {"store": dict(memory._store)}


def _deserialize_memory(data: dict[str, Any]) -> SharedMemory:
    """Deserialize SharedMemory from dict.

    Args:
        data: Dict representation of memory

    Returns:
        SharedMemory instance
    """
    memory = SharedMemory()
    memory._store = data["store"]
    return memory


__all__ = [
    "Checkpoint",
]
