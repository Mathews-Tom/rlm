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

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

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


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends.

    Defines standard interface for pluggable checkpoint persistence.
    Implementations can use memory, disk, Redis, S3, or other storage.

    All methods are async to support I/O-bound storage operations.

    Example:
        >>> class MyStore:
        ...     async def save(self, checkpoint: Checkpoint) -> None:
        ...         # Store checkpoint
        ...         pass
        ...
        ...     async def load(self, checkpoint_id: str) -> Checkpoint | None:
        ...         # Load checkpoint
        ...         pass
        ...
        ...     async def delete(self, checkpoint_id: str) -> bool:
        ...         # Delete checkpoint
        ...         pass
        ...
        ...     async def list(self) -> list[Checkpoint]:
        ...         # List all checkpoints
        ...         pass
    """

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to storage.

        Args:
            checkpoint: Checkpoint to persist

        Raises:
            Exception: If save operation fails
        """
        ...

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from storage.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            Checkpoint if found, None otherwise

        Raises:
            Exception: If load operation fails
        """
        ...

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from storage.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            True if deleted, False if not found

        Raises:
            Exception: If delete operation fails
        """
        ...

    async def list(self) -> list[Checkpoint]:
        """List all checkpoints, sorted by timestamp (newest first).

        Returns:
            List of checkpoints sorted by timestamp descending

        Raises:
            Exception: If list operation fails
        """
        ...


class InMemoryCheckpointStore:
    """In-memory checkpoint storage for testing and development.

    Thread-safe implementation using asyncio.Lock for concurrent access.
    Supports automatic cleanup of old checkpoints via TTL.

    Attributes:
        ttl_seconds: Time-to-live for checkpoints in seconds (None = no expiry)

    Example:
        >>> store = InMemoryCheckpointStore(ttl_seconds=3600)
        >>> checkpoint = Checkpoint.create(task="Test", context=ctx)
        >>> await store.save(checkpoint)
        >>> loaded = await store.load(checkpoint.checkpoint_id)
        >>> assert loaded.checkpoint_id == checkpoint.checkpoint_id
    """

    def __init__(self, ttl_seconds: int | None = None) -> None:
        """Initialize in-memory checkpoint store.

        Args:
            ttl_seconds: Time-to-live for checkpoints in seconds.
                        None means no automatic expiry.
        """
        self._store: dict[str, Checkpoint] = {}
        self._lock = asyncio.Lock()
        self.ttl_seconds = ttl_seconds

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to memory.

        Thread-safe operation using asyncio.Lock.

        Args:
            checkpoint: Checkpoint to save
        """
        async with self._lock:
            self._store[checkpoint.checkpoint_id] = checkpoint

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from memory.

        Performs TTL cleanup before loading if TTL is configured.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            Checkpoint if found and not expired, None otherwise
        """
        # Clean expired checkpoints first
        await self._cleanup_expired()

        async with self._lock:
            return self._store.get(checkpoint_id)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory.

        Args:
            checkpoint_id: Unique identifier of checkpoint

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if checkpoint_id in self._store:
                del self._store[checkpoint_id]
                return True
            return False

    async def list(self) -> list[Checkpoint]:
        """List all non-expired checkpoints, sorted by timestamp (newest first).

        Performs TTL cleanup before listing if TTL is configured.

        Returns:
            List of checkpoints sorted by timestamp descending
        """
        # Clean expired checkpoints first
        await self._cleanup_expired()

        async with self._lock:
            checkpoints = list(self._store.values())

        # Sort by timestamp descending (newest first)
        checkpoints.sort(
            key=lambda cp: datetime.fromisoformat(cp.timestamp.replace("Z", "+00:00")),
            reverse=True,
        )
        return checkpoints

    async def _cleanup_expired(self) -> None:
        """Remove expired checkpoints based on TTL.

        Only runs if ttl_seconds is configured.
        """
        if self.ttl_seconds is None:
            return

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.ttl_seconds)

        async with self._lock:
            expired_ids = [
                cp_id
                for cp_id, cp in self._store.items()
                if datetime.fromisoformat(cp.timestamp.replace("Z", "+00:00")) < cutoff
            ]

            for cp_id in expired_ids:
                del self._store[cp_id]

    async def clear(self) -> None:
        """Clear all checkpoints from storage.

        Useful for testing and development.
        """
        async with self._lock:
            self._store.clear()


__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
]
