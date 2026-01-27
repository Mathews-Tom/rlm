"""Unit tests for Checkpoint data model and storage.

Tests cover:
- Checkpoint creation and validation
- JSON serialization/deserialization
- RLMContext serialization support
- SharedMemory serialization support
- Timestamp handling (ISO 8601 format)
- Factory method with auto-generated IDs
- Error handling for invalid data
- CheckpointStore protocol
- InMemoryCheckpointStore implementation
- Thread-safe concurrent operations
- TTL-based automatic cleanup
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from rlm.checkpoints import Checkpoint, InMemoryCheckpointStore
from rlm.memory import RLMContext, SharedMemory


class TestCheckpointCreation:
    """Test checkpoint creation and validation."""

    @pytest.fixture
    def sample_context(self) -> RLMContext:
        """Create sample execution context."""
        return RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    def test_create_checkpoint_with_all_fields(self, sample_context: RLMContext) -> None:
        """Test creating checkpoint with all required fields."""
        checkpoint = Checkpoint(
            checkpoint_id="ckpt-123",
            task="Analyze dataset",
            context=sample_context,
            completed_steps=["load_data", "clean_data"],
            pending_steps=["analyze", "report"],
            results={"rows_loaded": 1000},
        )

        assert checkpoint.checkpoint_id == "ckpt-123"
        assert checkpoint.task == "Analyze dataset"
        assert checkpoint.context == sample_context
        assert checkpoint.completed_steps == ["load_data", "clean_data"]
        assert checkpoint.pending_steps == ["analyze", "report"]
        assert checkpoint.results == {"rows_loaded": 1000}
        assert checkpoint.timestamp is not None

    def test_checkpoint_auto_timestamp(self, sample_context: RLMContext) -> None:
        """Test that timestamp is auto-generated if not provided."""
        checkpoint = Checkpoint(
            checkpoint_id="ckpt-123",
            task="Test task",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
        )

        # Verify timestamp is valid ISO 8601 format
        timestamp = checkpoint.timestamp
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_checkpoint_custom_timestamp(self, sample_context: RLMContext) -> None:
        """Test checkpoint with custom timestamp."""
        custom_timestamp = "2026-01-27T12:00:00+00:00"
        checkpoint = Checkpoint(
            checkpoint_id="ckpt-123",
            task="Test task",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp=custom_timestamp,
        )

        assert checkpoint.timestamp == custom_timestamp

    def test_missing_checkpoint_id_fails(self, sample_context: RLMContext) -> None:
        """Test that missing checkpoint_id raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_id is required"):
            Checkpoint(
                checkpoint_id="",
                task="Test task",
                context=sample_context,
                completed_steps=[],
                pending_steps=[],
                results={},
            )

    def test_missing_task_fails(self, sample_context: RLMContext) -> None:
        """Test that missing task raises ValueError."""
        with pytest.raises(ValueError, match="task is required"):
            Checkpoint(
                checkpoint_id="ckpt-123",
                task="",
                context=sample_context,
                completed_steps=[],
                pending_steps=[],
                results={},
            )

    def test_invalid_completed_steps_type(self, sample_context: RLMContext) -> None:
        """Test that non-list completed_steps raises ValueError."""
        with pytest.raises(ValueError, match="completed_steps must be a list"):
            Checkpoint(
                checkpoint_id="ckpt-123",
                task="Test task",
                context=sample_context,
                completed_steps="not a list",  # type: ignore
                pending_steps=[],
                results={},
            )

    def test_invalid_pending_steps_type(self, sample_context: RLMContext) -> None:
        """Test that non-list pending_steps raises ValueError."""
        with pytest.raises(ValueError, match="pending_steps must be a list"):
            Checkpoint(
                checkpoint_id="ckpt-123",
                task="Test task",
                context=sample_context,
                completed_steps=[],
                pending_steps="not a list",  # type: ignore
                results={},
            )

    def test_invalid_results_type(self, sample_context: RLMContext) -> None:
        """Test that non-dict results raises ValueError."""
        with pytest.raises(ValueError, match="results must be a dict"):
            Checkpoint(
                checkpoint_id="ckpt-123",
                task="Test task",
                context=sample_context,
                completed_steps=[],
                pending_steps=[],
                results="not a dict",  # type: ignore
            )

    def test_invalid_timestamp_format(self, sample_context: RLMContext) -> None:
        """Test that invalid timestamp format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO 8601 timestamp"):
            Checkpoint(
                checkpoint_id="ckpt-123",
                task="Test task",
                context=sample_context,
                completed_steps=[],
                pending_steps=[],
                results={},
                timestamp="not a valid timestamp",
            )


class TestCheckpointSerialization:
    """Test JSON serialization and deserialization."""

    @pytest.fixture
    def sample_checkpoint(self) -> Checkpoint:
        """Create sample checkpoint for serialization tests."""
        memory = SharedMemory()
        # Store some data in memory and track reference IDs
        ref1 = memory.store("value1")
        ref2 = memory.store("nested data content")

        context = RLMContext(
            task_id="task-1",
            parent_id="parent-1",
            depth=1,
            breadcrumbs=("root", "task-1"),
            memory_ref=memory,
            active_agent="agent-1",
        )

        return Checkpoint(
            checkpoint_id="ckpt-abc123",
            task="Process documents",
            context=context,
            completed_steps=["load", "parse"],
            pending_steps=["analyze", "report"],
            results={"docs_loaded": 10, "errors": 0, "ref1": ref1, "ref2": ref2},
            timestamp="2026-01-27T12:00:00+00:00",
        )

    def test_to_json(self, sample_checkpoint: Checkpoint) -> None:
        """Test checkpoint serialization to JSON."""
        json_str = sample_checkpoint.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["checkpoint_id"] == "ckpt-abc123"
        assert data["task"] == "Process documents"
        assert data["completed_steps"] == ["load", "parse"]
        assert data["pending_steps"] == ["analyze", "report"]
        assert "docs_loaded" in data["results"]
        assert data["results"]["docs_loaded"] == 10
        assert data["timestamp"] == "2026-01-27T12:00:00+00:00"

        # Verify context is serialized
        assert "context" in data
        assert data["context"]["task_id"] == "task-1"
        assert data["context"]["parent_id"] == "parent-1"
        assert data["context"]["depth"] == 1
        assert data["context"]["breadcrumbs"] == ["root", "task-1"]
        assert data["context"]["active_agent"] == "agent-1"

        # Verify memory is serialized
        assert "memory" in data["context"]
        assert "store" in data["context"]["memory"]
        # Memory store contains reference IDs -> content mapping
        assert isinstance(data["context"]["memory"]["store"], dict)

    def test_from_json(self, sample_checkpoint: Checkpoint) -> None:
        """Test checkpoint deserialization from JSON."""
        json_str = sample_checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.task == sample_checkpoint.task
        assert restored.completed_steps == sample_checkpoint.completed_steps
        assert restored.pending_steps == sample_checkpoint.pending_steps
        assert restored.results["docs_loaded"] == 10
        assert restored.timestamp == sample_checkpoint.timestamp

        # Verify context is restored
        assert restored.context.task_id == "task-1"
        assert restored.context.parent_id == "parent-1"
        assert restored.context.depth == 1
        assert restored.context.breadcrumbs == ("root", "task-1")
        assert restored.context.active_agent == "agent-1"

        # Verify memory is restored with same content
        ref1 = sample_checkpoint.results["ref1"]
        ref2 = sample_checkpoint.results["ref2"]
        assert restored.context.memory_ref.resolve(ref1) == "value1"
        assert restored.context.memory_ref.resolve(ref2) == "nested data content"

    def test_roundtrip_serialization(self, sample_checkpoint: Checkpoint) -> None:
        """Test that serialize -> deserialize preserves checkpoint."""
        json_str = sample_checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        # Verify all fields match
        assert restored.checkpoint_id == sample_checkpoint.checkpoint_id
        assert restored.task == sample_checkpoint.task
        assert restored.context.task_id == sample_checkpoint.context.task_id
        assert restored.completed_steps == sample_checkpoint.completed_steps
        assert restored.pending_steps == sample_checkpoint.pending_steps
        assert restored.results == sample_checkpoint.results
        assert restored.timestamp == sample_checkpoint.timestamp

    def test_from_json_invalid_json(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            Checkpoint.from_json("not valid json")

    def test_from_json_missing_checkpoint_id(self) -> None:
        """Test that missing checkpoint_id raises ValueError."""
        json_str = '{"task": "Test", "context": {}, "completed_steps": [], "pending_steps": [], "results": {}, "timestamp": "2026-01-27T12:00:00+00:00"}'
        with pytest.raises(ValueError, match="Missing required field: checkpoint_id"):
            Checkpoint.from_json(json_str)

    def test_from_json_missing_task(self) -> None:
        """Test that missing task raises ValueError."""
        json_str = '{"checkpoint_id": "ckpt-123", "context": {}, "completed_steps": [], "pending_steps": [], "results": {}, "timestamp": "2026-01-27T12:00:00+00:00"}'
        with pytest.raises(ValueError, match="Missing required field: task"):
            Checkpoint.from_json(json_str)


class TestCheckpointFactory:
    """Test factory method for creating checkpoints."""

    @pytest.fixture
    def sample_context(self) -> RLMContext:
        """Create sample execution context."""
        return RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    def test_create_with_auto_id(self, sample_context: RLMContext) -> None:
        """Test factory method generates checkpoint_id automatically."""
        checkpoint = Checkpoint.create(
            task="Process data",
            context=sample_context,
        )

        assert checkpoint.checkpoint_id.startswith("ckpt-")
        assert checkpoint.task == "Process data"
        assert checkpoint.context == sample_context
        assert checkpoint.completed_steps == []
        assert checkpoint.pending_steps == []
        assert checkpoint.results == {}

    def test_create_with_custom_steps(self, sample_context: RLMContext) -> None:
        """Test factory method with custom steps and results."""
        checkpoint = Checkpoint.create(
            task="Process data",
            context=sample_context,
            completed_steps=["load"],
            pending_steps=["analyze", "report"],
            results={"loaded": True},
        )

        assert checkpoint.checkpoint_id.startswith("ckpt-")
        assert checkpoint.completed_steps == ["load"]
        assert checkpoint.pending_steps == ["analyze", "report"]
        assert checkpoint.results == {"loaded": True}

    def test_create_generates_unique_ids(self, sample_context: RLMContext) -> None:
        """Test that factory method generates unique IDs."""
        checkpoint1 = Checkpoint.create(task="Task 1", context=sample_context)
        checkpoint2 = Checkpoint.create(task="Task 2", context=sample_context)

        assert checkpoint1.checkpoint_id != checkpoint2.checkpoint_id


class TestContextSerialization:
    """Test RLMContext serialization edge cases."""

    def test_serialize_context_with_none_parent(self) -> None:
        """Test serialization of context with None parent_id."""
        context = RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

        checkpoint = Checkpoint.create(task="Test", context=context)
        json_str = checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        assert restored.context.parent_id is None
        assert restored.context.active_agent is None

    def test_serialize_context_with_nested_breadcrumbs(self) -> None:
        """Test serialization of context with nested breadcrumbs."""
        context = RLMContext(
            task_id="task-3",
            parent_id="task-2",
            depth=2,
            breadcrumbs=("root", "task-1", "task-2", "task-3"),
            memory_ref=SharedMemory(),
            active_agent="agent-1",
        )

        checkpoint = Checkpoint.create(task="Test", context=context)
        json_str = checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        assert restored.context.breadcrumbs == ("root", "task-1", "task-2", "task-3")
        assert restored.context.depth == 2


class TestMemorySerialization:
    """Test SharedMemory serialization edge cases."""

    def test_serialize_empty_memory(self) -> None:
        """Test serialization of empty SharedMemory."""
        context = RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

        checkpoint = Checkpoint.create(task="Test", context=context)
        json_str = checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        # Memory should be empty (resolve returns empty string for nonexistent refs)
        assert restored.context.memory_ref.resolve("nonexistent") == ""

    def test_serialize_memory_with_complex_data(self) -> None:
        """Test serialization of SharedMemory with simple string content."""
        memory = SharedMemory()
        ref1 = memory.store("simple value")
        ref2 = memory.store("nested data content")
        ref3 = memory.store("list content")

        context = RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=memory,
            active_agent=None,
        )

        checkpoint = Checkpoint.create(task="Test", context=context)
        json_str = checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        # Verify data is preserved (SharedMemory stores string content)
        assert restored.context.memory_ref.resolve(ref1) == "simple value"
        assert restored.context.memory_ref.resolve(ref2) == "nested data content"
        assert restored.context.memory_ref.resolve(ref3) == "list content"


class TestInMemoryCheckpointStore:
    """Test in-memory checkpoint storage."""

    @pytest.fixture
    def sample_context(self) -> RLMContext:
        """Create sample execution context."""
        return RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.mark.asyncio
    async def test_save_and_load(self, sample_context: RLMContext) -> None:
        """Test saving and loading checkpoint."""
        store = InMemoryCheckpointStore()
        checkpoint = Checkpoint.create(task="Test task", context=sample_context)

        await store.save(checkpoint)
        loaded = await store.load(checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.task == checkpoint.task

    @pytest.mark.asyncio
    async def test_load_nonexistent(self) -> None:
        """Test loading non-existent checkpoint returns None."""
        store = InMemoryCheckpointStore()
        loaded = await store.load("nonexistent-id")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, sample_context: RLMContext) -> None:
        """Test deleting existing checkpoint."""
        store = InMemoryCheckpointStore()
        checkpoint = Checkpoint.create(task="Test task", context=sample_context)

        await store.save(checkpoint)
        result = await store.delete(checkpoint.checkpoint_id)

        assert result is True

        # Verify it's gone
        loaded = await store.load(checkpoint.checkpoint_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting non-existent checkpoint returns False."""
        store = InMemoryCheckpointStore()
        result = await store.delete("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        """Test listing empty store returns empty list."""
        store = InMemoryCheckpointStore()
        checkpoints = await store.list()

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_sorted_by_timestamp(self, sample_context: RLMContext) -> None:
        """Test list returns checkpoints sorted by timestamp (newest first)."""
        store = InMemoryCheckpointStore()

        # Create checkpoints with different timestamps
        cp1 = Checkpoint(
            checkpoint_id="cp1",
            task="Task 1",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp="2026-01-27T10:00:00+00:00",
        )

        cp2 = Checkpoint(
            checkpoint_id="cp2",
            task="Task 2",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp="2026-01-27T12:00:00+00:00",  # Newest
        )

        cp3 = Checkpoint(
            checkpoint_id="cp3",
            task="Task 3",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp="2026-01-27T11:00:00+00:00",
        )

        await store.save(cp1)
        await store.save(cp2)
        await store.save(cp3)

        checkpoints = await store.list()

        assert len(checkpoints) == 3
        assert checkpoints[0].checkpoint_id == "cp2"  # Newest
        assert checkpoints[1].checkpoint_id == "cp3"
        assert checkpoints[2].checkpoint_id == "cp1"  # Oldest

    @pytest.mark.asyncio
    async def test_clear(self, sample_context: RLMContext) -> None:
        """Test clearing all checkpoints."""
        store = InMemoryCheckpointStore()

        cp1 = Checkpoint.create(task="Task 1", context=sample_context)
        cp2 = Checkpoint.create(task="Task 2", context=sample_context)

        await store.save(cp1)
        await store.save(cp2)

        await store.clear()

        checkpoints = await store.list()
        assert checkpoints == []


class TestInMemoryCheckpointStoreTTL:
    """Test TTL-based automatic cleanup."""

    @pytest.fixture
    def sample_context(self) -> RLMContext:
        """Create sample execution context."""
        return RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.mark.asyncio
    async def test_ttl_cleanup_old_checkpoints(self, sample_context: RLMContext) -> None:
        """Test that old checkpoints are cleaned up based on TTL."""
        # Store with 1 second TTL
        store = InMemoryCheckpointStore(ttl_seconds=1)

        # Create old checkpoint (2 seconds ago)
        now = datetime.now(timezone.utc)
        old_timestamp = (now - timedelta(seconds=2)).isoformat()

        old_checkpoint = Checkpoint(
            checkpoint_id="old",
            task="Old task",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp=old_timestamp,
        )

        # Create recent checkpoint
        recent_checkpoint = Checkpoint.create(task="Recent task", context=sample_context)

        await store.save(old_checkpoint)
        await store.save(recent_checkpoint)

        # List should trigger cleanup and only return recent checkpoint
        checkpoints = await store.list()

        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint_id == recent_checkpoint.checkpoint_id

    @pytest.mark.asyncio
    async def test_load_expired_checkpoint_returns_none(self, sample_context: RLMContext) -> None:
        """Test that loading expired checkpoint returns None."""
        store = InMemoryCheckpointStore(ttl_seconds=1)

        # Create old checkpoint (2 seconds ago)
        now = datetime.now(timezone.utc)
        old_timestamp = (now - timedelta(seconds=2)).isoformat()

        old_checkpoint = Checkpoint(
            checkpoint_id="old",
            task="Old task",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp=old_timestamp,
        )

        await store.save(old_checkpoint)

        # Load should trigger cleanup and return None
        loaded = await store.load("old")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_no_ttl_means_no_expiry(self, sample_context: RLMContext) -> None:
        """Test that checkpoints don't expire when TTL is None."""
        store = InMemoryCheckpointStore(ttl_seconds=None)

        # Create old checkpoint (1 hour ago)
        now = datetime.now(timezone.utc)
        old_timestamp = (now - timedelta(hours=1)).isoformat()

        old_checkpoint = Checkpoint(
            checkpoint_id="old",
            task="Old task",
            context=sample_context,
            completed_steps=[],
            pending_steps=[],
            results={},
            timestamp=old_timestamp,
        )

        await store.save(old_checkpoint)

        # Should still be loadable
        loaded = await store.load("old")
        assert loaded is not None
        assert loaded.checkpoint_id == "old"


class TestInMemoryCheckpointStoreThreadSafety:
    """Test concurrent access thread safety."""

    @pytest.fixture
    def sample_context(self) -> RLMContext:
        """Create sample execution context."""
        return RLMContext(
            task_id="task-1",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, sample_context: RLMContext) -> None:
        """Test concurrent saves are thread-safe."""
        store = InMemoryCheckpointStore()

        # Create 10 checkpoints concurrently
        checkpoints = [
            Checkpoint.create(task=f"Task {i}", context=sample_context)
            for i in range(10)
        ]

        # Save all concurrently
        await asyncio.gather(*[store.save(cp) for cp in checkpoints])

        # Verify all saved
        loaded = await store.list()
        assert len(loaded) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, sample_context: RLMContext) -> None:
        """Test concurrent reads and writes don't cause data corruption."""
        store = InMemoryCheckpointStore()

        checkpoint = Checkpoint.create(task="Concurrent test", context=sample_context)
        await store.save(checkpoint)

        # Perform concurrent reads and writes
        async def read_checkpoint() -> None:
            for _ in range(10):
                await store.load(checkpoint.checkpoint_id)

        async def write_checkpoint() -> None:
            for _ in range(10):
                await store.save(checkpoint)

        # Run 5 readers and 5 writers concurrently
        tasks = [read_checkpoint() for _ in range(5)] + [write_checkpoint() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Verify checkpoint is still intact
        loaded = await store.load(checkpoint.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
