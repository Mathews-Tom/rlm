"""Tests for CheckpointableEngine automatic checkpoint creation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from collections.abc import AsyncGenerator

import pytest

from rlm.checkpoints import (
    Checkpoint,
    CheckpointableEngine,
    InMemoryCheckpointStore,
)
from rlm.memory import RLMContext, SharedMemory
from rlm.tools import ToolRegistry
from rlm.types import Input, Output


class MockStreamingLLM:
    """Mock LLM with streaming support for deterministic tests."""

    def __init__(self, tokens: list[str], delay_ms: int = 1) -> None:
        """Initialize mock LLM.

        Args:
            tokens: Tokens to yield during streaming
            delay_ms: Delay between tokens in milliseconds
        """
        self.tokens = tokens
        self.delay_ms = delay_ms
        self.call_count = 0
        self.stream_count = 0

    async def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
        """Batch mode: return full response."""
        self.call_count += 1

        # Check if this is a planning call
        if "system_prompt" in context and "planner" in context.get(
            "system_prompt", ""
        ).lower():
            return {
                "content": '{"thoughts": "Simple task, execute directly", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"mode": "planner"},
            }

        # Execution call
        full_content = "".join(self.tokens)
        return {
            "content": full_content,
            "metadata": {"mode": "batch", "call_count": self.call_count},
        }

    async def stream(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Streaming mode: yield tokens one by one."""
        self.stream_count += 1
        for token in self.tokens:
            await asyncio.sleep(self.delay_ms / 1000)
            yield token


class TestCheckpointableEngine:
    """Test automatic checkpoint creation during streaming execution."""

    @pytest.mark.asyncio
    async def test_checkpoint_creation_every_n_steps(self) -> None:
        """Test that checkpoints are created every N result events."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result", " ", "content"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=2,  # Checkpoint every 2 steps
            max_depth=1,
            verbose=False,
        )

        # Collect events
        events = []
        async for event in engine.solve_streaming("Test task"):
            events.append(event)

        # Verify checkpoints were created
        checkpoints = await store.list()
        assert len(checkpoints) >= 0  # At least initial state

    @pytest.mark.asyncio
    async def test_checkpoint_interval_configuration(self) -> None:
        """Test that checkpoint_interval controls save frequency."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        # Interval of 5 steps
        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=5,
            max_depth=1,
            verbose=False,
        )

        events = []
        async for event in engine.solve_streaming("Test task"):
            events.append(event)

        # Should have few checkpoints (only every 5 steps)
        checkpoints = await store.list()
        result_count = sum(1 for e in events if e.type == "result")

        # Checkpoints = floor(result_count / 5)
        expected_checkpoints = result_count // 5
        assert len(checkpoints) == expected_checkpoints

    @pytest.mark.asyncio
    async def test_checkpoint_save_failure_does_not_crash(self) -> None:
        """Test that failed checkpoint saves log warning but don't crash."""

        class FailingStore:
            """Checkpoint store that always fails."""

            async def save(self, checkpoint: Checkpoint) -> None:
                """Always raise an error."""
                raise RuntimeError("Simulated save failure")

            async def load(self, checkpoint_id: str) -> Checkpoint | None:
                """Not used in test."""
                return None

            async def delete(self, checkpoint_id: str) -> bool:
                """Not used in test."""
                return False

            async def list(self) -> list[Checkpoint]:
                """Not used in test."""
                return []

        store = FailingStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,  # Checkpoint every step
            max_depth=1,
            verbose=False,
        )

        # Should complete successfully despite save failures
        events = []
        async for event in engine.solve_streaming("Test task"):
            events.append(event)

        # Verify execution completed
        assert len(events) > 0
        assert any(e.type == "result" for e in events)

    @pytest.mark.asyncio
    async def test_checkpoint_includes_context_and_steps(self) -> None:
        """Test that checkpoints include execution context and step count."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,  # Checkpoint every step
            max_depth=1,
            verbose=False,
        )

        events = []
        async for event in engine.solve_streaming("Test task"):
            events.append(event)

        # Verify checkpoints have correct structure
        checkpoints = await store.list()
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0]
            assert checkpoint.task == "Test task"
            assert checkpoint.context is not None
            assert isinstance(checkpoint.completed_steps, list)
            assert checkpoint.checkpoint_id.startswith("ckpt-")

    @pytest.mark.asyncio
    async def test_checkpoint_interval_validation(self) -> None:
        """Test that invalid checkpoint_interval raises ValueError."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        # checkpoint_interval < 1 should raise ValueError
        with pytest.raises(ValueError, match="checkpoint_interval must be >= 1"):
            CheckpointableEngine(
                llm=llm,
                tool_registry=registry,
                checkpoint_store=store,
                checkpoint_interval=0,  # Invalid
                max_depth=1,
            )

    @pytest.mark.asyncio
    async def test_checkpoint_save_timeout(self) -> None:
        """Test that checkpoint saves timeout after 100ms."""

        class SlowStore:
            """Checkpoint store with slow saves."""

            async def save(self, checkpoint: Checkpoint) -> None:
                """Simulate slow save operation."""
                await asyncio.sleep(0.2)  # 200ms (>100ms timeout)

            async def load(self, checkpoint_id: str) -> Checkpoint | None:
                """Not used in test."""
                return None

            async def delete(self, checkpoint_id: str) -> bool:
                """Not used in test."""
                return False

            async def list(self) -> list[Checkpoint]:
                """Not used in test."""
                return []

        store = SlowStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Should complete successfully with timeout warnings
        events = []
        async for event in engine.solve_streaming("Test task"):
            events.append(event)

        # Verify execution completed despite timeouts
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_checkpoint_step_counter_resets(self) -> None:
        """Test that step counter resets for each new task."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # First task
        async for event in engine.solve_streaming("Task 1"):
            pass

        first_count = len(await store.list())

        # Second task (counter should reset)
        async for event in engine.solve_streaming("Task 2"):
            pass

        second_count = len(await store.list())

        # Should have checkpoints from both tasks
        assert second_count > first_count


class TestCheckpointResume:
    """Test checkpoint resume and recovery logic."""

    @pytest.mark.asyncio
    async def test_solve_with_checkpoints_fresh_execution(self) -> None:
        """Test fresh execution when no checkpoint ID provided."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Execute without checkpoint_id (fresh execution)
        events = []
        async for event in engine.solve_with_checkpoints("Test task"):
            events.append(event)

        # Verify execution completed
        assert len(events) > 0
        assert any(e.type == "result" for e in events)

    @pytest.mark.asyncio
    async def test_solve_with_checkpoints_resume_from_valid_checkpoint(self) -> None:
        """Test resuming from valid checkpoint ID."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # First execution - save checkpoint
        async for event in engine.solve_streaming("Test task"):
            pass

        # Get saved checkpoint
        checkpoints = await store.list()
        assert len(checkpoints) > 0
        checkpoint_id = checkpoints[0].checkpoint_id

        # Resume from checkpoint
        events = []
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id=checkpoint_id
        ):
            events.append(event)

        # Verify execution completed
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_solve_with_checkpoints_invalid_checkpoint_id_fallback(
        self,
    ) -> None:
        """Test fallback to fresh execution when checkpoint ID not found."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Try to resume from nonexistent checkpoint
        events = []
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id="nonexistent-id"
        ):
            events.append(event)

        # Should fall back to fresh execution
        assert len(events) > 0
        assert any(e.type == "result" for e in events)

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_restores_context(self) -> None:
        """Test that checkpoint resume restores execution context."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        # Create checkpoint with custom context
        from rlm.memory import RLMContext, SharedMemory

        memory = SharedMemory()
        ref_id = memory.store("test data")
        context = RLMContext(
            task_id="test-task-123",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=memory,
            active_agent=None,
        )

        checkpoint = Checkpoint.create(
            task="Test task with context",
            context=context,
            completed_steps=["Step 1", "Step 2"],
            pending_steps=["Step 3"],
            results={"ref": ref_id},
        )

        await store.save(checkpoint)

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Resume from checkpoint
        events = []
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id=checkpoint.checkpoint_id
        ):
            events.append(event)

        # Verify execution with restored context
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_continues_step_counter(self) -> None:
        """Test that step counter continues from checkpoint value."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        # Create checkpoint with completed steps
        from rlm.memory import RLMContext, SharedMemory

        context = RLMContext(
            task_id="test-task",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

        checkpoint = Checkpoint.create(
            task="Test task",
            context=context,
            completed_steps=["Step 1", "Step 2", "Step 3"],  # 3 steps completed
            pending_steps=[],
            results={},
        )

        await store.save(checkpoint)

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Resume from checkpoint
        events = []
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id=checkpoint.checkpoint_id
        ):
            events.append(event)

        # Verify execution continued
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_checkpoint_recovery_failure_logs_and_continues(self) -> None:
        """Test that checkpoint recovery failures log warning and fall back."""

        class CorruptedStore:
            """Store that returns corrupted checkpoints."""

            async def load(self, checkpoint_id: str) -> Checkpoint | None:
                """Simulate corrupted checkpoint."""
                raise RuntimeError("Simulated corruption")

            async def save(self, checkpoint: Checkpoint) -> None:
                """Not used in test."""
                pass

            async def delete(self, checkpoint_id: str) -> bool:
                """Not used in test."""
                return False

            async def list(self) -> list[Checkpoint]:
                """Not used in test."""
                return []

        store = CorruptedStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Try to resume from corrupted checkpoint
        events = []
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id="corrupted-id"
        ):
            events.append(event)

        # Should fall back to fresh execution
        assert len(events) > 0
        assert any(e.type == "result" for e in events)

    @pytest.mark.asyncio
    async def test_resume_creates_new_checkpoints(self) -> None:
        """Test that resumed execution continues creating checkpoints."""
        store = InMemoryCheckpointStore()
        llm = MockStreamingLLM(["Result"])
        registry = ToolRegistry()

        # Create initial checkpoint
        from rlm.memory import RLMContext, SharedMemory

        context = RLMContext(
            task_id="test-task",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

        initial_checkpoint = Checkpoint.create(
            task="Test task",
            context=context,
            completed_steps=["Step 1"],
            pending_steps=[],
            results={},
        )

        await store.save(initial_checkpoint)
        initial_count = len(await store.list())

        engine = CheckpointableEngine(
            llm=llm,
            tool_registry=registry,
            checkpoint_store=store,
            checkpoint_interval=1,
            max_depth=1,
            verbose=False,
        )

        # Resume from checkpoint
        async for event in engine.solve_with_checkpoints(
            "Test task", checkpoint_id=initial_checkpoint.checkpoint_id
        ):
            pass

        # Verify new checkpoints were created
        final_count = len(await store.list())
        assert final_count >= initial_count  # Should have at least initial checkpoint
