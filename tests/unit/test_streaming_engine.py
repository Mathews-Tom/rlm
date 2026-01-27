"""Unit tests for StreamingEngine with mock streaming LLMs.

Tests cover:
- Token-level streaming with mock LLMs
- Event emission sequence (plan → token → result)
- Fallback to batch mode for non-streaming LLMs
- Error handling during streaming
- Partial content recovery on failures
- Streaming integration with tool calling
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from rlm.memory import RLMContext, SharedMemory
from rlm.streaming import StreamEvent, StreamingEngine
from rlm.tools import ToolRegistry
from rlm.types import Input, Output


class MockStreamingLLM:
    """Mock LLM with streaming support for deterministic tests."""

    def __init__(self, tokens: list[str], delay_ms: int = 10) -> None:
        """Initialize with pre-configured tokens to stream.

        Args:
            tokens: List of token strings to yield during streaming
            delay_ms: Delay between tokens in milliseconds (simulates TTFT)
        """
        self.tokens = tokens
        self.delay_ms = delay_ms
        self.call_count = 0
        self.stream_count = 0

    async def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Batch mode: return full response.

        If called as planner (system_prompt in context), returns planning JSON.
        Otherwise returns the full token content.
        """
        self.call_count += 1

        # Check if this is a planning call
        if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
            # Return planning JSON for execute decision
            return {
                "content": '{"thoughts": "Simple task, execute directly", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"mode": "planner"},
            }

        # Normal execution: return full token content
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
            await asyncio.sleep(self.delay_ms / 1000)  # Simulate TTFT
            yield token


class MockBatchOnlyLLM:
    """Mock LLM without streaming support (batch only)."""

    async def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Only supports batch mode."""
        # Check if this is a planning call
        if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
            return {
                "content": '{"thoughts": "Simple task, execute directly", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"mode": "planner"},
            }

        return {
            "content": "Batch mode response",
            "metadata": {"mode": "batch"},
        }


class MockFailingStreamLLM:
    """Mock LLM that fails during streaming."""

    def __init__(self, fail_after_tokens: int) -> None:
        """Initialize with failure configuration.

        Args:
            fail_after_tokens: Number of tokens to yield before failing
        """
        self.fail_after_tokens = fail_after_tokens
        self.tokens_yielded = 0

    async def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Batch mode always succeeds."""
        # Check if this is a planning call
        if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
            return {
                "content": '{"thoughts": "Simple task, execute directly", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"mode": "planner"},
            }

        return {
            "content": "Success",
            "metadata": {},
        }

    async def stream(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Streaming mode fails after N tokens."""
        self.tokens_yielded = 0
        for token in ["Hello", " ", "world", "!"]:
            if self.tokens_yielded >= self.fail_after_tokens:
                raise RuntimeError("Simulated streaming failure")
            yield token
            self.tokens_yielded += 1
            await asyncio.sleep(0.01)


class TestStreamingEngineTokenStreaming:
    """Test token-level streaming with mock LLMs."""

    @pytest.fixture
    def context(self) -> RLMContext:
        """Create test execution context."""
        return RLMContext(
            task_id="test-streaming",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create empty tool registry."""
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_streaming_llm_emits_token_events(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test that streaming LLM emits token events."""
        mock_llm = MockStreamingLLM(["Hello", " ", "world", "!"])

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        events: list[StreamEvent] = []
        async for event in engine.solve_streaming("Test task", context):
            events.append(event)

        # Should emit: plan, token (x4), result
        assert len(events) >= 6

        # Check event types in sequence
        assert events[0].type == "plan"
        assert events[0].data["decision"] == "execute"

        # Next 4 should be token events
        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 4
        assert [e.data["content"] for e in token_events] == ["Hello", " ", "world", "!"]

        # Last should be result event
        result_events = [e for e in events if e.type == "result"]
        assert len(result_events) == 1
        assert result_events[0].data["content"] == "Hello world!"
        assert result_events[0].data["result_metadata"]["streaming"] is True

    @pytest.mark.asyncio
    async def test_batch_llm_fallback(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test fallback to batch mode for non-streaming LLMs."""
        mock_llm = MockBatchOnlyLLM()

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        events: list[StreamEvent] = []
        async for event in engine.solve_streaming("Test task", context):
            events.append(event)

        # Should emit: plan, result (no token events)
        assert len(events) == 2
        assert events[0].type == "plan"
        assert events[1].type == "result"
        assert events[1].data["content"] == "Batch mode response"

        # Verify no token events emitted
        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 0

    @pytest.mark.asyncio
    async def test_streaming_error_handling(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test error handling during streaming with partial content."""
        mock_llm = MockFailingStreamLLM(fail_after_tokens=2)

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        events: list[StreamEvent] = []
        with pytest.raises(RuntimeError, match="Simulated streaming failure"):
            async for event in engine.solve_streaming("Test task", context):
                events.append(event)

        # Should emit: plan, token (x2), error
        assert len(events) >= 3

        # Check we got plan and some tokens
        assert events[0].type == "plan"
        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 2
        assert token_events[0].data["content"] == "Hello"
        assert token_events[1].data["content"] == " "

        # Last should be error event with partial content
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Streaming failed" in error_events[0].data["error"]
        assert "partial content" in error_events[0].data["error"]

    @pytest.mark.asyncio
    async def test_streaming_ttft_performance(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test time-to-first-token is under 500ms."""
        import time

        mock_llm = MockStreamingLLM(["Fast", "!"], delay_ms=50)

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        start_time = time.time()
        first_token_time: float | None = None

        async for event in engine.solve_streaming("Test task", context):
            if event.type == "token" and first_token_time is None:
                first_token_time = time.time()
                break

        assert first_token_time is not None
        ttft_ms = (first_token_time - start_time) * 1000

        # TTFT should be under 500ms (in practice it's ~50ms)
        assert ttft_ms < 500, f"TTFT was {ttft_ms:.1f}ms, expected <500ms"

    @pytest.mark.asyncio
    async def test_streaming_metadata_tracking(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test that streaming metadata is tracked correctly."""
        mock_llm = MockStreamingLLM(["Test", " ", "streaming"])

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        events: list[StreamEvent] = []
        async for event in engine.solve_streaming("Test task", context):
            events.append(event)

        # Get result event
        result_events = [e for e in events if e.type == "result"]
        assert len(result_events) == 1

        # Check metadata
        result = result_events[0]
        assert result.data["result_metadata"]["streaming"] is True
        assert result.data["result_metadata"]["token_count"] == 3
        assert result.metadata["task_id"] == "test-streaming"
        assert result.metadata["depth"] == 0

    @pytest.mark.asyncio
    async def test_streaming_with_empty_response(
        self, context: RLMContext, registry: ToolRegistry
    ) -> None:
        """Test streaming with empty LLM response."""
        mock_llm = MockStreamingLLM([])  # No tokens

        engine = StreamingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        events: list[StreamEvent] = []
        async for event in engine.solve_streaming("Test task", context):
            events.append(event)

        # Should emit: plan, result (no token events)
        assert len(events) == 2
        assert events[0].type == "plan"
        assert events[1].type == "result"
        assert events[1].data["content"] == ""
