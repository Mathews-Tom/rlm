"""Unit tests for ToolCallingEngine.

Tests cover:
- Tool execution with retries
- Error handling and timeout behavior
- Iterative tool calling loop
- Max iterations enforcement
- Read-before-write pattern (sequential tool calls)
- Conversation history preservation
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any

from rlm.exceptions import ExecutionError
from rlm.memory import RLMContext, SharedMemory
from rlm.tools import Tool, ToolRegistry, ToolCallingEngine
from rlm.types import AsyncLLMCaller, Input, Output


class MockLLM:
    """Mock LLM for testing tool calling."""

    def __init__(self, responses: list[Output]) -> None:
        """Initialize with pre-configured responses.

        Args:
            responses: List of Output dicts to return in sequence
        """
        self.responses = responses
        self.call_count = 0
        self.call_history: list[tuple[list[Input], dict[str, Any]]] = []

    async def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Return next pre-configured response."""
        self.call_history.append((inputs, context))
        if self.call_count >= len(self.responses):
            # Return final response if out of pre-configured responses
            return {"content": "default response", "metadata": {}}

        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class TestToolCallingEngine:
    """Test ToolCallingEngine functionality."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create tool registry with test tools."""
        reg = ToolRegistry()

        # Add echo tool
        def echo(params: dict[str, Any]) -> str:
            return params["message"]

        reg.register(
            Tool(
                "echo",
                "Echo message",
                {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                echo,
            )
        )

        # Add calculator tool
        def calc(params: dict[str, Any]) -> str:
            expr = params["expression"]
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)

        reg.register(
            Tool(
                "calculator",
                "Calculate expression",
                {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                calc,
            )
        )

        return reg

    @pytest.fixture
    def context(self) -> RLMContext:
        """Create test execution context."""
        return RLMContext(
            task_id="test-task",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.mark.asyncio
    async def test_no_tool_calls(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test execution when LLM doesn't request tools."""
        # LLM returns final answer without tool calls
        mock_llm = MockLLM([
            {"content": "Final answer", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        assert result["content"] == "Final answer"
        assert result["metadata"]["tool_iterations"] == 0
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_single_tool_call(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test execution with single tool call."""
        # First call: LLM requests tool
        # Second call: LLM returns final answer after seeing tool result
        mock_llm = MockLLM([
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "echo", "arguments": {"message": "hello"}}],
            },
            {"content": "The tool returned: hello", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        assert result["content"] == "The tool returned: hello"
        assert result["metadata"]["tool_iterations"] == 1
        assert mock_llm.call_count == 2

        # Verify conversation history includes tool result
        second_call_inputs = mock_llm.call_history[1][0]
        assert len(second_call_inputs) == 4  # system, user, assistant, tool result
        assert "hello" in second_call_inputs[3]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test execution with multiple sequential tool calls."""
        mock_llm = MockLLM([
            # First iteration: request calculator
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}],
            },
            # Second iteration: request echo
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "echo", "arguments": {"message": "result is 4"}}],
            },
            # Third iteration: final answer
            {"content": "The calculation is complete", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        assert result["content"] == "The calculation is complete"
        assert result["metadata"]["tool_iterations"] == 2
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_handling(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test graceful handling of unknown tool requests."""
        mock_llm = MockLLM([
            # LLM requests unknown tool
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "unknown_tool", "arguments": {}}],
            },
            # LLM should receive error and provide final answer
            {"content": "Tool not available", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        # Should complete without crashing
        assert result["content"] == "Tool not available"

        # Verify error was injected into conversation
        second_call_inputs = mock_llm.call_history[1][0]
        error_message = second_call_inputs[3]["content"]
        assert "ERROR" in error_message
        assert "unknown_tool" in error_message

    @pytest.mark.asyncio
    async def test_max_iterations_enforcement(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test that max iterations limit is enforced."""
        # LLM keeps requesting tools indefinitely
        mock_llm = MockLLM([
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "1"}}]},
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "2"}}]},
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "3"}}]},
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "4"}}]},
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "5"}}]},
            {"content": "", "metadata": {}, "tool_calls": [{"name": "echo", "arguments": {"message": "6"}}]},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            max_tool_iterations=3,  # Limit to 3 iterations
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        # Should stop after 3 iterations
        assert result["metadata"]["tool_iterations"] == 3
        assert result["metadata"]["max_iterations_reached"] is True
        assert mock_llm.call_count == 3  # Should not call LLM more than max_iterations

    @pytest.mark.asyncio
    async def test_tool_execution_retry(self, registry: ToolRegistry, context: RLMContext) -> None:
        """Test tool execution retry logic for transient failures."""
        # Create tool that fails twice then succeeds
        attempts = [0]

        def flaky_tool(params: dict[str, Any]) -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("Transient failure")
            return "success"

        registry.register(
            Tool(
                "flaky",
                "Flaky tool",
                {"type": "object", "properties": {}},
                flaky_tool,
            )
        )

        mock_llm = MockLLM([
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "flaky", "arguments": {}}],
            },
            {"content": "Tool succeeded", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            max_tool_retries=2,  # Allow 2 retries
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        # Should succeed after retries
        assert result["content"] == "Tool succeeded"
        assert attempts[0] == 3  # Initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_tool_timeout_no_retry(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test that tool timeout errors are not retried."""
        async def slow_tool(params: dict[str, Any]) -> str:
            await asyncio.sleep(10)
            return "done"

        registry.register(
            Tool(
                "slow",
                "Slow tool",
                {"type": "object", "properties": {}},
                slow_tool,
            )
        )

        mock_llm = MockLLM([
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "slow", "arguments": {}}],
            },
            {"content": "Tool timed out", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            tool_timeout=0.1,  # Very short timeout
            max_tool_retries=2,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        # Should handle timeout gracefully
        assert "timed out" in result["content"]

        # Verify error was injected
        second_call_inputs = mock_llm.call_history[1][0]
        error_message = second_call_inputs[3]["content"]
        assert "ERROR" in error_message
        assert "timed out" in error_message

    @pytest.mark.asyncio
    async def test_read_before_write_pattern(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test read-before-write pattern with sequential tool calls.

        This simulates an LLM that:
        1. Reads data with one tool
        2. Processes it
        3. Writes result with another tool
        """
        # Track tool call order
        call_order: list[str] = []

        def read_tool(params: dict[str, Any]) -> str:
            call_order.append("read")
            return "data from file"

        def write_tool(params: dict[str, Any]) -> str:
            call_order.append("write")
            return "written successfully"

        registry.register(
            Tool(
                "read",
                "Read data",
                {"type": "object", "properties": {}},
                read_tool,
            )
        )
        registry.register(
            Tool(
                "write",
                "Write data",
                {"type": "object", "properties": {"data": {"type": "string"}}},
                write_tool,
            )
        )

        mock_llm = MockLLM([
            # First: read
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "read", "arguments": {}}],
            },
            # Second: write (after seeing read result)
            {
                "content": "",
                "metadata": {},
                "tool_calls": [{"name": "write", "arguments": {"data": "processed"}}],
            },
            # Third: final answer
            {"content": "Data processed successfully", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Test task", context)

        # Verify correct order
        assert call_order == ["read", "write"]
        assert result["content"] == "Data processed successfully"

    @pytest.mark.asyncio
    async def test_conversation_history_preserved(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test that conversation history is preserved across iterations."""
        mock_llm = MockLLM([
            {
                "content": "Let me use a tool",
                "metadata": {},
                "tool_calls": [{"name": "echo", "arguments": {"message": "test"}}],
            },
            {"content": "Based on the result", "metadata": {}},
        ])

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        await engine._execute_leaf_async("Test task", context)

        # Check conversation history in second call
        second_call_inputs = mock_llm.call_history[1][0]

        # Should have: system, user, assistant (with tool request), user (with tool result)
        assert len(second_call_inputs) == 4
        assert second_call_inputs[0]["role"] == "system"
        assert second_call_inputs[1]["role"] == "user"
        assert second_call_inputs[2]["role"] == "assistant"
        assert second_call_inputs[3]["role"] == "user"

        # Verify tool result is in conversation
        assert "test" in second_call_inputs[3]["content"]

    @pytest.mark.asyncio
    async def test_llm_call_failure(
        self, registry: ToolRegistry, context: RLMContext
    ) -> None:
        """Test handling of LLM call failures."""
        async def failing_llm(inputs: list[Input], context: dict[str, Any]) -> Output:
            raise RuntimeError("LLM API failed")

        engine = ToolCallingEngine(
            llm=failing_llm,
            tool_registry=registry,
            max_depth=1,
            verbose=False,
        )

        with pytest.raises(ExecutionError, match="LLM call failed"):
            await engine._execute_leaf_async("Test task", context)
