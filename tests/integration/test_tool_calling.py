"""Integration tests for tool calling with real-world scenarios.

These tests demonstrate end-to-end tool calling workflows.
Tests with real LLM APIs are marked with @pytest.mark.integration
and can be skipped if API credentials are not available.

Tests cover:
- End-to-end tool calling workflow
- Real tool examples (calculator, file operations)
- Error recovery scenarios
- Multi-step workflows
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from rlm.memory import RLMContext, SharedMemory
from rlm.tools import ToolRegistry, ToolCallingEngine
from rlm.tools.examples import (
    create_calculator_tool,
    create_file_read_tool,
    create_file_write_tool,
    create_current_time_tool,
)
from rlm.types import Input, Output


class MockToolCallingLLM:
    """Mock LLM that simulates realistic tool calling behavior."""

    def __init__(self, scenario: str) -> None:
        """Initialize with specific test scenario."""
        self.scenario = scenario
        self.call_count = 0

    async def __call__(
        self, inputs: list[Input], context: dict[str, Any]
    ) -> Output:
        """Simulate tool calling based on scenario."""
        self.call_count += 1

        if self.scenario == "calculator":
            return self._calculator_scenario(inputs)
        elif self.scenario == "file_operations":
            return self._file_operations_scenario(inputs)
        elif self.scenario == "current_time":
            return self._current_time_scenario(inputs)
        else:
            return {"content": "Unknown scenario", "metadata": {}}

    def _calculator_scenario(self, inputs: list[Input]) -> Output:
        """Simulate calculator workflow."""
        if self.call_count == 1:
            # First call: request calculation
            return {
                "content": "Let me calculate that",
                "metadata": {},
                "tool_calls": [
                    {
                        "name": "calculator",
                        "arguments": {"expression": "(10 + 5) * 2"},
                    }
                ],
            }
        else:
            # Second call: provide final answer after seeing result
            # Extract tool result from conversation history
            return {
                "content": "The calculation result is 30",
                "metadata": {},
            }

    def _file_operations_scenario(self, inputs: list[Input]) -> Output:
        """Simulate file read-write workflow."""
        if self.call_count == 1:
            # Write file first
            return {
                "content": "",
                "metadata": {},
                "tool_calls": [
                    {
                        "name": "file_write",
                        "arguments": {
                            "path": "test.txt",
                            "content": "Hello, World!",
                        },
                    }
                ],
            }
        elif self.call_count == 2:
            # Then read it back
            return {
                "content": "",
                "metadata": {},
                "tool_calls": [
                    {"name": "file_read", "arguments": {"path": "test.txt"}}
                ],
            }
        else:
            # Final answer
            return {
                "content": "File operations completed successfully",
                "metadata": {},
            }

    def _current_time_scenario(self, inputs: list[Input]) -> Output:
        """Simulate getting current time."""
        if self.call_count == 1:
            return {
                "content": "",
                "metadata": {},
                "tool_calls": [
                    {"name": "current_time", "arguments": {"format": "iso"}}
                ],
            }
        else:
            return {
                "content": "Current time retrieved successfully",
                "metadata": {},
            }


class TestToolCallingIntegration:
    """Integration tests for tool calling workflows."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry_with_examples(self, temp_dir: Path) -> ToolRegistry:
        """Create registry with example tools."""
        registry = ToolRegistry()
        registry.register(create_calculator_tool())
        registry.register(create_file_read_tool(base_dir=str(temp_dir)))
        registry.register(create_file_write_tool(base_dir=str(temp_dir)))
        registry.register(create_current_time_tool())
        return registry

    @pytest.fixture
    def context(self) -> RLMContext:
        """Create test execution context."""
        return RLMContext(
            task_id="integration-test",
            parent_id=None,
            depth=0,
            breadcrumbs=(),
            memory_ref=SharedMemory(),
            active_agent=None,
        )

    @pytest.mark.asyncio
    async def test_calculator_workflow(
        self, registry_with_examples: ToolRegistry, context: RLMContext
    ) -> None:
        """Test calculator tool integration."""
        mock_llm = MockToolCallingLLM("calculator")

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry_with_examples,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async(
            "Calculate (10 + 5) * 2", context
        )

        assert "30" in result["content"]
        assert result["metadata"]["tool_iterations"] == 1

    @pytest.mark.asyncio
    async def test_file_operations_workflow(
        self,
        registry_with_examples: ToolRegistry,
        context: RLMContext,
        temp_dir: Path,
    ) -> None:
        """Test file read/write workflow."""
        mock_llm = MockToolCallingLLM("file_operations")

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry_with_examples,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async(
            "Write and read test file", context
        )

        assert "completed successfully" in result["content"]
        assert result["metadata"]["tool_iterations"] == 2

        # Verify file was actually created
        test_file = temp_dir / "test.txt"
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_current_time_workflow(
        self, registry_with_examples: ToolRegistry, context: RLMContext
    ) -> None:
        """Test current time tool integration."""
        mock_llm = MockToolCallingLLM("current_time")

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry_with_examples,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("What's the current time?", context)

        assert "retrieved successfully" in result["content"]
        assert result["metadata"]["tool_iterations"] == 1

    @pytest.mark.asyncio
    async def test_error_recovery(
        self, registry_with_examples: ToolRegistry, context: RLMContext
    ) -> None:
        """Test graceful error recovery when tool fails."""

        class ErrorRecoveryLLM:
            def __init__(self) -> None:
                self.call_count = 0

            async def __call__(
                self, inputs: list[Input], context: dict[str, Any]
            ) -> Output:
                self.call_count += 1

                if self.call_count == 1:
                    # Try invalid calculation
                    return {
                        "content": "",
                        "metadata": {},
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "arguments": {"expression": "invalid syntax"},
                            }
                        ],
                    }
                else:
                    # Recover from error
                    return {
                        "content": "The calculation had an error, but I handled it gracefully",
                        "metadata": {},
                    }

        mock_llm = ErrorRecoveryLLM()

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry_with_examples,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async(
            "Calculate invalid syntax", context
        )

        # Should complete without crashing
        assert "handled it gracefully" in result["content"]

    @pytest.mark.asyncio
    async def test_sequential_calculations(
        self, registry_with_examples: ToolRegistry, context: RLMContext
    ) -> None:
        """Test multiple sequential calculations."""

        class SequentialCalcLLM:
            def __init__(self) -> None:
                self.call_count = 0

            async def __call__(
                self, inputs: list[Input], context: dict[str, Any]
            ) -> Output:
                self.call_count += 1

                if self.call_count == 1:
                    # First calculation
                    return {
                        "content": "",
                        "metadata": {},
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "arguments": {"expression": "10 + 5"},
                            }
                        ],
                    }
                elif self.call_count == 2:
                    # Second calculation using first result
                    return {
                        "content": "",
                        "metadata": {},
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "arguments": {"expression": "15 * 2"},
                            }
                        ],
                    }
                else:
                    # Final answer
                    return {
                        "content": "The final result is 30",
                        "metadata": {},
                    }

        mock_llm = SequentialCalcLLM()

        engine = ToolCallingEngine(
            llm=mock_llm,
            tool_registry=registry_with_examples,
            max_depth=1,
            verbose=False,
        )

        result = await engine._execute_leaf_async("Multi-step calculation", context)

        assert "30" in result["content"]
        assert result["metadata"]["tool_iterations"] == 2


@pytest.mark.integration
class TestRealLLMIntegration:
    """Integration tests with real LLM APIs.

    These tests require:
    1. LLM API credentials (e.g., OPENAI_API_KEY)
    2. pytest flag: --run-integration

    Run with: pytest tests/integration/test_tool_calling.py --run-integration

    Note: These tests are currently placeholders and will be skipped.
    """

    @pytest.mark.asyncio
    async def test_real_openai_tool_calling(self) -> None:
        """Test with real OpenAI function calling API.

        This test is a placeholder for integration with real OpenAI API.
        Requires OPENAI_API_KEY environment variable.
        """
        pytest.skip(
            "Real OpenAI integration test requires API setup. "
            "Implement when OpenAI client is integrated."
        )
