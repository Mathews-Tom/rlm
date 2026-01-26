from __future__ import annotations

import pytest

from rlm.engine import RecursiveEngine
from rlm.exceptions import (
    ExecutionError,
    InvalidJSONError,
    MaxStepsError,
    RecursionDepthError,
)
from rlm.types import Input, Output


def test_execute_decision() -> None:
    """Test EXECUTE decision (leaf node)."""

    def mock_llm(inputs: list[Input], context: dict) -> Output:
        if context.get("mode") == "planner":
            return {
                "content": '{"decision": "EXECUTE", "thoughts": "Simple task"}',
                "metadata": {},
            }
        else:
            return {"content": "Final result", "metadata": {}}

    engine = RecursiveEngine(llm=mock_llm, max_depth=3)
    result = engine.solve("Simple task")

    assert result["content"] == "Final result"
    assert "metadata" in result


def test_recurse_decision() -> None:
    """Test RECURSE decision with sub-tasks."""
    call_count = {"planner": 0, "worker": 0}

    def mock_llm(inputs: list[Input], context: dict) -> Output:
        if context.get("mode") == "planner":
            call_count["planner"] += 1
            if call_count["planner"] == 1:
                # First call: RECURSE with 2 sub-tasks
                return {
                    "content": '{"decision": "RECURSE", "thoughts": "Complex", "sub_tasks": [{"description": "task1"}, {"description": "task2"}]}',
                    "metadata": {},
                }
            else:
                # Sub-tasks: EXECUTE
                return {
                    "content": '{"decision": "EXECUTE", "thoughts": "Simple"}',
                    "metadata": {},
                }
        else:
            call_count["worker"] += 1
            return {"content": f"Result {call_count['worker']}", "metadata": {}}

    engine = RecursiveEngine(llm=mock_llm, max_depth=3)
    result = engine.solve("Complex task")

    # Should have called planner 3 times (root + 2 sub-tasks)
    assert call_count["planner"] == 3
    # Should have called worker 2 times (2 leaf nodes) + 1 synthesizer
    assert call_count["worker"] >= 2
    assert "content" in result


def test_max_depth_enforcement() -> None:
    """Test that engine enforces max_depth limit."""

    def always_recurse(inputs: list[Input], context: dict) -> Output:
        return {
            "content": '{"decision": "RECURSE", "thoughts": "Always recurse", "sub_tasks": [{"description": "infinite"}]}',
            "metadata": {},
        }

    engine = RecursiveEngine(llm=always_recurse, max_depth=2)

    with pytest.raises(RecursionDepthError) as exc_info:
        engine.solve("Infinite task")

    assert "Exceeded max_depth=2" in str(exc_info.value)


def test_max_steps_enforcement() -> None:
    """Test that engine enforces max_steps limit."""
    call_count = {"count": 0}

    def wide_recurse(inputs: list[Input], context: dict) -> Output:
        if context.get("mode") == "planner":
            call_count["count"] += 1
            # Create sub-tasks only if under step limit
            if call_count["count"] <= 2:
                return {
                    "content": '{"decision": "RECURSE", "thoughts": "Wide", "sub_tasks": [{"description": "t1"},{"description": "t2"},{"description": "t3"},{"description": "t4"},{"description": "t5"}]}',
                    "metadata": {},
                }
            else:
                return {
                    "content": '{"decision": "EXECUTE", "thoughts": "Stop"}',
                    "metadata": {},
                }
        return {"content": "result", "metadata": {}}

    engine = RecursiveEngine(llm=wide_recurse, max_depth=5, max_steps=3)

    with pytest.raises(MaxStepsError) as exc_info:
        engine.solve("Wide task")

    assert "Exceeded max_steps=3" in str(exc_info.value)


def test_invalid_json_handling() -> None:
    """Test error handling for malformed JSON.

    After 3 retry attempts with malformed JSON, should raise ExecutionError
    (wrapping the underlying InvalidJSONError).
    """

    def bad_json_llm(inputs: list[Input], context: dict) -> Output:
        return {"content": "Not JSON {{", "metadata": {}}

    engine = RecursiveEngine(llm=bad_json_llm)

    with pytest.raises(ExecutionError) as exc_info:
        engine.solve("Test task")

    # Verify it's wrapping InvalidJSONError
    assert "Failed to get valid JSON after 3 attempts" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, InvalidJSONError)


def test_missing_decision_field() -> None:
    """Test error handling for missing decision field."""

    def missing_decision_llm(inputs: list[Input], context: dict) -> Output:
        return {"content": '{"thoughts": "No decision"}', "metadata": {}}

    engine = RecursiveEngine(llm=missing_decision_llm)

    with pytest.raises(InvalidJSONError) as exc_info:
        engine.solve("Test task")

    assert "Missing required field 'decision'" in str(exc_info.value)


def test_llm_exception_handling() -> None:
    """Test error handling when LLM raises exception."""

    def failing_llm(inputs: list[Input], context: dict) -> Output:
        raise RuntimeError("LLM API error")

    engine = RecursiveEngine(llm=failing_llm)

    with pytest.raises(ExecutionError) as exc_info:
        engine.solve("Test task")

    assert "LLM call failed" in str(exc_info.value)


def test_context_depth_tracking() -> None:
    """Test that context depth is tracked correctly."""
    call_count = {"planner": 0}

    def track_depth_llm(inputs: list[Input], context: dict) -> Output:
        if context.get("mode") == "planner":
            call_count["planner"] += 1
            if call_count["planner"] == 1:
                return {
                    "content": '{"decision": "RECURSE", "thoughts": "Recurse", "sub_tasks": [{"description": "task1"}]}',
                    "metadata": {},
                }
            else:
                return {
                    "content": '{"decision": "EXECUTE", "thoughts": "Execute"}',
                    "metadata": {},
                }
        else:
            return {"content": "result", "metadata": {}}

    engine = RecursiveEngine(llm=track_depth_llm, max_depth=3)
    result = engine.solve("Test task")

    # Should have metadata with depth
    assert "metadata" in result
    assert "depth" in result["metadata"]
    assert "task_id" in result["metadata"]
    assert "breadcrumbs" in result["metadata"]


def test_verbose_mode() -> None:
    """Test that verbose mode doesn't crash."""

    def simple_llm(inputs: list[Input], context: dict) -> Output:
        if context.get("mode") == "planner":
            return {
                "content": '{"decision": "EXECUTE", "thoughts": "Simple"}',
                "metadata": {},
            }
        return {"content": "result", "metadata": {}}

    engine = RecursiveEngine(llm=simple_llm, verbose=True)
    result = engine.solve("Test task")

    assert result["content"] == "result"


def test_synthesis_with_multiple_results() -> None:
    """Test synthesis combines results from multiple sub-tasks."""
    call_count = {"planner": 0, "worker": 0, "synthesizer": 0}

    def mock_llm(inputs: list[Input], context: dict) -> Output:
        mode = context.get("mode")

        if mode == "planner":
            call_count["planner"] += 1
            if call_count["planner"] == 1:
                return {
                    "content": '{"decision": "RECURSE", "thoughts": "Complex", "sub_tasks": [{"description": "task1"}, {"description": "task2"}, {"description": "task3"}]}',
                    "metadata": {},
                }
            else:
                return {
                    "content": '{"decision": "EXECUTE", "thoughts": "Simple"}',
                    "metadata": {},
                }
        elif mode == "worker":
            call_count["worker"] += 1
            return {
                "content": f"Worker result {call_count['worker']}",
                "metadata": {},
            }
        elif mode == "synthesizer":
            call_count["synthesizer"] += 1
            return {
                "content": "Synthesized final result",
                "metadata": {},
            }
        else:
            return {"content": "Unknown mode", "metadata": {}}

    engine = RecursiveEngine(llm=mock_llm, max_depth=3)
    result = engine.solve("Complex task")

    # Should have synthesized results
    assert call_count["synthesizer"] >= 1
    assert "content" in result
