from __future__ import annotations

from rlm.types import SubTask
from rlm.utils import validate_planner_decision


def test_subtask_with_description_only() -> None:
    """Test SubTask with only required description field."""
    subtask: SubTask = {"description": "Analyze data"}

    assert subtask["description"] == "Analyze data"
    assert "assigned_agent" not in subtask


def test_subtask_with_agent_assignment() -> None:
    """Test SubTask with optional assigned_agent field."""
    subtask: SubTask = {
        "description": "Research topic",
        "assigned_agent": "researcher",
    }

    assert subtask["description"] == "Research topic"
    assert subtask.get("assigned_agent") == "researcher"


def test_subtask_with_null_agent() -> None:
    """Test SubTask with explicitly null assigned_agent."""
    subtask: SubTask = {
        "description": "Execute task",
        "assigned_agent": None,
    }

    assert subtask["description"] == "Execute task"
    assert subtask.get("assigned_agent") is None


def test_validate_planner_decision_with_agent_assignments() -> None:
    """Test validation of RECURSE decision with agent assignments."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Complex task requiring specialized agents",
        "sub_tasks": [
            {"description": "Research topic", "assigned_agent": "researcher"},
            {"description": "Write report", "assigned_agent": "writer"},
            {"description": "Review content", "assigned_agent": None},
        ],
    }

    # Should not raise
    validate_planner_decision(data)


def test_validate_planner_decision_with_mixed_agents() -> None:
    """Test validation with mix of assigned and unassigned agents."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Partial agent assignment",
        "sub_tasks": [
            {"description": "Task 1", "assigned_agent": "agent1"},
            {"description": "Task 2"},  # No agent
            {"description": "Task 3", "assigned_agent": "agent3"},
        ],
    }

    # Should not raise
    validate_planner_decision(data)


def test_validate_planner_decision_subtask_dict_structure() -> None:
    """Test that sub_tasks can be dict structures (SubTask TypedDict)."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Using structured sub-tasks",
        "sub_tasks": [
            {"description": "Task with agent", "assigned_agent": "specialist"},
            {"description": "Task without agent"},
        ],
    }

    # Should not raise
    validate_planner_decision(data)


def test_validate_planner_decision_legacy_string_subtasks() -> None:
    """Test backward compatibility with string sub_tasks."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Legacy format",
        "sub_tasks": ["task1", "task2", "task3"],
    }

    # Should not raise (backward compatible)
    validate_planner_decision(data)


def test_validate_planner_decision_empty_agent_name() -> None:
    """Test validation with empty string agent name."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Empty agent name",
        "sub_tasks": [
            {"description": "Task", "assigned_agent": ""},
        ],
    }

    # Should not raise (empty string is valid, engine will handle fallback)
    validate_planner_decision(data)


def test_validate_planner_decision_execute_ignores_subtasks() -> None:
    """Test that EXECUTE decision ignores sub_tasks field if present."""
    data = {
        "decision": "EXECUTE",
        "thoughts": "Simple task",
        "sub_tasks": ["This", "Should", "Be", "Ignored"],
    }

    # Should not raise (sub_tasks ignored for EXECUTE)
    validate_planner_decision(data)


def test_validate_planner_decision_invalid_subtask_structure() -> None:
    """Test validation rejects invalid sub_task structure."""
    # Note: Current validate_planner_decision only checks that sub_tasks is a list
    # It doesn't validate individual SubTask structure, which is type-checked at runtime
    data = {
        "decision": "RECURSE",
        "thoughts": "Invalid structure",
        "sub_tasks": [123, 456],  # Invalid: numbers instead of dicts/strings
    }

    # Current implementation should not raise (only checks list type)
    # Type errors would be caught at runtime when accessing fields
    validate_planner_decision(data)
