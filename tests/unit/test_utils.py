from __future__ import annotations

import pytest

from rlm.exceptions import InvalidJSONError
from rlm.utils import safe_parse_json, validate_planner_decision


def test_parse_valid_json() -> None:
    """Test parsing of valid JSON."""
    content = '{"decision": "EXECUTE", "thoughts": "Simple task"}'
    result = safe_parse_json(content)

    assert result["decision"] == "EXECUTE"
    assert result["thoughts"] == "Simple task"


def test_parse_json_with_markdown() -> None:
    """Test parsing JSON wrapped in markdown code blocks."""
    content = """```json
    {"decision": "RECURSE", "sub_tasks": ["task1", "task2"]}
    ```"""

    result = safe_parse_json(content)
    assert result["decision"] == "RECURSE"
    assert len(result["sub_tasks"]) == 2


def test_parse_json_with_whitespace() -> None:
    """Test parsing JSON with surrounding whitespace."""
    content = """

    {"decision": "EXECUTE"}

    """

    result = safe_parse_json(content)
    assert result["decision"] == "EXECUTE"


def test_parse_invalid_json() -> None:
    """Test error handling for malformed JSON."""
    content = "This is not JSON {{"

    with pytest.raises(InvalidJSONError) as exc_info:
        safe_parse_json(content)

    assert "Failed to parse JSON" in str(exc_info.value)


def test_parse_non_dict_json() -> None:
    """Test error handling for non-dict JSON."""
    content = '["array", "not", "dict"]'

    with pytest.raises(InvalidJSONError) as exc_info:
        safe_parse_json(content)

    assert "Expected dict" in str(exc_info.value)


def test_parse_number_json() -> None:
    """Test error handling for number JSON."""
    content = "123"

    with pytest.raises(InvalidJSONError) as exc_info:
        safe_parse_json(content)

    assert "Expected dict" in str(exc_info.value)


def test_validate_planner_decision_execute() -> None:
    """Test valid EXECUTE decision."""
    data = {"decision": "EXECUTE", "thoughts": "Simple task"}
    # Should not raise
    validate_planner_decision(data)


def test_validate_planner_decision_recurse() -> None:
    """Test valid RECURSE decision."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Complex task",
        "sub_tasks": ["task1", "task2"],
    }
    # Should not raise
    validate_planner_decision(data)


def test_validate_planner_decision_missing_decision() -> None:
    """Test error for missing decision field."""
    data = {"thoughts": "Some thoughts"}

    with pytest.raises(InvalidJSONError) as exc_info:
        validate_planner_decision(data)

    assert "Missing required field 'decision'" in str(exc_info.value)


def test_validate_planner_decision_invalid_decision() -> None:
    """Test error for invalid decision value."""
    data = {"decision": "INVALID", "thoughts": "Some thoughts"}

    with pytest.raises(InvalidJSONError) as exc_info:
        validate_planner_decision(data)

    assert "Invalid decision" in str(exc_info.value)


def test_validate_planner_decision_missing_subtasks() -> None:
    """Test error for RECURSE without sub_tasks."""
    data = {"decision": "RECURSE", "thoughts": "Complex task"}

    with pytest.raises(InvalidJSONError) as exc_info:
        validate_planner_decision(data)

    assert "Missing required field 'sub_tasks'" in str(exc_info.value)


def test_validate_planner_decision_empty_subtasks() -> None:
    """Test error for empty sub_tasks list."""
    from typing import Any

    data: dict[str, Any] = {
        "decision": "RECURSE",
        "thoughts": "Complex task",
        "sub_tasks": [],
    }

    with pytest.raises(InvalidJSONError) as exc_info:
        validate_planner_decision(data)

    assert "cannot be empty" in str(exc_info.value)


def test_validate_planner_decision_invalid_subtasks_type() -> None:
    """Test error for non-list sub_tasks."""
    data = {
        "decision": "RECURSE",
        "thoughts": "Complex task",
        "sub_tasks": "not a list",
    }

    with pytest.raises(InvalidJSONError) as exc_info:
        validate_planner_decision(data)

    assert "must be a list" in str(exc_info.value)
