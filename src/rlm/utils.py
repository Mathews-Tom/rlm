from __future__ import annotations

import json
import re
from typing import Any

from rlm.types import InvalidJSONError


def safe_parse_json(content: str) -> dict[str, Any]:
    """Parse JSON from LLM output safely.

    Strips markdown code blocks and validates JSON structure.
    Uses json.loads exclusively (never eval).

    Args:
        content: Raw string from LLM (may contain markdown)

    Returns:
        Parsed JSON as dict

    Raises:
        InvalidJSONError: If JSON is malformed or invalid

    Example:
        >>> safe_parse_json('{"decision": "EXECUTE"}')
        {'decision': 'EXECUTE'}
        >>> safe_parse_json('```json\\n{"decision": "EXECUTE"}\\n```')
        {'decision': 'EXECUTE'}
    """
    # Strip markdown code blocks (handle various formats)
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    try:
        # Use json.loads (safe), never eval() (dangerous)
        data = json.loads(content)

        if not isinstance(data, dict):
            raise InvalidJSONError(f"Expected dict, got {type(data).__name__}")

        return data
    except json.JSONDecodeError as e:
        raise InvalidJSONError(f"Failed to parse JSON: {e}") from e


def validate_planner_decision(data: dict[str, Any]) -> None:
    """Validate PlannerDecision schema.

    Args:
        data: Parsed JSON dict from LLM

    Raises:
        InvalidJSONError: If required fields missing or invalid
    """
    if "decision" not in data:
        raise InvalidJSONError("Missing required field 'decision'")

    if data["decision"] not in ("EXECUTE", "RECURSE"):
        raise InvalidJSONError(
            f"Invalid decision: {data['decision']!r}, expected 'EXECUTE' or 'RECURSE'"
        )

    if data["decision"] == "RECURSE" and "sub_tasks" not in data:
        raise InvalidJSONError(
            "Missing required field 'sub_tasks' for RECURSE decision"
        )

    if data["decision"] == "RECURSE":
        if not isinstance(data["sub_tasks"], list):
            raise InvalidJSONError("Field 'sub_tasks' must be a list")

        if not data["sub_tasks"]:
            raise InvalidJSONError("Field 'sub_tasks' cannot be empty for RECURSE")
