from __future__ import annotations

from rlm.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT


def test_planner_prompt_has_placeholders() -> None:
    """Test that PLANNER_SYSTEM_PROMPT contains required placeholders."""
    assert "{available_agents}" in PLANNER_SYSTEM_PROMPT
    assert "{agent_descriptions}" in PLANNER_SYSTEM_PROMPT


def test_planner_prompt_formatting() -> None:
    """Test that PLANNER_SYSTEM_PROMPT can be formatted with agent data."""
    formatted = PLANNER_SYSTEM_PROMPT.format(
        available_agents="planner, researcher, writer",
        agent_descriptions="- planner: Planning agent\n- researcher: Research agent\n- writer: Writing agent",
    )

    assert "planner, researcher, writer" in formatted
    assert "Planning agent" in formatted
    assert "{available_agents}" not in formatted
    assert "{agent_descriptions}" not in formatted


def test_planner_prompt_with_no_agents() -> None:
    """Test PLANNER_SYSTEM_PROMPT formatting with no agents."""
    formatted = PLANNER_SYSTEM_PROMPT.format(
        available_agents="None (single-agent mode)",
        agent_descriptions="No specialized agents available.",
    )

    assert "None (single-agent mode)" in formatted
    assert "No specialized agents available" in formatted


def test_planner_prompt_structure() -> None:
    """Test that PLANNER_SYSTEM_PROMPT has required instructions."""
    assert "EXECUTE" in PLANNER_SYSTEM_PROMPT
    assert "RECURSE" in PLANNER_SYSTEM_PROMPT
    assert "JSON" in PLANNER_SYSTEM_PROMPT
    assert "decision" in PLANNER_SYSTEM_PROMPT
    assert "sub_tasks" in PLANNER_SYSTEM_PROMPT
    assert "assigned_agent" in PLANNER_SYSTEM_PROMPT


def test_synthesizer_prompt_exists() -> None:
    """Test that SYNTHESIZER_SYSTEM_PROMPT is defined."""
    assert isinstance(SYNTHESIZER_SYSTEM_PROMPT, str)
    assert len(SYNTHESIZER_SYSTEM_PROMPT) > 0


def test_synthesizer_prompt_structure() -> None:
    """Test that SYNTHESIZER_SYSTEM_PROMPT has synthesis guidance."""
    assert "synthesis" in SYNTHESIZER_SYSTEM_PROMPT.lower()
    assert "combine" in SYNTHESIZER_SYSTEM_PROMPT.lower()
    assert "result" in SYNTHESIZER_SYSTEM_PROMPT.lower()


def test_synthesizer_prompt_output_types() -> None:
    """Test that SYNTHESIZER_SYSTEM_PROMPT mentions different output types."""
    prompt_lower = SYNTHESIZER_SYSTEM_PROMPT.lower()
    assert "text" in prompt_lower or "narrative" in prompt_lower
    assert "list" in prompt_lower
    assert "data" in prompt_lower or "json" in prompt_lower
