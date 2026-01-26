from __future__ import annotations

import pytest

from rlm.engine import RecursiveEngine
from rlm.types import Input, Output


def mock_llm(inputs: list[Input], context: dict[str, str]) -> Output:  # noqa: ARG001
    """Mock LLM that returns predefined responses."""
    mode = context.get("mode", "worker")

    if mode == "planner":
        return {
            "content": '{"decision": "EXECUTE", "thoughts": "Simple task"}',
            "metadata": {},
        }
    else:
        return {"content": "Result from default LLM", "metadata": {}}


def mock_researcher(inputs: list[Input], context: dict[str, str]) -> Output:  # noqa: ARG001
    """Mock researcher agent."""
    return {"content": "Research findings", "metadata": {"agent": "researcher"}}


def mock_writer(inputs: list[Input], context: dict[str, str]) -> Output:  # noqa: ARG001
    """Mock writer agent."""
    return {"content": "Written content", "metadata": {"agent": "writer"}}


def test_engine_without_agents() -> None:
    """Test engine in single-agent mode (no agent registry)."""
    engine = RecursiveEngine(llm=mock_llm, max_depth=2, max_steps=10)

    assert engine.agents == {}
    assert engine.router_model == "planner"


def test_engine_with_agent_registry() -> None:
    """Test engine with agent registry."""
    agents = {
        "planner": mock_llm,
        "researcher": mock_researcher,
        "writer": mock_writer,
    }

    engine = RecursiveEngine(
        llm=mock_llm, agents=agents, router_model="planner", max_depth=2
    )

    assert len(engine.agents) == 3
    assert "planner" in engine.agents
    assert "researcher" in engine.agents
    assert "writer" in engine.agents


def test_engine_get_agent_by_name() -> None:
    """Test _get_agent retrieves correct agent from registry."""
    agents = {
        "planner": mock_llm,
        "researcher": mock_researcher,
    }

    engine = RecursiveEngine(llm=mock_llm, agents=agents, router_model="planner")

    # Get existing agent
    agent = engine._get_agent("researcher")  # type: ignore[reportPrivateUsage]
    assert agent == mock_researcher

    # Get planner agent
    planner = engine._get_agent("planner")  # type: ignore[reportPrivateUsage]
    assert planner == mock_llm


def test_engine_get_agent_fallback() -> None:
    """Test _get_agent falls back to default LLM for unknown agent."""
    agents = {"planner": mock_llm}

    engine = RecursiveEngine(llm=mock_llm, agents=agents, router_model="planner")

    # Request non-existent agent
    agent = engine._get_agent("nonexistent")  # type: ignore[reportPrivateUsage]
    assert agent == mock_llm  # Falls back to default


def test_engine_get_agent_with_none() -> None:
    """Test _get_agent with None returns default LLM."""
    agents = {"planner": mock_llm}

    engine = RecursiveEngine(llm=mock_llm, agents=agents, router_model="planner")

    agent = engine._get_agent(None)  # type: ignore[reportPrivateUsage]
    assert agent == mock_llm


def test_engine_get_agent_single_agent_mode() -> None:
    """Test _get_agent in single-agent mode (no registry)."""
    engine = RecursiveEngine(llm=mock_llm, max_depth=2)

    # All requests return default LLM
    agent1 = engine._get_agent("researcher")  # type: ignore[reportPrivateUsage]
    agent2 = engine._get_agent(None)  # type: ignore[reportPrivateUsage]
    agent3 = engine._get_agent("writer")  # type: ignore[reportPrivateUsage]

    assert agent1 == mock_llm
    assert agent2 == mock_llm
    assert agent3 == mock_llm


def test_engine_router_model_validation() -> None:
    """Test that invalid router_model raises error."""
    agents = {"researcher": mock_researcher, "writer": mock_writer}

    with pytest.raises(ValueError) as exc_info:
        RecursiveEngine(
            llm=mock_llm,
            agents=agents,
            router_model="nonexistent",  # Not in registry
        )

    assert "router_model 'nonexistent' not found" in str(exc_info.value)
    assert "Available agents" in str(exc_info.value)


def test_engine_router_model_default() -> None:
    """Test default router_model value."""
    engine = RecursiveEngine(llm=mock_llm)

    assert engine.router_model == "planner"


def test_engine_router_model_custom() -> None:
    """Test custom router_model."""
    agents = {"custom_planner": mock_llm}

    engine = RecursiveEngine(
        llm=mock_llm, agents=agents, router_model="custom_planner"
    )

    assert engine.router_model == "custom_planner"


def test_engine_agent_registry_immutable() -> None:
    """Test that agent registry cannot be modified after initialization."""
    agents = {"planner": mock_llm}

    engine = RecursiveEngine(llm=mock_llm, agents=agents, router_model="planner")

    # Modifying the original dict should not affect engine
    agents["hacker"] = mock_researcher

    assert "hacker" not in engine.agents
    assert len(engine.agents) == 1
