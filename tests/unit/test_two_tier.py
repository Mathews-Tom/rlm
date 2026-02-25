from __future__ import annotations

from typing import Any

import pytest

from rlm.engine import RecursiveEngine
from rlm.types import Input, Output


def make_llm(name: str):  # type: ignore[return]
    """Create a mock LLM that returns its name."""
    def llm(inputs: list[Input], context: dict[str, Any]) -> Output:
        return {"content": f"from_{name}", "metadata": {"model": name}}
    return llm


class TestGetModelForRole:
    def test_no_sub_model(self) -> None:
        primary = make_llm("primary")
        engine = RecursiveEngine(llm=primary)
        assert engine._get_model_for_role("plan") is primary
        assert engine._get_model_for_role("execute") is primary
        assert engine._get_model_for_role("synthesize") is primary

    def test_with_sub_model(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(llm=primary, sub_model=sub)
        assert engine._get_model_for_role("plan") is primary
        assert engine._get_model_for_role("execute") is sub
        assert engine._get_model_for_role("synthesize") is primary

    def test_sub_model_only_for_execute(self) -> None:
        """sub_model must not be used for plan or synthesize roles."""
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(llm=primary, sub_model=sub)
        for role in ("plan", "synthesize"):
            assert engine._get_model_for_role(role) is primary

    def test_unknown_role_falls_back_to_llm(self) -> None:
        """Unrecognised roles fall back to primary llm."""
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(llm=primary, sub_model=sub)
        assert engine._get_model_for_role("unknown") is primary


class TestGetAgentWithRole:
    def test_named_agent_overrides_tier(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        researcher = make_llm("researcher")
        engine = RecursiveEngine(
            llm=primary,
            sub_model=sub,
            agents={"planner": primary, "researcher": researcher},
            router_model="planner",
        )
        # Named agent wins over tier
        assert engine._get_agent("researcher", role="execute") is researcher
        # Fallback to tier
        assert engine._get_agent("unknown", role="execute") is sub
        assert engine._get_agent(None, role="execute") is sub
        assert engine._get_agent(None, role="plan") is primary

    def test_none_agent_uses_role_model(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(llm=primary, sub_model=sub)
        assert engine._get_agent(None, role="execute") is sub
        assert engine._get_agent(None, role="plan") is primary

    def test_single_agent_mode_ignores_name(self) -> None:
        """Without agents registry, named lookup always resolves via role."""
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(llm=primary, sub_model=sub)
        # No agents dict — any name falls through to role-based model
        assert engine._get_agent("researcher", role="execute") is sub
        assert engine._get_agent("researcher", role="plan") is primary

    def test_registered_agent_wins_over_sub_model(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        specialist = make_llm("specialist")
        engine = RecursiveEngine(
            llm=primary,
            sub_model=sub,
            agents={"planner": primary, "specialist": specialist},
            router_model="planner",
        )
        # Named specialist wins even though role is "execute" (which would pick sub_model)
        assert engine._get_agent("specialist", role="execute") is specialist

    def test_unknown_agent_falls_back_to_sub_model_for_execute(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(
            llm=primary,
            sub_model=sub,
            agents={"planner": primary},
            router_model="planner",
        )
        assert engine._get_agent("nonexistent", role="execute") is sub

    def test_unknown_agent_falls_back_to_primary_for_plan(self) -> None:
        primary = make_llm("primary")
        sub = make_llm("sub")
        engine = RecursiveEngine(
            llm=primary,
            sub_model=sub,
            agents={"planner": primary},
            router_model="planner",
        )
        assert engine._get_agent("nonexistent", role="plan") is primary

    def test_router_model_missing_raises(self) -> None:
        """Providing agents without a matching router_model raises ValueError."""
        primary = make_llm("primary")
        with pytest.raises(ValueError, match="router_model"):
            RecursiveEngine(
                llm=primary,
                agents={"researcher": make_llm("researcher")},
                router_model="planner",
            )
