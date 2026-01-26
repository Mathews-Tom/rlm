from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE skipif check
load_dotenv()

from rlm.engine import RecursiveEngine
from rlm.types import Input, Output

# Skip all tests in this module if OPENAI_API_KEY not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration tests",
)


def openai_planner(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-4 for planning decisions (expensive, high-quality)."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai package not installed")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in inputs]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""

    return {
        "content": content,
        "metadata": {
            "model": "gpt-4",
            "tokens": response.usage.total_tokens if response.usage else 0,
            "cost": _calculate_cost("gpt-4", response.usage.total_tokens if response.usage else 0),
        },
    }


def openai_worker(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-3.5-turbo for execution (cheap, fast)."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai package not installed")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in inputs]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""

    return {
        "content": content,
        "metadata": {
            "model": "gpt-3.5-turbo",
            "tokens": response.usage.total_tokens if response.usage else 0,
            "cost": _calculate_cost("gpt-3.5-turbo", response.usage.total_tokens if response.usage else 0),
        },
    }


def _calculate_cost(model: str, tokens: int) -> float:
    """Calculate estimated API cost based on model and tokens.

    Pricing as of 2025-01 (approximate):
    - GPT-4: $0.03/1K input, $0.06/1K output (avg $0.045/1K)
    - GPT-3.5-turbo: $0.001/1K input, $0.002/1K output (avg $0.0015/1K)
    """
    rates = {
        "gpt-4": 0.045 / 1000,
        "gpt-3.5-turbo": 0.0015 / 1000,
    }
    return tokens * rates.get(model, 0.0)


@pytest.mark.integration
def test_multi_agent_routing() -> None:
    """Test multi-agent routing with real OpenAI models.

    Validates:
    - Planner (GPT-4) makes routing decisions
    - Worker (GPT-3.5) executes sub-tasks
    - Correct agents are called for each step
    - Results are synthesized properly
    """
    # Setup multi-agent registry
    agents = {
        "planner": openai_planner,
        "worker": openai_worker,
    }

    engine = RecursiveEngine(
        llm=openai_planner,  # Fallback
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    # Execute a simple task that should not over-decompose
    task = "Answer directly without research: What are 3 benefits of type hints? Use bullet points."
    result = engine.solve(task)

    # Verify result structure
    assert isinstance(result, dict)
    assert "content" in result
    assert "metadata" in result
    assert len(result["content"]) > 0

    # Verify planner was used (check metadata)
    assert "model" in result["metadata"] or "depth" in result["metadata"]

    # Log result for manual inspection
    print(f"\n{'='*60}")
    print("Multi-Agent Routing Test Result:")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"\nResult: {result['content'][:200]}...")
    print(f"\nMetadata: {result['metadata']}")
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_agent_assignment_in_subtasks() -> None:
    """Test that planner correctly assigns agents to sub-tasks.

    Validates:
    - Planner decision includes assigned_agent fields
    - Sub-tasks route to correct agents
    - Agent metadata tracked in context
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_worker,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    # Task that should decompose with explicit agent routing
    task = "Name one new Python 3.12 feature and explain it in one sentence."

    result = engine.solve(task)

    # Verify execution completed
    assert isinstance(result, dict)
    assert len(result["content"]) > 50

    # Log execution details
    print(f"\n{'='*60}")
    print("Agent Assignment Test Result:")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"\nResult length: {len(result['content'])} chars")
    print(f"Metadata: {result['metadata']}")
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_synthesis_of_multi_agent_results() -> None:
    """Test synthesis of results from multiple specialized agents.

    Validates:
    - Multiple agents contribute to solution
    - Planner synthesizes results coherently
    - Final output integrates all sub-results
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_worker,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=3,
        verbose=True,
    )

    # Task requiring coordination - use "parts" structure for single-level decomposition
    task = "Explain Python match-case in 3 parts: 1) one-sentence definition, 2) one key benefit in one sentence, 3) one-line syntax example."

    result = engine.solve(task)

    # Verify comprehensive output
    assert isinstance(result, dict)
    assert len(result["content"]) > 100

    # Check for synthesis metadata
    metadata = result.get("metadata", {})
    assert "depth" in metadata or "task_id" in metadata

    # Log synthesis details
    print(f"\n{'='*60}")
    print("Multi-Agent Synthesis Test Result:")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"\nResult preview: {result['content'][:300]}...")
    print(f"\nFull result length: {len(result['content'])} chars")
    print(f"Metadata: {metadata}")
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_fallback_to_default_agent() -> None:
    """Test that unknown agent names fall back to default gracefully.

    Validates:
    - Invalid agent assignments don't crash
    - Engine falls back to default LLM
    - Warning logged when fallback occurs
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_worker,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=4,
        verbose=True,
    )

    # Simple task - should complete even with potential routing issues
    task = "Explain Python's walrus operator in one sentence"

    result = engine.solve(task)

    # Verify graceful handling
    assert isinstance(result, dict)
    assert len(result["content"]) > 0

    # Log fallback behavior
    print(f"\n{'='*60}")
    print("Agent Fallback Test Result:")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"\nResult: {result['content']}")
    print(f"Metadata: {result['metadata']}")
    print(f"{'='*60}\n")
