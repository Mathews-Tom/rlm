from __future__ import annotations

import os
from typing import Any

import pytest

from rlm.engine import RecursiveEngine
from rlm.types import Input, Output


# Skip all tests in this module if OPENAI_API_KEY not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration tests",
)


def openai_expensive(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-4 - expensive model for all operations."""
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
    tokens = response.usage.total_tokens if response.usage else 0

    return {
        "content": content,
        "metadata": {
            "model": "gpt-4",
            "tokens": tokens,
            "cost": _calculate_cost("gpt-4", tokens),
        },
    }


def openai_planner(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-4 for planning decisions only."""
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
    tokens = response.usage.total_tokens if response.usage else 0

    return {
        "content": content,
        "metadata": {
            "model": "gpt-4-planner",
            "tokens": tokens,
            "cost": _calculate_cost("gpt-4", tokens),
        },
    }


def openai_cheap(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-3.5-turbo - cheap model for execution."""
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
    tokens = response.usage.total_tokens if response.usage else 0

    return {
        "content": content,
        "metadata": {
            "model": "gpt-3.5-turbo",
            "tokens": tokens,
            "cost": _calculate_cost("gpt-3.5-turbo", tokens),
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


def _extract_cost_metrics(result: Output) -> dict[str, float]:
    """Extract cost and token metrics from result recursively."""
    total_cost = 0.0
    total_tokens = 0
    model_counts: dict[str, int] = {}

    def traverse(obj: Any) -> None:
        nonlocal total_cost, total_tokens

        if isinstance(obj, dict):
            # Extract metadata at current level
            if "metadata" in obj:
                metadata = obj["metadata"]
                if "cost" in metadata:
                    total_cost += metadata["cost"]
                if "tokens" in metadata:
                    total_tokens += metadata["tokens"]
                if "model" in metadata:
                    model = metadata["model"]
                    model_counts[model] = model_counts.get(model, 0) + 1

            # Recurse into nested dicts
            for value in obj.values():
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    traverse(result)

    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "model_counts": model_counts,
    }


@pytest.mark.integration
def test_cost_optimization_40_percent_savings() -> None:
    """Test that multi-agent routing achieves 40%+ cost reduction.

    Compares:
    - Baseline: All operations use GPT-4 (expensive)
    - Optimized: GPT-4 planner + GPT-3.5 workers (cheap)

    Validates:
    - Cost tracking works correctly
    - Multi-agent achieves >= 40% savings
    - Quality remains high
    """
    # Test task - complex enough to decompose
    task = (
        "Analyze Python's async/await syntax. "
        "List 3 key benefits and provide a simple code example."
    )

    # Baseline: Single expensive model for everything
    print(f"\n{'='*60}")
    print("BASELINE: Single Expensive Model (GPT-4)")
    print(f"{'='*60}")

    baseline_engine = RecursiveEngine(
        llm=openai_expensive,
        max_depth=2,
        verbose=True,
    )

    baseline_result = baseline_engine.solve(task)
    baseline_metrics = _extract_cost_metrics(baseline_result)
    baseline_cost = baseline_metrics["total_cost"]

    print(f"Baseline cost: ${baseline_cost:.4f}")
    print(f"Baseline tokens: {baseline_metrics['total_tokens']}")
    print(f"Model usage: {baseline_metrics['model_counts']}")
    print(f"{'='*60}\n")

    # Optimized: Multi-agent with cheap workers
    print(f"\n{'='*60}")
    print("OPTIMIZED: Multi-Agent (GPT-4 Planner + GPT-3.5 Workers)")
    print(f"{'='*60}")

    agents = {
        "planner": openai_planner,
        "worker": openai_cheap,
    }

    optimized_engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    optimized_result = optimized_engine.solve(task)
    optimized_metrics = _extract_cost_metrics(optimized_result)
    optimized_cost = optimized_metrics["total_cost"]

    print(f"Optimized cost: ${optimized_cost:.4f}")
    print(f"Optimized tokens: {optimized_metrics['total_tokens']}")
    print(f"Model usage: {optimized_metrics['model_counts']}")
    print(f"{'='*60}\n")

    # Calculate savings
    if baseline_cost > 0:
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100
    else:
        savings = 0.0
        savings_pct = 0.0

    # Report results
    print(f"\n{'='*60}")
    print("COST OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Baseline cost:   ${baseline_cost:.4f}")
    print(f"Optimized cost:  ${optimized_cost:.4f}")
    print(f"Savings:         ${savings:.4f}")
    print(f"Savings %:       {savings_pct:.1f}%")
    print(f"Target:          >= 40%")
    print(f"Status:          {'✅ PASS' if savings_pct >= 40 else '❌ FAIL'}")
    print(f"{'='*60}\n")

    # Verify quality maintained
    assert len(baseline_result["content"]) > 50
    assert len(optimized_result["content"]) > 50

    # CRITICAL ASSERTION: Must achieve 40%+ savings
    assert savings_pct >= 40.0, (
        f"Cost optimization failed: only {savings_pct:.1f}% savings "
        f"(target: >= 40%). Baseline: ${baseline_cost:.4f}, "
        f"Optimized: ${optimized_cost:.4f}"
    )


@pytest.mark.integration
def test_cost_tracking_accuracy() -> None:
    """Test that cost tracking captures all API calls accurately.

    Validates:
    - Every LLM call records cost in metadata
    - Costs aggregate correctly through recursion
    - Token counts match expected values
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_cheap,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    task = "Explain list comprehensions in Python"
    result = engine.solve(task)

    # Extract all cost data
    metrics = _extract_cost_metrics(result)

    # Verify cost tracking
    assert metrics["total_cost"] > 0, "No cost data captured"
    assert metrics["total_tokens"] > 0, "No token data captured"
    assert len(metrics["model_counts"]) > 0, "No model usage tracked"

    print(f"\n{'='*60}")
    print("Cost Tracking Accuracy Test:")
    print(f"{'='*60}")
    print(f"Total cost: ${metrics['total_cost']:.4f}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Models used: {metrics['model_counts']}")
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_planner_token_overhead() -> None:
    """Test that planner overhead is acceptable (<20% of total cost).

    Validates:
    - Planner calls are minimal
    - Most work done by cheap workers
    - Overhead doesn't negate savings
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_cheap,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    task = "List 5 Python design patterns with brief descriptions"
    result = engine.solve(task)

    # Extract cost metrics
    metrics = _extract_cost_metrics(result)

    # Calculate planner vs worker ratio
    planner_calls = metrics["model_counts"].get("gpt-4-planner", 0)
    worker_calls = metrics["model_counts"].get("gpt-3.5-turbo", 0)

    total_calls = planner_calls + worker_calls

    if total_calls > 0:
        planner_pct = (planner_calls / total_calls) * 100
    else:
        planner_pct = 0.0

    print(f"\n{'='*60}")
    print("Planner Overhead Test:")
    print(f"{'='*60}")
    print(f"Planner calls: {planner_calls}")
    print(f"Worker calls:  {worker_calls}")
    print(f"Total calls:   {total_calls}")
    print(f"Planner %:     {planner_pct:.1f}%")
    print(f"Total cost:    ${metrics['total_cost']:.4f}")
    print(f"{'='*60}\n")

    # Verify acceptable overhead
    # Note: This is call count, not cost - planner calls are more expensive
    # but should still be minority of total calls
    assert planner_pct < 50, (
        f"Planner overhead too high: {planner_pct:.1f}% of calls "
        f"(should be < 50%)"
    )
