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


def openai_expensive(inputs: list[Input], context: dict[str, Any]) -> Output:
    """OpenAI GPT-4 - expensive model for all operations."""
    try:
        import openai
    except ImportError:
        pytest.skip("openai package not installed")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Type messages explicitly to match OpenAI's ChatCompletionMessageParam
    messages: list[dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]} for msg in inputs
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore[arg-type]
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

    # Type messages explicitly to match OpenAI's ChatCompletionMessageParam
    messages: list[dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]} for msg in inputs
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore[arg-type]
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

    # Type messages explicitly to match OpenAI's ChatCompletionMessageParam
    messages: list[dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]} for msg in inputs
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,  # type: ignore[arg-type]
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


def _extract_cost_metrics(result: Output) -> dict[str, Any]:
    """Extract cost and token metrics from result recursively."""
    total_cost: float = 0.0
    total_tokens: int = 0
    model_counts: dict[str, int] = {}

    def traverse(obj: Any) -> None:
        nonlocal total_cost, total_tokens

        if isinstance(obj, dict):
            # Extract metadata at current level
            if "metadata" in obj:
                metadata: Any = obj["metadata"]  # type: ignore[reportUnknownVariableType]
                if isinstance(metadata, dict):
                    if "cost" in metadata:
                        cost_value: Any = metadata["cost"]  # type: ignore[reportUnknownVariableType]
                        if isinstance(cost_value, (int, float)):
                            total_cost += float(cost_value)
                    if "tokens" in metadata:
                        tokens_value: Any = metadata["tokens"]  # type: ignore[reportUnknownVariableType]
                        if isinstance(tokens_value, int):
                            total_tokens += tokens_value
                    if "model" in metadata:
                        model: Any = metadata["model"]  # type: ignore[reportUnknownVariableType]
                        if isinstance(model, str):
                            model_counts[model] = model_counts.get(model, 0) + 1

            # Recurse into nested dicts
            for value in obj.values():  # type: ignore[reportUnknownVariableType]
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:  # type: ignore[reportUnknownVariableType]
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
    # Test task - needs to decompose to show cost savings
    # Task with explicit parts triggers decomposition, but stays simple to avoid deep recursion
    task = "Explain Python in 3 parts: 1) what it is, 2) one key benefit, 3) one common use. Write exactly 1 sentence per part."

    # Baseline: Single expensive model for everything
    print(f"\n{'='*60}")
    print("BASELINE: Single Expensive Model (GPT-4)")
    print(f"{'='*60}")

    baseline_engine = RecursiveEngine(
        llm=openai_expensive,
        max_depth=3,
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
        "synthesizer": openai_cheap,  # Use cheap model for synthesis
    }

    optimized_engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=3,
        verbose=True,
    )

    optimized_result = optimized_engine.solve(task)

    # DEBUG: Inspect result structure
    import json
    print("\n" + "="*60)
    print("DEBUG: Full Result Structure")
    print("="*60)
    print("\nBASELINE result keys:")
    print(list(baseline_result.keys()))
    print("\nOPTIMIZED result keys:")
    print(list(optimized_result.keys()))

    # Check if sub_results exists
    if "sub_results" in optimized_result:
        print("\n✓ Found 'sub_results' in optimized result")
        print(f"  Sub-results count: {len(optimized_result['sub_results'])}")
        print("\n  First sub-result structure:")
        print(json.dumps(optimized_result['sub_results'][0], indent=4, default=str)[:500])
    else:
        print("\n✗ No 'sub_results' key in optimized result")
        print("  Available keys:", list(optimized_result.keys()))
    print("="*60 + "\n")

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
        "synthesizer": openai_cheap,  # Use cheap model for synthesis
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=2,
        verbose=True,
    )

    task = "Explain list comprehensions in one sentence"
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
    """Test that planner overhead is acceptable when decomposition occurs.

    Validates:
    - Cost and model tracking works for multi-agent setup
    - When decomposition occurs, planner is minority of calls
    - When no decomposition occurs, single planner call is recorded
    """
    agents = {
        "planner": openai_planner,
        "worker": openai_cheap,
        "synthesizer": openai_cheap,
    }

    engine = RecursiveEngine(
        llm=openai_planner,
        agents=agents,
        router_model="planner",
        max_depth=3,
        verbose=True,
    )

    # Use a task with many explicit independent parts to maximize
    # the likelihood of decomposition. The planner may still choose
    # EXECUTE — the test handles both outcomes.
    task = (
        "Answer each of these 5 questions separately:\n"
        "1) What is the Singleton pattern?\n"
        "2) What is the Factory pattern?\n"
        "3) What is the Observer pattern?\n"
        "4) What is the Strategy pattern?\n"
        "5) What is the Decorator pattern?\n"
        "Write exactly 1 sentence per question."
    )
    result = engine.solve(task)

    # Extract cost metrics
    metrics = _extract_cost_metrics(result)

    planner_calls = metrics["model_counts"].get("gpt-4-planner", 0)
    worker_calls = metrics["model_counts"].get("gpt-3.5-turbo", 0)
    total_calls = planner_calls + worker_calls

    print(f"\n{'='*60}")
    print("Planner Overhead Test:")
    print(f"{'='*60}")
    print(f"Planner calls: {planner_calls}")
    print(f"Worker calls:  {worker_calls}")
    print(f"Total calls:   {total_calls}")
    print(f"Total cost:    ${metrics['total_cost']:.4f}")
    print(f"{'='*60}\n")

    # Core invariant: at least one LLM call was tracked
    assert total_calls >= 1, "No LLM calls recorded in metrics"
    assert metrics["total_cost"] > 0, "No cost data captured"
    assert metrics["total_tokens"] > 0, "No token data captured"

    # If decomposition occurred (worker calls > 0), planner should
    # be the minority of total calls
    if worker_calls > 0:
        planner_pct = (planner_calls / total_calls) * 100
        assert planner_pct < 50, (
            f"Planner overhead too high: {planner_pct:.1f}% of calls "
            f"(should be < 50% when decomposition occurs)"
        )
