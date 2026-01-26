from __future__ import annotations

import os

import pytest

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)


@pytest.fixture
def openai_engine():
    """Create RecursiveEngine with OpenAI backend."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("OpenAI package not installed")

    from rlm import RecursiveEngine
    from rlm.types import Input, Output

    client = OpenAI()

    def openai_llm(inputs: list[Input], context: dict) -> Output:
        messages = [
            {"role": inp["role"], "content": inp["content"]} for inp in inputs
        ]

        # Use JSON mode for planner
        if context.get("mode") == "planner":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            )

        return {
            "content": response.choices[0].message.content or "",
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens": (
                    response.usage.total_tokens if response.usage else 0
                ),
            },
        }

    return RecursiveEngine(llm=openai_llm, max_depth=3, verbose=True)


def test_simple_execution(openai_engine):
    """Test simple task execution with real OpenAI API."""
    result = openai_engine.solve(
        "What is 2 + 2? Respond in one sentence."
    )

    assert "content" in result
    assert len(result["content"]) > 0
    assert "4" in result["content"]


def test_recursive_task(openai_engine):
    """Test recursive task decomposition with real API."""
    result = openai_engine.solve(
        "Write a 3-sentence summary of the benefits of recursion in programming."
    )

    assert "content" in result
    assert len(result["content"]) > 100
    # Check for recursion-related terms
    content_lower = result["content"].lower()
    assert "recurs" in content_lower or "decompos" in content_lower


def test_metadata_tracking(openai_engine):
    """Test that metadata is tracked correctly."""
    result = openai_engine.solve("Explain Python in one sentence.")

    assert "metadata" in result
    assert "depth" in result["metadata"]
    assert "task_id" in result["metadata"]
