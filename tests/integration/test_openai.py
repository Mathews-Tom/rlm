from __future__ import annotations

import logging
import os
from typing import Any

import pytest

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Skip all tests if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable",
    ),
]


@pytest.fixture
def openai_engine() -> Any:
    """Create RecursiveEngine with OpenAI backend."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("OpenAI package not installed")

    from rlm import RecursiveEngine
    from rlm.types import Input, Output

    client = OpenAI()

    def openai_llm(inputs: list[Input], context: dict[str, Any]) -> Output:
        # Convert Input TypedDict to OpenAI message format
        from typing import cast

        messages: list[dict[str, Any]] = [
            {"role": inp["role"], "content": inp["content"]} for inp in inputs
        ]

        # Use JSON mode for planner
        if context.get("mode") == "planner":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=cast(Any, messages),  # Type cast for OpenAI SDK compatibility
                response_format={"type": "json_object"},
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=cast(Any, messages)
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


def test_simple_execution(openai_engine: Any) -> None:
    """Test simple task execution with real OpenAI API."""
    logger.info("=" * 60)
    logger.info("TEST: Simple Execution (EXECUTE decision)")
    logger.info("=" * 60)

    result = openai_engine.solve(
        "What is 2 + 2? Respond in one sentence."
    )

    logger.info(f"Result: {result['content']}")
    logger.info(f"Metadata: {result.get('metadata', {})}")

    assert "content" in result
    assert len(result["content"]) > 0
    assert "4" in result["content"]


def test_recursive_task(openai_engine: Any) -> None:
    """Test recursive task decomposition with real API."""
    logger.info("=" * 60)
    logger.info("TEST: Recursive Task (RECURSE with sub-tasks)")
    logger.info("=" * 60)

    result = openai_engine.solve(
        "In 3 sentences: What are the benefits of recursion in programming?"
    )

    logger.info(f"Result length: {len(result['content'])} chars")
    logger.info(f"Result: {result['content'][:200]}...")
    logger.info(f"Metadata: {result.get('metadata', {})}")

    assert "content" in result
    assert len(result["content"]) > 100
    # Check for recursion-related terms
    content_lower = result["content"].lower()
    assert "recurs" in content_lower or "decompos" in content_lower


def test_metadata_tracking(openai_engine: Any) -> None:
    """Test that metadata is tracked correctly."""
    logger.info("=" * 60)
    logger.info("TEST: Metadata Tracking")
    logger.info("=" * 60)

    result = openai_engine.solve("Explain Python in one sentence.")

    logger.info(f"Result: {result['content']}")
    logger.info(f"Metadata keys: {list(result.get('metadata', {}).keys())}")
    logger.info(f"Full metadata: {result.get('metadata', {})}")

    assert "metadata" in result
    assert "depth" in result["metadata"]
    assert "task_id" in result["metadata"]


def test_depth_limit_enforcement(openai_engine: Any) -> None:
    """Test that depth limit is enforced with real API.

    Creates a task that would naturally recurse deeply,
    then verifies the engine stops at max_depth.
    """
    from rlm.exceptions import RecursionDepthError

    logger.info("=" * 60)
    logger.info("TEST: Depth Limit Enforcement")
    logger.info("=" * 60)

    # Task that encourages deep recursion
    task = (
        "Break this down into subtasks recursively: "
        "Plan a complete software project with architecture, "
        "implementation, testing, and deployment phases. "
        "Break each phase into detailed sub-phases."
    )

    logger.info(f"Task: {task}")
    logger.info("Expecting RecursionDepthError...")

    # Engine has max_depth=3, so depth 3+ should fail
    with pytest.raises(RecursionDepthError) as exc_info:
        openai_engine.solve(task)

    logger.info(f"Caught expected error: {exc_info.value}")
    assert "max_depth=3" in str(exc_info.value)


def test_large_document_offloading(openai_engine: Any) -> None:
    """Test variable offloading with large documents using real API.

    Verifies that the engine can handle tasks involving large content
    by offloading to SharedMemory.
    """
    logger.info("=" * 60)
    logger.info("TEST: Large Document Offloading")
    logger.info("=" * 60)

    # Create a task with large input (10k+ characters)
    large_document = "Technical documentation: " + "x" * 10_000

    # Use directive prefix to prevent over-decomposition
    task = f"Answer directly in one sentence: What is this document about? {large_document}"

    logger.info(f"Task size: {len(task)} characters")
    logger.info("Processing large document...")

    result = openai_engine.solve(task)

    logger.info(f"Result length: {len(result['content'])} chars")
    logger.info(f"Result: {result['content']}")
    logger.info(f"Compression ratio: {len(task)}:{len(result['content'])}")

    assert "content" in result
    assert len(result["content"]) > 0
    # Summary should be much shorter than input
    assert len(result["content"]) < 1000
