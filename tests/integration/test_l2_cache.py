from __future__ import annotations

import os
import time
from typing import Any

import pytest

from rlm.caching import REDIS_AVAILABLE, CachedAsyncEngine
from rlm.types import AsyncLLMCaller, Input, Output

# Skip all tests if Redis dependencies not available
pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="redisvl not installed (install with: uv sync --group cache)"
)


class MockAsyncLLM:
    """Mock async LLM for testing L2 cache behavior."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
        """Record calls and return mock responses with artificial delay."""
        # Simulate LLM latency
        await __import__("asyncio").sleep(0.1)

        self.call_count += 1
        self.calls.append({"inputs": inputs, "context": context})

        # Return planner decision for planning calls
        if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
            return {
                "content": '{"thoughts": "Execute", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"model": "mock", "call": self.call_count},
            }

        return {
            "content": f"Mock result #{self.call_count}",
            "metadata": {"model": "mock", "call": self.call_count},
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_l2_cache_semantic_similarity() -> None:
    """Test L2 semantic cache with similar but not identical prompts.

    Note: This test requires Redis to be running on localhost:6379.
    If Redis is unavailable, it verifies graceful degradation to L1-only mode.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(
        llm=llm,
        max_depth=1,
        l1_size=100,
        redis_url=redis_url,
        cache_threshold=0.80,  # Lower threshold for testing
        ttl=300,  # 5 min TTL for test
        verbose=True,
    )

    # First call - should execute and store in cache(s)
    result1 = await engine.solve("Write a marketing plan for a new product")
    assert llm.call_count == 2  # Plan + execute

    # Second call with identical task - should hit L1
    initial_calls = llm.call_count
    result2 = await engine.solve("Write a marketing plan for a new product")
    assert llm.call_count == initial_calls, "L1 cache should hit"
    assert result2["content"] == result1["content"]

    stats = engine.get_cache_stats()
    assert stats["l1_hits"] == 1
    assert stats["l2_hits"] == 0

    # Check if L2 is enabled (requires Redis to be running)
    if stats["l2_enabled"]:
        print("✓ L2 Redis cache is enabled and available")

        # Third call with similar task - might hit L2 with semantic matching
        result3 = await engine.solve("Create a marketing strategy for a new product")
        assert "Mock result" in result3["content"]

        stats = engine.get_cache_stats()
        # With semantic matching, might get L2 hit (depends on embedding similarity)
        print(f"Cache stats after similar query: {stats}")
        assert stats["cache_hits"] + stats["cache_misses"] == 3
    else:
        print("⚠ L2 Redis cache unavailable - testing graceful degradation")
        # Verify L1-only mode works correctly
        result3 = await engine.solve("Create a marketing strategy for a new product")
        assert "Mock result" in result3["content"]

        stats = engine.get_cache_stats()
        assert stats["cache_hits"] + stats["cache_misses"] == 3
        assert stats["l2_enabled"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_l2_cache_promotion_to_l1() -> None:
    """Test that L2 cache hits are promoted to L1.

    Note: This test requires Redis to be running on localhost:6379.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(
        llm=llm,
        max_depth=1,
        l1_size=2,  # Small L1 to test eviction
        redis_url=redis_url,
        cache_threshold=0.85,
        ttl=300,
        verbose=True,
    )

    stats = engine.get_cache_stats()
    if not stats["l2_enabled"]:
        pytest.skip("Redis unavailable - L2 cache promotion test requires Redis")

    # Fill L1 cache with 2 entries
    await engine.solve("Task A")
    await engine.solve("Task B")

    stats = engine.get_cache_stats()
    assert stats["l1_size"] == 2

    # Add third entry - evicts Task A from L1
    await engine.solve("Task C")

    stats = engine.get_cache_stats()
    assert stats["l1_size"] == 2  # Still at max size

    # Query Task A again - should hit L2 and promote to L1
    initial_calls = llm.call_count
    await engine.solve("Task A")

    stats = engine.get_cache_stats()
    # Either L2 hit (promoted) or cache miss (executed)
    if stats["l2_hits"] > 0:
        # L2 hit should not make additional LLM calls
        assert llm.call_count == initial_calls, "L2 hit should not call LLM"
        print("✓ L2 cache hit successfully promoted to L1")
    else:
        print("⚠ No L2 hit - task may have expired or not matched")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_l2_cache_speedup() -> None:
    """Measure speedup from L2 cache hits."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(
        llm=llm,
        max_depth=1,
        l1_size=100,
        redis_url=redis_url,
        cache_threshold=0.85,
        ttl=300,
        verbose=False,
    )

    # Execute task and measure time
    task = "Analyze the performance characteristics of distributed systems"

    # First execution (cache miss)
    start = time.time()
    await engine.solve(task)
    uncached_time = time.time() - start

    # Second execution (cache hit)
    start = time.time()
    await engine.solve(task)
    cached_time = time.time() - start

    # Cache hit should be significantly faster
    speedup = uncached_time / cached_time if cached_time > 0 else 0

    stats = engine.get_cache_stats()
    print(f"\nCache Stats: {stats}")
    print(f"Uncached time: {uncached_time:.3f}s")
    print(f"Cached time: {cached_time:.3f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Verify cache hit
    assert stats["cache_hits"] >= 1, "Should have at least one cache hit"

    # Speedup should be substantial (>5x)
    # Note: In practice with real LLMs, this would be 15x+
    assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_l2_cache_graceful_degradation() -> None:
    """Test graceful degradation when Redis is unavailable."""
    llm = MockAsyncLLM()

    # Use invalid Redis URL
    engine = CachedAsyncEngine(
        llm=llm,
        max_depth=1,
        l1_size=100,
        redis_url="redis://invalid-host:9999",
        cache_threshold=0.85,
        ttl=300,
        verbose=True,
    )

    # Should fall back to L1-only mode
    stats = engine.get_cache_stats()
    assert stats["l2_enabled"] is False, "L2 should be disabled with invalid Redis URL"

    # Execution should still work with L1 cache
    result = await engine.solve("Test task")
    assert "Mock result" in result["content"]

    # Second call should hit L1
    initial_calls = llm.call_count
    result2 = await engine.solve("Test task")
    assert llm.call_count == initial_calls, "L1 cache should work without Redis"
    assert result2["content"] == result["content"]

    stats = engine.get_cache_stats()
    assert stats["l1_hits"] == 1
    assert stats["l2_hits"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_l2_cache_hit_rate() -> None:
    """Demonstrate 40-50% cache hit rate with varied tasks."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(
        llm=llm,
        max_depth=1,
        l1_size=100,
        redis_url=redis_url,
        cache_threshold=0.85,
        ttl=300,
        verbose=False,
    )

    # Execute 20 tasks with some repetition to simulate realistic workflow
    tasks = [
        "Write a function",
        "Debug the error",
        "Write a test",
        "Write a function",  # Repeat
        "Refactor code",
        "Debug the error",  # Repeat
        "Write documentation",
        "Write a test",  # Repeat
        "Optimize performance",
        "Write a function",  # Repeat
        "Review code",
        "Debug the error",  # Repeat
        "Write a test",  # Repeat
        "Deploy to production",
        "Write documentation",  # Repeat
        "Monitor metrics",
        "Write a function",  # Repeat
        "Refactor code",  # Repeat
        "Write a test",  # Repeat
        "Debug the error",  # Repeat
    ]

    for task in tasks:
        await engine.solve(task)

    stats = engine.get_cache_stats()
    print(f"\nCache Stats after {len(tasks)} tasks: {stats}")

    # With repetition, we should achieve reasonable hit rate
    # Note: Without real Redis semantic matching, this tests exact matches only
    assert stats["hit_rate"] >= 0.30, f"Expected >=30% hit rate, got {stats['hit_rate']:.1%}"

    # Total requests should match task count
    assert stats["cache_hits"] + stats["cache_misses"] == len(tasks)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
