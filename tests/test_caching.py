from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rlm.caching import CachedAsyncEngine
from rlm.memory import RLMContext, SharedMemory
from rlm.types import AsyncLLMCaller, Input, Output


class MockAsyncLLM:
    """Mock async LLM for testing caching behavior."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
        """Record calls and return mock responses."""
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


@pytest.mark.asyncio
async def test_cache_key_computation() -> None:
    """Test cache key generation from task, agent, and depth."""
    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(llm=llm, max_depth=3, l1_size=100)

    # Create contexts with different parameters
    ctx1 = RLMContext(
        task_id="1", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="planner"
    )
    ctx2 = RLMContext(
        task_id="2", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="planner"
    )
    ctx3 = RLMContext(
        task_id="3", parent_id=None, depth=1,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="planner"
    )
    ctx4 = RLMContext(
        task_id="4", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="writer"
    )

    # Same task, same agent, same depth = same key
    key1a = engine._compute_cache_key("Write a plan", ctx1)
    key1b = engine._compute_cache_key("Write a plan", ctx2)
    assert key1a == key1b, "Same task/agent/depth should produce same key"

    # Same task, same agent, different depth = different key
    key2 = engine._compute_cache_key("Write a plan", ctx3)
    assert key1a != key2, "Different depth should produce different key"

    # Same task, different agent, same depth = different key
    key3 = engine._compute_cache_key("Write a plan", ctx4)
    assert key1a != key3, "Different agent should produce different key"

    # Different task, same agent, same depth = different key
    key4 = engine._compute_cache_key("Different task", ctx1)
    assert key1a != key4, "Different task should produce different key"

    # Verify key format
    assert len(key1a) == 16, "Key should be 16 characters"
    assert key1a.isalnum(), "Key should be alphanumeric"


@pytest.mark.asyncio
async def test_cache_hit_and_miss() -> None:
    """Test cache hit/miss behavior and metrics."""
    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(llm=llm, max_depth=1, l1_size=100)

    # First call - cache miss
    result1 = await engine.solve("Write a plan")
    assert llm.call_count == 2, "Should make LLM calls (plan + execute)"
    assert "Mock result" in result1["content"]

    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.0

    # Second call with same task - cache hit
    result2 = await engine.solve("Write a plan")
    assert llm.call_count == 2, "Should not make additional LLM calls"
    assert result2["content"] == result1["content"], "Should return cached result"

    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.5

    # Third call with different task - cache miss
    result3 = await engine.solve("Different task")
    assert llm.call_count == 4, "Should make new LLM calls"
    assert result3["content"] != result1["content"]

    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert abs(stats["hit_rate"] - 0.333) < 0.01


@pytest.mark.asyncio
async def test_lru_eviction() -> None:
    """Test LRU eviction policy when cache size exceeded."""
    llm = MockAsyncLLM()
    # Small cache for testing eviction
    engine = CachedAsyncEngine(llm=llm, max_depth=1, l1_size=3)

    # Fill cache with 3 entries
    await engine.solve("Task 1")
    await engine.solve("Task 2")
    await engine.solve("Task 3")

    stats = engine.get_cache_stats()
    assert stats["l1_size"] == 3
    assert stats["cache_misses"] == 3

    # Add 4th entry - should evict oldest (Task 1)
    await engine.solve("Task 4")

    stats = engine.get_cache_stats()
    assert stats["l1_size"] == 3, "Cache size should not exceed max"
    assert stats["cache_misses"] == 4

    # Access Task 1 again - should be cache miss (was evicted)
    initial_calls = llm.call_count
    await engine.solve("Task 1")
    assert llm.call_count > initial_calls, "Task 1 should require LLM call (evicted)"

    stats = engine.get_cache_stats()
    assert stats["cache_misses"] == 5
    # Cache now contains: Task 3, Task 4, Task 1 (Task 2 was evicted when Task 1 was added)

    # Access Task 3 - should be cache hit (still in cache)
    initial_calls = llm.call_count
    await engine.solve("Task 3")
    assert llm.call_count == initial_calls, "Task 3 should be cached"

    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 1

    # Access Task 2 - should be cache miss (was evicted)
    initial_calls = llm.call_count
    await engine.solve("Task 2")
    assert llm.call_count > initial_calls, "Task 2 should require LLM call (evicted)"

    stats = engine.get_cache_stats()
    assert stats["cache_misses"] == 6


@pytest.mark.asyncio
async def test_cache_with_different_contexts() -> None:
    """Test cache behavior with different agents and depths."""
    llm = MockAsyncLLM()
    agents: dict[str, AsyncLLMCaller] = {"planner": llm, "writer": llm}
    engine = CachedAsyncEngine(
        llm=llm, agents=agents, max_depth=3, l1_size=100
    )

    # Same task, different depths - should be separate cache entries
    ctx_depth0 = RLMContext(
        task_id="1", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent=None
    )
    ctx_depth1 = RLMContext(
        task_id="2", parent_id="1", depth=1,
        breadcrumbs=("step1",), memory_ref=SharedMemory(), active_agent=None
    )

    await engine.solve("Write code", ctx_depth0)
    await engine.solve("Write code", ctx_depth1)

    stats = engine.get_cache_stats()
    assert stats["cache_misses"] == 2, "Different depths should be separate entries"
    assert stats["cache_hits"] == 0

    # Same task, different agents - should be separate cache entries
    ctx_planner = RLMContext(
        task_id="3", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="planner"
    )
    ctx_writer = RLMContext(
        task_id="4", parent_id=None, depth=0,
        breadcrumbs=(), memory_ref=SharedMemory(), active_agent="writer"
    )

    await engine.solve("Write docs", ctx_planner)
    await engine.solve("Write docs", ctx_writer)

    stats = engine.get_cache_stats()
    assert stats["cache_misses"] == 4, "Different agents should be separate entries"
    assert stats["cache_hits"] == 0


@pytest.mark.asyncio
async def test_cache_stats_accuracy() -> None:
    """Test cache statistics calculation."""
    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(llm=llm, max_depth=1, l1_size=100)

    # Initial stats
    stats = engine.get_cache_stats()
    assert stats["hit_rate"] == 0.0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0
    assert stats["l1_size"] == 0
    assert stats["l1_max_size"] == 100

    # Execute same task 10 times
    for _ in range(10):
        await engine.solve("Repeated task")

    stats = engine.get_cache_stats()
    assert stats["cache_hits"] == 9, "9 out of 10 should be hits"
    assert stats["cache_misses"] == 1, "First call should be miss"
    assert stats["hit_rate"] == 0.9
    assert stats["l1_size"] == 1, "Should only have 1 unique entry"


@pytest.mark.asyncio
async def test_verbose_logging(capsys: Any) -> None:
    """Test verbose cache logging output."""
    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(llm=llm, max_depth=1, l1_size=100, verbose=True)

    # First call - should log L1 MISS
    await engine.solve("Test task")
    captured = capsys.readouterr()
    assert "[cache] L1 MISS" in captured.out

    # Second call - should log L1 HIT
    await engine.solve("Test task")
    captured = capsys.readouterr()
    assert "[cache] L1 HIT" in captured.out


@pytest.mark.asyncio
async def test_concurrent_cache_access() -> None:
    """Test cache behavior with concurrent task execution."""
    llm = MockAsyncLLM()
    engine = CachedAsyncEngine(llm=llm, max_depth=1, l1_size=100, max_concurrency=5)

    # Execute 10 identical tasks concurrently
    tasks = [engine.solve("Concurrent task") for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All results should be identical
    first_content = results[0]["content"]
    assert all(r["content"] == first_content for r in results)

    # Note: Due to race conditions, multiple tasks might execute before first cache
    # This is acceptable - we verify cache works for subsequent calls
    stats = engine.get_cache_stats()
    total_requests = stats["cache_hits"] + stats["cache_misses"]
    assert total_requests == 10, "Should track all 10 requests"

    # Execute same task again - should definitely be cached
    initial_calls = llm.call_count
    await engine.solve("Concurrent task")
    assert llm.call_count == initial_calls, "Should use cached result"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
