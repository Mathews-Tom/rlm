from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from rlm.async_engine import AsyncRecursiveEngine
from rlm.engine import RecursiveEngine
from rlm.memory import RLMContext, SharedMemory
from rlm.types import AsyncLLMCaller, Input, LLMCaller, Output


# Mock LLM callers for benchmarking
class MockSyncLLM:
    """Mock synchronous LLM with configurable delay."""

    def __init__(self, delay_ms: int = 100) -> None:
        self.delay_ms = delay_ms
        self.call_count = 0

    def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
        """Simulate LLM call with delay."""
        time.sleep(self.delay_ms / 1000)
        self.call_count += 1

        # Return simple execution decision for planner
        if context.get("mode") == "planner":
            return {
                "content": '{"thoughts": "Execute atomically", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"model": "mock-sync", "delay_ms": self.delay_ms},
            }

        return {
            "content": f"Mock result from sync LLM (call #{self.call_count})",
            "metadata": {"model": "mock-sync", "delay_ms": self.delay_ms},
        }


class MockAsyncLLM:
    """Mock async LLM with configurable delay."""

    def __init__(self, delay_ms: int = 100) -> None:
        self.delay_ms = delay_ms
        self.call_count = 0

    async def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
        """Simulate async LLM call with delay."""
        await asyncio.sleep(self.delay_ms / 1000)
        self.call_count += 1

        # Return simple execution decision
        if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
            return {
                "content": '{"thoughts": "Execute atomically", "decision": "EXECUTE", "sub_tasks": []}',
                "metadata": {"model": "mock-async", "delay_ms": self.delay_ms},
            }

        return {
            "content": f"Mock result from async LLM (call #{self.call_count})",
            "metadata": {"model": "mock-async", "delay_ms": self.delay_ms},
        }


@pytest.mark.benchmark
def test_async_overhead_single_call() -> None:
    """Measure async overhead for single task execution.

    Acceptance: Async overhead <100ms per call
    """
    llm = MockAsyncLLM(delay_ms=50)
    engine = AsyncRecursiveEngine(llm=llm, max_depth=1, verbose=False)

    # Measure async execution time
    start = time.perf_counter()
    result = asyncio.run(engine.solve("Simple task"))
    async_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Measure synchronous wrapper overhead
    start = time.perf_counter()
    result_sync = engine.solve_sync("Simple task")
    sync_wrapper_time = (time.perf_counter() - start) * 1000

    # Calculate overhead (should be minimal for single call)
    overhead = async_time - llm.delay_ms
    wrapper_overhead = sync_wrapper_time - async_time

    print(f"\nSingle Call Performance:")
    print(f"  LLM delay: {llm.delay_ms}ms")
    print(f"  Async execution: {async_time:.2f}ms")
    print(f"  Async overhead: {overhead:.2f}ms")
    print(f"  solve_sync() wrapper: {sync_wrapper_time:.2f}ms")
    print(f"  Wrapper overhead: {wrapper_overhead:.2f}ms")

    # Verify results are equivalent (both contain mock result)
    assert "Mock result from async LLM" in result["content"]
    assert "Mock result from async LLM" in result_sync["content"]

    # Acceptance criteria: async overhead <100ms
    assert overhead < 100, f"Async overhead {overhead:.2f}ms exceeds 100ms threshold"
    assert wrapper_overhead < 50, f"Wrapper overhead {wrapper_overhead:.2f}ms too high"


@pytest.mark.benchmark
async def test_parallel_throughput_improvement() -> None:
    """Demonstrate 8-10× throughput improvement with parallel execution.

    Acceptance: 8-10× speedup for 10 concurrent tasks
    """
    delay_ms = 100
    num_tasks = 10

    # Create sync and async engines
    sync_llm = MockSyncLLM(delay_ms=delay_ms)
    async_llm = MockAsyncLLM(delay_ms=delay_ms)

    sync_engine = RecursiveEngine(llm=sync_llm, max_depth=1, verbose=False)
    async_engine = AsyncRecursiveEngine(
        llm=async_llm, max_depth=1, max_concurrency=10, verbose=False
    )

    # Benchmark synchronous execution (sequential)
    start = time.perf_counter()
    sync_results = []
    for i in range(num_tasks):
        result = sync_engine.solve(f"Task {i + 1}")
        sync_results.append(result)
    sync_time = time.perf_counter() - start

    # Benchmark async execution (parallel)
    start = time.perf_counter()
    async_results = await asyncio.gather(*[
        async_engine.solve(f"Task {i + 1}") for i in range(num_tasks)
    ])
    async_time = time.perf_counter() - start

    # Calculate speedup
    speedup = sync_time / async_time

    print(f"\nThroughput Benchmark ({num_tasks} tasks, {delay_ms}ms LLM delay):")
    print(f"  Synchronous (sequential): {sync_time:.3f}s")
    print(f"  Async (parallel): {async_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  LLM calls - Sync: {sync_llm.call_count}, Async: {async_llm.call_count}")

    # Verify results are equivalent
    assert len(sync_results) == len(async_results) == num_tasks

    # Acceptance criteria: 8-10× throughput improvement
    assert speedup >= 8.0, f"Speedup {speedup:.2f}× below 8× threshold"
    assert speedup <= 12.0, f"Speedup {speedup:.2f}× suspiciously high (expected 8-10×)"


@pytest.mark.benchmark
async def test_semaphore_rate_limiting() -> None:
    """Verify semaphore prevents resource exhaustion under high concurrency.

    Tests that max_concurrency properly limits parallel execution.
    """
    delay_ms = 50
    num_tasks = 20
    max_concurrency = 5

    # Track concurrent execution count
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    class ConcurrencyTrackingLLM:
        """Mock LLM that tracks concurrent executions."""

        async def __call__(self, inputs: list[Input], context: dict[str, Any]) -> Output:
            nonlocal concurrent_count, max_concurrent

            async with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent:
                    max_concurrent = concurrent_count

            await asyncio.sleep(delay_ms / 1000)

            async with lock:
                concurrent_count -= 1

            if "system_prompt" in context and "planner" in context.get("system_prompt", "").lower():
                return {
                    "content": '{"thoughts": "Execute", "decision": "EXECUTE", "sub_tasks": []}',
                    "metadata": {},
                }

            return {"content": "Result", "metadata": {}}

    llm = ConcurrencyTrackingLLM()
    engine = AsyncRecursiveEngine(
        llm=llm, max_depth=1, max_concurrency=max_concurrency, verbose=False
    )

    # Execute tasks in parallel
    start = time.perf_counter()
    results = await asyncio.gather(*[
        engine.solve(f"Task {i + 1}") for i in range(num_tasks)
    ])
    elapsed = time.perf_counter() - start

    print(f"\nSemaphore Rate Limiting ({num_tasks} tasks, max_concurrency={max_concurrency}):")
    print(f"  Execution time: {elapsed:.3f}s")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Expected batches: {num_tasks // max_concurrency}")

    # Verify rate limiting worked
    assert len(results) == num_tasks
    assert max_concurrent <= max_concurrency, (
        f"Max concurrent {max_concurrent} exceeded limit {max_concurrency}"
    )
    assert max_concurrent >= max_concurrency - 1, (
        f"Max concurrent {max_concurrent} too low, semaphore may not be working"
    )


@pytest.mark.benchmark
def test_backward_compatibility_solve_sync() -> None:
    """Verify solve_sync() provides backward compatibility with synchronous code.

    Tests that synchronous code can use AsyncRecursiveEngine without modification.
    """
    llm = MockAsyncLLM(delay_ms=50)
    engine = AsyncRecursiveEngine(llm=llm, max_depth=2, verbose=False)

    # Test basic execution
    result = engine.solve_sync("Write a simple task")
    assert "content" in result
    assert "metadata" in result

    # Test with custom context
    memory = SharedMemory()
    context = RLMContext(
        task_id="test-123",
        parent_id=None,
        depth=0,
        breadcrumbs=(),
        memory_ref=memory,
        active_agent=None,
    )

    result_with_context = engine.solve_sync("Another task", context)
    assert "content" in result_with_context

    # Test exception propagation
    from rlm.exceptions import RecursionDepthError

    shallow_engine = AsyncRecursiveEngine(llm=llm, max_depth=0, verbose=False)

    with pytest.raises(RecursionDepthError):
        shallow_engine.solve_sync("This will exceed depth")

    print("\nBackward Compatibility:")
    print("  ✓ solve_sync() works from synchronous code")
    print("  ✓ Context handling preserved")
    print("  ✓ Exception propagation works correctly")


if __name__ == "__main__":
    # Run benchmarks directly
    print("=" * 60)
    print("ASYNC PERFORMANCE BENCHMARKS")
    print("=" * 60)

    print("\n[1/4] Testing async overhead...")
    test_async_overhead_single_call()

    print("\n[2/4] Testing parallel throughput...")
    asyncio.run(test_parallel_throughput_improvement())

    print("\n[3/4] Testing semaphore rate limiting...")
    asyncio.run(test_semaphore_rate_limiting())

    print("\n[4/4] Testing backward compatibility...")
    test_backward_compatibility_solve_sync()

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS PASSED ✓")
    print("=" * 60)
