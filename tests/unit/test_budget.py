from __future__ import annotations

import pytest

from rlm.budget import TokenUsage, UsageAccumulator, extract_usage


class TestTokenUsage:
    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.cost_usd == 0.0

    def test_custom_values(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.01)
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 150

    def test_all_fields(self) -> None:
        usage = TokenUsage(
            prompt_tokens=200,
            completion_tokens=75,
            total_tokens=275,
            cached_tokens=50,
            cost_usd=0.005,
        )
        assert usage.completion_tokens == 75
        assert usage.cached_tokens == 50
        assert usage.cost_usd == pytest.approx(0.005)


class TestUsageAccumulator:
    def test_add_sync(self) -> None:
        acc = UsageAccumulator()
        acc.add(TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.01))
        acc.add(TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost_usd=0.02))
        assert acc.prompt_tokens == 300
        assert acc.completion_tokens == 150
        assert acc.total_tokens == 450
        assert acc.cost_usd == pytest.approx(0.03)
        assert acc.call_count == 2

    async def test_add_async(self) -> None:
        acc = UsageAccumulator()
        await acc.add_async(TokenUsage(prompt_tokens=100, total_tokens=100))
        await acc.add_async(TokenUsage(prompt_tokens=200, total_tokens=200))
        assert acc.prompt_tokens == 300
        assert acc.call_count == 2

    def test_snapshot(self) -> None:
        acc = UsageAccumulator()
        acc.add(TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75))
        snap = acc.snapshot()
        assert snap == {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "call_count": 1,
        }

    def test_reset(self) -> None:
        acc = UsageAccumulator()
        acc.add(TokenUsage(prompt_tokens=100, total_tokens=100))
        acc.reset()
        assert acc.prompt_tokens == 0
        assert acc.call_count == 0

    def test_initial_state(self) -> None:
        acc = UsageAccumulator()
        assert acc.prompt_tokens == 0
        assert acc.completion_tokens == 0
        assert acc.total_tokens == 0
        assert acc.cached_tokens == 0
        assert acc.cost_usd == 0.0
        assert acc.call_count == 0

    def test_cached_tokens_accumulated(self) -> None:
        acc = UsageAccumulator()
        acc.add(TokenUsage(cached_tokens=30))
        acc.add(TokenUsage(cached_tokens=70))
        assert acc.cached_tokens == 100

    def test_snapshot_after_reset(self) -> None:
        acc = UsageAccumulator()
        acc.add(TokenUsage(prompt_tokens=100, total_tokens=100, cost_usd=0.01))
        acc.reset()
        snap = acc.snapshot()
        assert snap["prompt_tokens"] == 0
        assert snap["cost_usd"] == 0.0
        assert snap["call_count"] == 0

    async def test_add_async_accumulates_cost(self) -> None:
        acc = UsageAccumulator()
        await acc.add_async(TokenUsage(cost_usd=0.005))
        await acc.add_async(TokenUsage(cost_usd=0.003))
        assert acc.cost_usd == pytest.approx(0.008)
        assert acc.call_count == 2


class TestExtractUsage:
    def test_from_usage_key(self) -> None:
        result = {"content": "hi", "metadata": {}, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        usage = extract_usage(result)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15  # auto-computed

    def test_from_metadata_legacy(self) -> None:
        result = {"content": "hi", "metadata": {"usage": {"prompt_tokens": 20, "total_tokens": 30}}}
        usage = extract_usage(result)
        assert usage.prompt_tokens == 20
        assert usage.total_tokens == 30

    def test_no_usage(self) -> None:
        result = {"content": "hi", "metadata": {}}
        usage = extract_usage(result)
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

    def test_empty_usage(self) -> None:
        result = {"content": "hi", "metadata": {}, "usage": {}}
        usage = extract_usage(result)
        assert usage.prompt_tokens == 0

    def test_top_level_usage_takes_priority(self) -> None:
        """Top-level usage key wins over metadata.usage."""
        result = {
            "content": "hi",
            "metadata": {"usage": {"prompt_tokens": 99}},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        usage = extract_usage(result)
        assert usage.prompt_tokens == 10

    def test_cost_usd_extracted(self) -> None:
        result = {"content": "hi", "metadata": {}, "usage": {"cost_usd": 0.042}}
        usage = extract_usage(result)
        assert usage.cost_usd == pytest.approx(0.042)

    def test_cached_tokens_extracted(self) -> None:
        result = {
            "content": "hi",
            "metadata": {},
            "usage": {"prompt_tokens": 100, "cached_tokens": 40, "total_tokens": 100},
        }
        usage = extract_usage(result)
        assert usage.cached_tokens == 40

    def test_total_tokens_auto_computed_when_absent(self) -> None:
        result = {
            "content": "hi",
            "metadata": {},
            "usage": {"prompt_tokens": 30, "completion_tokens": 20},
        }
        usage = extract_usage(result)
        assert usage.total_tokens == 50

    def test_total_tokens_used_when_present(self) -> None:
        result = {
            "content": "hi",
            "metadata": {},
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 60},
        }
        usage = extract_usage(result)
        assert usage.total_tokens == 60
