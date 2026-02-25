from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class UsageAccumulator:
    """Thread/async-safe running total of token usage and cost.

    Sync engines use _lock (threading.Lock).
    Async engines use _async_lock (asyncio.Lock, created lazily).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _async_lock: asyncio.Lock | None = field(default=None, repr=False, compare=False)

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create asyncio.Lock (must be called inside event loop)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def add(self, usage: TokenUsage) -> None:
        """Thread-safe accumulation (for sync engine)."""
        with self._lock:
            self._add_unsafe(usage)

    async def add_async(self, usage: TokenUsage) -> None:
        """Async-safe accumulation (for async engine)."""
        async with self._get_async_lock():
            self._add_unsafe(usage)

    def _add_unsafe(self, usage: TokenUsage) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cached_tokens += usage.cached_tokens
        self.cost_usd += usage.cost_usd
        self.call_count += 1

    def snapshot(self) -> dict[str, int | float]:
        """Return point-in-time copy of all counters."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
            "call_count": self.call_count,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.cached_tokens = 0
            self.cost_usd = 0.0
            self.call_count = 0


def extract_usage(result: Mapping[str, Any]) -> TokenUsage:
    """Extract TokenUsage from an Output dict.

    Checks Output["usage"] first (typed path), then
    Output["metadata"]["usage"] (legacy path).
    Returns zero-usage if absent (backward compatible).
    """
    raw: dict[str, Any] = result.get("usage", {})
    if not raw:
        raw = result.get("metadata", {}).get("usage", {})
    if not raw:
        return TokenUsage()

    prompt = int(raw.get("prompt_tokens", 0))
    completion = int(raw.get("completion_tokens", 0))
    total = int(raw.get("total_tokens", prompt + completion))
    cached = int(raw.get("cached_tokens", 0))
    cost = float(raw.get("cost_usd", 0.0))

    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        cost_usd=cost,
    )


__all__ = [
    "TokenUsage",
    "UsageAccumulator",
    "extract_usage",
]
