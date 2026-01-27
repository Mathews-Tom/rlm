from __future__ import annotations

from rlm.types import AsyncLLMCaller, AsyncToolCaller, Input, Item, Output


class AsyncInput(Input):
    """Async variant of Input for async execution contexts.

    Example:
        >>> async_input: AsyncInput = {"role": "user", "content": "Summarize"}
    """


class AsyncItem(Item):
    """Async variant of Item for tool and structured outputs.

    Example:
        >>> async_item: AsyncItem = {"type": "tool", "content": {"id": 1}}
    """


class AsyncOutput(Output):
    """Async variant of Output for async LLM responses.

    Example:
        >>> async_output: AsyncOutput = {"content": "result", "metadata": {}}
    """


__all__ = [
    "AsyncInput",
    "AsyncItem",
    "AsyncOutput",
    "AsyncLLMCaller",
    "AsyncToolCaller",
]
