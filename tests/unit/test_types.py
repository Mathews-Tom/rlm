from __future__ import annotations

from typing import Any

from rlm.types import Input, LLMCaller, Output


def test_input_typeddict() -> None:
    """Verify Input TypedDict structure."""
    inp: Input = {"role": "user", "content": "Hello"}
    assert inp["role"] == "user"
    assert inp["content"] == "Hello"


def test_output_typeddict() -> None:
    """Verify Output TypedDict structure."""
    out: Output = {"content": "Response", "metadata": {}}
    assert "content" in out
    assert "metadata" in out
    assert out["content"] == "Response"


def test_llm_caller_protocol() -> None:
    """Verify LLMCaller protocol compliance."""

    def my_llm(inputs: list[Input], context: dict[str, Any]) -> Output:
        # Echo the inputs count in response to use the parameter
        return {
            "content": f"test response (processed {len(inputs)} inputs)",
            "metadata": {"context_keys": len(context)},
        }

    # Protocol check (type checker validates this)
    caller: LLMCaller = my_llm
    result = caller([{"role": "user", "content": "test"}], {"key": "value"})
    assert "test response" in result["content"]
    assert "metadata" in result
