from __future__ import annotations


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

    def my_llm(inputs: list[Input], context: dict) -> Output:
        return {"content": "test response", "metadata": {}}

    # Protocol check (type checker validates this)
    caller: LLMCaller = my_llm
    result = caller([], {})
    assert result["content"] == "test response"
    assert "metadata" in result
