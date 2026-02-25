from __future__ import annotations

from typing import Any

import pytest

from rlm.exceptions import MaxREPLIterationsError
from rlm.repl_engine import (
    CodeExecutionEngine,
    ContextProxy,
    REPLConfig,
    _extract_code_block,
)
from rlm.types import Input, Output


class TestExtractCodeBlock:
    def test_python_block(self) -> None:
        text = 'Some text\n```python\nprint("hello")\n```\nMore text'
        assert _extract_code_block(text) == 'print("hello")'

    def test_repl_block(self) -> None:
        text = '```repl\nx = 1\n```'
        assert _extract_code_block(text) == 'x = 1'

    def test_no_block(self) -> None:
        assert _extract_code_block("just text") is None

    def test_empty_block(self) -> None:
        text = '```python\n```'
        assert _extract_code_block(text) == ''

    def test_multiline_code_block(self) -> None:
        text = '```python\nx = 1\ny = 2\nprint(x + y)\n```'
        assert _extract_code_block(text) == 'x = 1\ny = 2\nprint(x + y)'

    def test_first_block_returned_when_multiple(self) -> None:
        text = '```python\nfirst = 1\n```\n```python\nsecond = 2\n```'
        assert _extract_code_block(text) == 'first = 1'

    def test_plain_code_fence_not_matched(self) -> None:
        """Generic ``` without python/repl tag is not extracted."""
        text = '```\nx = 1\n```'
        assert _extract_code_block(text) is None


class TestContextProxy:
    def test_from_raw(self) -> None:
        raw = "Name: Alice\nAge: 30\nCity: NYC"
        proxy = ContextProxy.from_raw(raw)
        assert proxy.get("Name") == "Alice"
        assert proxy.get("Age") == "30"
        assert proxy.get("missing") == ""

    def test_search_regex(self) -> None:
        raw = "Revenue Q1: $1M\nRevenue Q2: $2M\nCost Q1: $500K"
        proxy = ContextProxy.from_raw(raw)
        results = proxy.search("Revenue")
        assert len(results) == 2
        assert all("Revenue" in r for r in results)

    def test_search_invalid_regex_falls_back(self) -> None:
        raw = "test [bracket\nother line"
        proxy = ContextProxy.from_raw(raw)
        results = proxy.search("[bracket")
        assert len(results) == 1

    def test_keys(self) -> None:
        raw = "B: 2\nA: 1"
        proxy = ContextProxy.from_raw(raw)
        assert proxy.keys() == ["A", "B"]

    def test_memory_property(self) -> None:
        raw = "full content here"
        proxy = ContextProxy.from_raw(raw)
        assert proxy.memory == raw

    def test_empty_proxy(self) -> None:
        proxy = ContextProxy()
        assert proxy.get("key") == ""
        assert proxy.search("anything") == []
        assert proxy.keys() == []
        assert proxy.memory == ""

    def test_search_case_insensitive(self) -> None:
        raw = "Revenue Q1: $1M\nrevenue Q2: $2M\ncost: $500K"
        proxy = ContextProxy.from_raw(raw)
        results = proxy.search("revenue")
        assert len(results) == 2

    def test_keys_sorted(self) -> None:
        raw = "Zebra: 1\nApple: 2\nMango: 3"
        proxy = ContextProxy.from_raw(raw)
        assert proxy.keys() == ["Apple", "Mango", "Zebra"]

    def test_get_strips_whitespace_from_value(self) -> None:
        raw = "Key:   value with spaces   "
        proxy = ContextProxy.from_raw(raw)
        assert proxy.get("Key") == "value with spaces"

    def test_line_without_colon_not_parsed_as_key(self) -> None:
        raw = "no colon here\nKey: value"
        proxy = ContextProxy.from_raw(raw)
        assert proxy.get("no colon here") == ""
        assert proxy.get("Key") == "value"


class TestREPLConfig:
    def test_defaults(self) -> None:
        config = REPLConfig()
        assert config.max_repl_iterations == 10
        assert config.max_output_chars == 4000
        assert config.exec_timeout_seconds == 10.0
        assert config.allowed_builtins == []

    def test_custom_values(self) -> None:
        config = REPLConfig(
            max_repl_iterations=5,
            max_output_chars=1000,
            exec_timeout_seconds=30.0,
            allowed_builtins=["open"],
        )
        assert config.max_repl_iterations == 5
        assert config.max_output_chars == 1000
        assert config.exec_timeout_seconds == 30.0
        assert config.allowed_builtins == ["open"]


_EXECUTE_JSON = '{"thoughts": "atomic task", "decision": "EXECUTE", "sub_tasks": []}'


def _is_planner_call(context: dict[str, Any]) -> bool:
    """Return True when the engine is calling for a planning decision."""
    return "system_prompt" in context


class TestCodeExecutionEngine:
    async def test_single_iteration_final(self) -> None:
        """LLM returns code with FINAL() on first iteration."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {
                "content": '```python\nFINAL("the answer")\n```',
                "metadata": {},
            }

        engine = CodeExecutionEngine(
            llm=mock_llm,
            repl_config=REPLConfig(max_repl_iterations=5),
            max_depth=3,
        )
        result = await engine.solve("test task", context_data="some data")
        assert result["content"] == "the answer"
        assert result["metadata"]["mode"] == "repl"
        assert result["metadata"]["repl_iterations"] == 1

    async def test_context_query(self) -> None:
        """LLM queries context then calls FINAL."""
        repl_call_count = 0

        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            nonlocal repl_call_count
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            repl_call_count += 1
            if repl_call_count == 1:
                return {
                    "content": '```python\nresult = context.search("Q3")\nprint(result)\n```',
                    "metadata": {},
                }
            return {
                "content": '```python\nFINAL("found Q3 data")\n```',
                "metadata": {},
            }

        engine = CodeExecutionEngine(
            llm=mock_llm,
            repl_config=REPLConfig(max_repl_iterations=5),
            max_depth=3,
        )
        result = await engine.solve("find Q3", context_data="Q3 Revenue: $4.2M\nQ4 Revenue: $5.1M")
        assert result["content"] == "found Q3 data"

    async def test_no_code_block_returns_text(self) -> None:
        """LLM responds without code block — treated as final answer."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {"content": "Just a plain answer", "metadata": {}}

        engine = CodeExecutionEngine(llm=mock_llm, max_depth=3)
        result = await engine.solve("test", context_data="data")
        assert result["content"] == "Just a plain answer"

    async def test_max_iterations_exceeded(self) -> None:
        """Raises MaxREPLIterationsError when loop exhausted."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {
                "content": '```python\nprint("still working")\n```',
                "metadata": {},
            }

        engine = CodeExecutionEngine(
            llm=mock_llm,
            repl_config=REPLConfig(max_repl_iterations=2),
            max_depth=3,
        )
        with pytest.raises(MaxREPLIterationsError):
            await engine.solve("test", context_data="data")

    async def test_truncate_output(self) -> None:
        engine = CodeExecutionEngine(
            llm=lambda i, c: None,  # type: ignore[arg-type, return-value]
            repl_config=REPLConfig(max_output_chars=20),
            max_depth=3,
        )
        short = engine._truncate_output("short")
        assert short == "short"

        long_text = "a" * 100
        truncated = engine._truncate_output(long_text)
        assert truncated.endswith("a" * 20)
        assert "truncated" in truncated

    async def test_no_code_block_metadata_mode_repl(self) -> None:
        """Plain text response has mode=repl in metadata."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {"content": "Direct answer", "metadata": {}}

        engine = CodeExecutionEngine(llm=mock_llm, max_depth=3)
        result = await engine.solve("test", context_data="x: 1")
        assert result["metadata"]["mode"] == "repl"

    async def test_final_with_numeric_string(self) -> None:
        """FINAL() coerces argument to str."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {
                "content": '```python\nFINAL(42)\n```',
                "metadata": {},
            }

        engine = CodeExecutionEngine(
            llm=mock_llm,
            repl_config=REPLConfig(max_repl_iterations=3),
            max_depth=3,
        )
        result = await engine.solve("compute", context_data="")
        assert result["content"] == "42"

    async def test_context_data_dict_converted_to_raw(self) -> None:
        """Dict context_data is flattened to 'key: value' lines."""
        async def mock_llm(inputs: list[Any], context: dict[str, Any]) -> Output:
            if _is_planner_call(context):
                return {"content": _EXECUTE_JSON, "metadata": {}}
            return {
                "content": '```python\nvalue = context.get("revenue")\nFINAL(value)\n```',
                "metadata": {},
            }

        engine = CodeExecutionEngine(
            llm=mock_llm,
            repl_config=REPLConfig(max_repl_iterations=3),
            max_depth=3,
        )
        result = await engine.solve("get revenue", context_data={"revenue": "$5M"})
        assert result["content"] == "$5M"
