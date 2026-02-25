from __future__ import annotations

import asyncio
import concurrent.futures
import io
import re
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from rlm.async_engine import AsyncRecursiveEngine
from rlm.exceptions import CodeExecutionError, MaxREPLIterationsError
from rlm.memory import RLMContext, SharedMemory
from rlm.prompts import REPL_SYSTEM_PROMPT
from rlm.types import AsyncLLMCaller, Output, REPLIteration


class _FinalSignal(BaseException):
    """Raised by FINAL() inside REPL code to terminate the loop.

    Uses BaseException so except Exception in LLM-generated code won't catch it.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        super().__init__(answer)


_CODE_BLOCK_RE = re.compile(r"```(?:python|repl)\s*(.*?)\s*```", re.DOTALL)

_SAFE_BUILTINS: frozenset[str] = frozenset({
    "print", "len", "str", "int", "float", "bool", "list", "dict", "tuple",
    "set", "range", "enumerate", "zip", "sorted", "min", "max", "sum",
    "abs", "round", "isinstance", "issubclass", "type", "repr", "hash",
    "iter", "next", "map", "filter", "any", "all",
})


@dataclass
class ContextProxy:
    """Read-only view of context data for REPL access.

    Injected into the REPL namespace as `context`.
    """

    _data: dict[str, str] = field(default_factory=lambda: {})
    _raw: str = ""

    @property
    def memory(self) -> str:
        """Raw string content for programmatic access."""
        return self._raw

    def get(self, key: str) -> str:
        """Return value for exact key, empty string if not found."""
        return self._data.get(key, "")

    def search(self, pattern: str) -> list[str]:
        """Return lines matching pattern (regex or substring)."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            return [line for line in self._raw.splitlines() if regex.search(line)]
        except re.error:
            return [line for line in self._raw.splitlines() if pattern.lower() in line.lower()]

    def keys(self) -> list[str]:
        """Return sorted list of available data keys."""
        return sorted(self._data.keys())

    @classmethod
    def from_raw(cls, raw: str) -> ContextProxy:
        """Build ContextProxy from raw string content."""
        data: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                data[key.strip()] = val.strip()
        return cls(_data=data, _raw=raw)


@dataclass
class REPLConfig:
    """Configuration for the REPL execution sandbox."""

    max_repl_iterations: int = 10
    max_output_chars: int = 4000
    exec_timeout_seconds: float = 10.0
    allowed_builtins: list[str] = field(default_factory=lambda: [])


def _extract_code_block(text: str) -> str | None:
    """Extract first Python/REPL fenced code block from LLM response."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1) if match else None


def _build_safe_globals(
    context_proxy: ContextProxy,
    llm_query_sync: Any,
    final_fn: Any,
    extra_builtins: list[str],
) -> dict[str, Any]:
    """Build restricted globals dict for exec() sandbox."""
    import builtins as _builtins

    allowed = _SAFE_BUILTINS | set(extra_builtins)
    safe_builtins = {
        name: getattr(_builtins, name)
        for name in allowed
        if hasattr(_builtins, name)
    }
    return {
        "__builtins__": safe_builtins,
        "context": context_proxy,
        "llm_query": llm_query_sync,
        "FINAL": final_fn,
        "re": re,
        "json": __import__("json"),
    }


class CodeExecutionEngine(AsyncRecursiveEngine):
    """Async recursive engine where LLMs query context via REPL code.

    The LLM never sees full context in its token window. Instead, it emits
    Python code blocks that query a ContextProxy programmatically.

    Inherits solve(), _plan_async(), _recurse_async() from AsyncRecursiveEngine.
    Overrides solve() to accept context_data and _execute_leaf_async() to
    implement the REPL loop.
    """

    def __init__(
        self,
        llm: AsyncLLMCaller,
        repl_config: REPLConfig | None = None,
        agents: dict[str, AsyncLLMCaller] | None = None,
        router_model: str = "planner",
        max_depth: int = 3,
        max_steps: int = 100,
        max_concurrency: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            agents=agents,
            router_model=router_model,
            max_depth=max_depth,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.repl_config = repl_config or REPLConfig()
        self._context_data_raw: str | None = None

    async def solve(
        self,
        task: str,
        context: RLMContext | None = None,
        context_data: str | dict[str, str] | None = None,
    ) -> Output:
        """Solve task, storing context_data in SharedMemory.

        Args:
            task: Task description
            context: Optional RLMContext (creates root if None)
            context_data: Raw string or key->value dict to store as queryable context.
                          Only consumed at root depth.
        """
        if context is None:
            memory = SharedMemory()
            context = RLMContext(
                task_id=uuid.uuid4().hex[:8],
                parent_id=None,
                depth=0,
                breadcrumbs=(),
                memory_ref=memory,
                active_agent=None,
            )

        if context_data is not None and context.depth == 0:
            if isinstance(context_data, dict):
                raw = "\n".join(f"{k}: {v}" for k, v in context_data.items())
            else:
                raw = context_data
            self._context_data_raw = raw
            ref_id = context.memory_ref.store(raw)
            context.memory_ref.store_named("__repl_context_ref__", ref_id)
            context.memory_ref.store_named("__repl_context_raw__", raw)

        return await super().solve(task, context)

    async def _execute_leaf_async(self, task: str, context: RLMContext) -> Output:
        """Execute leaf task via REPL loop."""
        agent_name = context.active_agent or "default"
        agent = self.agents.get(agent_name, self.llm) if self.agents else self.llm
        context_proxy = self._get_context_proxy(context)
        return await self._run_repl_loop(task, context, context_proxy, agent)

    async def _run_repl_loop(
        self,
        task: str,
        context: RLMContext,
        context_proxy: ContextProxy,
        agent: AsyncLLMCaller,
    ) -> Output:
        """Core REPL iteration loop."""
        loop = asyncio.get_running_loop()
        llm_query_sync = self._make_llm_query_sync(context, loop)
        safe_globals = _build_safe_globals(
            context_proxy,
            llm_query_sync,
            self._make_final_fn(),
            self.repl_config.allowed_builtins,
        )
        messages: list[dict[str, str]] = self._build_initial_messages(task)
        iterations: list[REPLIteration] = []

        for iteration in range(self.repl_config.max_repl_iterations):
            # LLM call
            async with self._semaphore:
                result = await agent(messages, {"mode": "repl"})  # type: ignore[arg-type]

            llm_response = result["content"]
            messages.append({"role": "assistant", "content": llm_response})

            # Extract code block
            code = _extract_code_block(llm_response)
            if code is None:
                # LLM responded without a code block — treat as terminal answer
                return Output(
                    content=llm_response,
                    metadata={
                        "repl_iterations": iteration,
                        "depth": context.depth,
                        "task_id": context.task_id,
                        "mode": "repl",
                    },
                )

            # Execute code
            stdout, error, final_answer = await self._exec_code_async(
                code, safe_globals, loop
            )

            iterations.append(REPLIteration(
                code=code,
                output=stdout,
                error=error,
                iteration=iteration,
            ))

            # Check for FINAL()
            if final_answer is not None:
                return Output(
                    content=final_answer,
                    metadata={
                        "repl_iterations": iteration + 1,
                        "depth": context.depth,
                        "task_id": context.task_id,
                        "mode": "repl",
                        "iterations_detail": iterations,
                    },
                )

            # Build observation
            if error:
                observation = f"Error:\n{error}\n\nOutput (before error):\n{stdout}"
            else:
                observation = stdout
            observation = self._truncate_output(observation)
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

            if self.verbose:
                print(
                    f"[REPL depth={context.depth}] iter={iteration} "
                    f"output_len={len(observation)}"
                )

        raise MaxREPLIterationsError(
            f"REPL loop exceeded max_repl_iterations="
            f"{self.repl_config.max_repl_iterations} for task: {task!r}"
        )

    async def _exec_code_async(
        self,
        code: str,
        safe_globals: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[str, str | None, str | None]:
        """Run code in ThreadPoolExecutor.

        Returns:
            Tuple of (stdout, error_or_None, final_answer_or_None)
        """
        buf = io.StringIO()
        final_answer_container: list[str] = []

        def _captured_print(*args: Any, **kwargs: Any) -> None:
            end = kwargs.get("end", "\n")
            sep = kwargs.get("sep", " ")
            buf.write(sep.join(str(a) for a in args) + end)

        safe_globals["__builtins__"]["print"] = _captured_print

        def _run() -> tuple[str, str | None]:
            try:
                compiled = compile(code, "<repl>", "exec")
                exec(compiled, safe_globals)  # noqa: S102
                return buf.getvalue(), None
            except _FinalSignal as sig:
                final_answer_container.append(sig.answer)
                return buf.getvalue(), None
            except Exception:
                return buf.getvalue(), traceback.format_exc()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = loop.run_in_executor(executor, _run)
            stdout, error = await asyncio.wait_for(
                fut, timeout=self.repl_config.exec_timeout_seconds
            )
        except asyncio.TimeoutError:
            raise CodeExecutionError(
                f"REPL code execution timed out after "
                f"{self.repl_config.exec_timeout_seconds}s"
            )
        finally:
            executor.shutdown(wait=False)

        final_answer = final_answer_container[0] if final_answer_container else None
        return stdout, error, final_answer

    def _make_llm_query_sync(
        self,
        context: RLMContext,
        loop: asyncio.AbstractEventLoop,
    ) -> Any:
        """Create sync llm_query stub for REPL injection.

        Uses asyncio.run_coroutine_threadsafe to bridge sync->async.
        """
        engine = self

        def llm_query(task: str) -> str:
            child_context = context.create_child(
                task_id=uuid.uuid4().hex[:8],
                step_description=f"llm_query: {task[:50]}",
                active_agent=context.active_agent,
            )
            coro = engine.solve(task, child_context)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                result = future.result(timeout=60.0)
                return result["content"]
            except concurrent.futures.TimeoutError:
                raise CodeExecutionError(
                    f"llm_query timed out for task: {task!r}"
                )

        return llm_query

    def _make_final_fn(self) -> Any:
        """Create FINAL() function for REPL injection."""
        def FINAL(answer: str) -> None:
            raise _FinalSignal(str(answer))
        return FINAL

    def _get_context_proxy(self, context: RLMContext) -> ContextProxy:
        """Resolve ContextProxy from shared memory."""
        raw = context.memory_ref.resolve_named("__repl_context_raw__")
        if raw is None and self._context_data_raw is not None:
            raw = self._context_data_raw
        if raw is None:
            return ContextProxy()
        return ContextProxy.from_raw(raw)

    def _truncate_output(self, output: str) -> str:
        """Tail-truncate output to max_output_chars."""
        max_chars = self.repl_config.max_output_chars
        if len(output) <= max_chars:
            return output
        return (
            f"[output truncated — showing last {max_chars} chars]\n"
            + output[-max_chars:]
        )

    def _build_initial_messages(self, task: str) -> list[dict[str, str]]:
        """Build initial conversation history for REPL loop."""
        return [
            {
                "role": "system",
                "content": REPL_SYSTEM_PROMPT.format(
                    max_output_chars=self.repl_config.max_output_chars,
                ),
            },
            {"role": "user", "content": task},
        ]
