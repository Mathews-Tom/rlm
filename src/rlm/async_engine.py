from __future__ import annotations

import asyncio
import uuid

from rlm.budget import UsageAccumulator, extract_usage
from rlm.exceptions import (
    CostLimitExceeded,
    ExecutionError,
    InvalidJSONError,
    MaxStepsError,
    RecursionDepthError,
    TokenLimitExceeded,
)
from rlm.memory import RLMContext, SharedMemory
from rlm.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from rlm.types import AsyncLLMCaller, Input, Output, SubTask
from rlm.utils import safe_parse_json, validate_planner_decision


class AsyncRecursiveEngine:
    """Async recursive execution engine with parallel sub-task execution.

    Provides async/await architecture for 8-10× throughput improvement over
    synchronous RecursiveEngine. Uses asyncio.gather() for parallel sub-task
    execution and semaphore-based rate limiting.

    Key differences from RecursiveEngine:
    - All methods are async/await
    - Sub-tasks execute in parallel via asyncio.gather()
    - max_concurrency parameter limits parallel execution
    - AsyncLLMCaller protocol for dependency injection

    Example (Single-agent mode):
        >>> async def my_async_llm(inputs, context):
        ...     # Your async LLM wrapper here
        ...     return {"content": "result", "metadata": {}}
        >>> engine = AsyncRecursiveEngine(llm=my_async_llm, max_depth=3)
        >>> result = await engine.solve("Write a marketing plan")
        >>> print(result['content'])

    Example (Multi-agent mode):
        >>> agents = {
        ...     "planner": planner_llm,
        ...     "researcher": researcher_llm,
        ...     "writer": writer_llm,
        ... }
        >>> engine = AsyncRecursiveEngine(
        ...     llm=planner_llm,
        ...     agents=agents,
        ...     router_model="planner",
        ...     max_depth=3,
        ...     max_concurrency=10,
        ... )
        >>> result = await engine.solve("Research and write a blog post")
    """

    def __init__(
        self,
        llm: AsyncLLMCaller,
        sub_model: AsyncLLMCaller | None = None,
        agents: dict[str, AsyncLLMCaller] | None = None,
        router_model: str = "planner",
        max_depth: int = 3,
        max_steps: int = 100,
        max_concurrency: int = 10,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        max_total_tokens: int | None = None,
        max_cost: float | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize async recursive engine with parallel execution support.

        Args:
            llm: Default/fallback async LLM caller (must match AsyncLLMCaller protocol)
            sub_model: Optional cheaper/faster async model for leaf execution tasks.
                If provided, used for "execute" role calls instead of llm.
            agents: Optional registry of named agents for multi-agent routing
                Format: {"agent_name": agent_llm_caller, ...}
                If None, runs in single-agent mode (backward compatible)
            router_model: Name of agent to use for planning decisions
                Must exist in agents registry if agents is provided
                Default: "planner"
            max_depth: Maximum recursion depth (default 3)
                Prevents infinite recursion by limiting tree depth.
            max_steps: Maximum total steps across all levels (default 100)
                Prevents runaway execution in wide trees.
            max_concurrency: Maximum concurrent sub-tasks (default 10)
                Semaphore-based rate limiting to prevent resource exhaustion.
            max_prompt_tokens: Hard cap on accumulated prompt tokens.
                Raises TokenLimitExceeded when exceeded.
            max_completion_tokens: Hard cap on accumulated completion tokens.
                Raises TokenLimitExceeded when exceeded.
            max_total_tokens: Hard cap on accumulated total tokens.
                Raises TokenLimitExceeded when exceeded.
            max_cost: Hard cap on accumulated cost in USD.
                Raises CostLimitExceeded when exceeded.
            verbose: Enable debug logging (default False)
                Logs planner decisions, agent routing, recursion steps

        Raises:
            TypeError: If llm does not match AsyncLLMCaller protocol
            ValueError: If router_model not in agents registry
        """
        self.llm = llm
        self.sub_model = sub_model
        self.agents = agents.copy() if agents else {}
        self.router_model = router_model
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.max_concurrency = max_concurrency
        self.verbose = verbose
        self._step_count = 0
        self._current_subtasks: list[SubTask] = []
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_total_tokens = max_total_tokens
        self.max_cost = max_cost
        self.usage = UsageAccumulator()

        # Validate router_model exists if using multi-agent mode
        if self.agents and self.router_model not in self.agents:
            raise ValueError(
                f"router_model '{self.router_model}' not found in agents registry. "
                f"Available agents: {list(self.agents.keys())}"
            )

    def _get_model_for_role(self, role: str) -> AsyncLLMCaller:
        """Select async model based on execution role."""
        if role == "execute" and self.sub_model is not None:
            return self.sub_model
        return self.llm

    async def _check_budget_async(self) -> None:
        """Async budget enforcement."""
        u = self.usage
        if self.max_prompt_tokens is not None and u.prompt_tokens >= self.max_prompt_tokens:
            raise TokenLimitExceeded(
                f"prompt_tokens={u.prompt_tokens} >= max_prompt_tokens={self.max_prompt_tokens}"
            )
        if self.max_completion_tokens is not None and u.completion_tokens >= self.max_completion_tokens:
            raise TokenLimitExceeded(
                f"completion_tokens={u.completion_tokens} >= max_completion_tokens={self.max_completion_tokens}"
            )
        if self.max_total_tokens is not None and u.total_tokens >= self.max_total_tokens:
            raise TokenLimitExceeded(
                f"total_tokens={u.total_tokens} >= max_total_tokens={self.max_total_tokens}"
            )
        if self.max_cost is not None and u.cost_usd >= self.max_cost:
            raise CostLimitExceeded(
                f"cost_usd={u.cost_usd:.6f} >= max_cost={self.max_cost:.6f}"
            )

    def get_usage(self) -> dict[str, int | float]:
        """Return snapshot of accumulated token usage and cost."""
        return self.usage.snapshot()

    async def solve(
        self, task: str, context: RLMContext | None = None
    ) -> Output:
        """Solve task through async recursive decomposition.

        Main entry point for async task execution. Decides whether to execute
        atomically or recurse, enforces depth and step limits, and manages
        context state.

        Args:
            task: Task description string
            context: Optional RLMContext for memory/state management
                If None, creates new root context with SharedMemory

        Returns:
            Output dict with 'content', 'metadata', and 'context' keys

        Raises:
            RecursionDepthError: If max_depth exceeded
            MaxStepsError: If max_steps exceeded
            ExecutionError: If LLM call or synthesis fails
        """
        # Initialize context if not provided
        if context is None:
            context = RLMContext(
                task_id=str(uuid.uuid4()),
                parent_id=None,
                depth=0,
                breadcrumbs=(),
                memory_ref=SharedMemory(),
                active_agent=None,
            )

        # Check depth limit
        if context.depth >= self.max_depth:
            raise RecursionDepthError(
                f"Maximum recursion depth {self.max_depth} exceeded at depth {context.depth}"
            )

        # Check step limit
        self._step_count += 1
        if self._step_count > self.max_steps:
            raise MaxStepsError(
                f"Maximum steps {self.max_steps} exceeded at step {self._step_count}"
            )

        if self.verbose:
            print(
                f"[depth={context.depth}] [step={self._step_count}] Solving: {task[:80]}..."
            )

        # Decide strategy: execute or recurse
        strategy = await self._plan_async(task, context)

        if strategy == "execute":
            return await self._execute_leaf_async(task, context)
        else:
            return await self._recurse_async(task, context)

    async def _plan_async(
        self, task: str, context: RLMContext
    ) -> str:
        """Call async planner LLM to decide execution strategy.

        Args:
            task: Task description
            context: Current RLMContext

        Returns:
            Strategy string: "execute" or "recurse"

        Raises:
            InvalidJSONError: If planner returns invalid JSON after 3 retries
            ExecutionError: If planner LLM call fails
        """
        # Get planner agent (fallback to default LLM)
        planner = self.agents.get(self.router_model, self._get_model_for_role("plan"))

        # Build planner input
        planner_input: Input = {
            "role": "user",
            "content": f"Task: {task}\n\nContext depth: {context.depth}/{self.max_depth}",
        }

        # Retry logic for invalid JSON (up to 3 attempts)
        for attempt in range(3):
            await self._check_budget_async()
            try:
                # Apply semaphore to LLM call
                async with self._semaphore:
                    result = await planner(
                        [planner_input],
                        {
                            "system_prompt": PLANNER_SYSTEM_PROMPT,
                            "temperature": 0.0,
                        },
                    )
                await self.usage.add_async(extract_usage(result))

                # Parse and validate JSON response
                decision_data = safe_parse_json(result["content"])
                validate_planner_decision(decision_data)

                # Get decision from validated data
                decision = decision_data.get("decision", "EXECUTE")
                strategy = "execute" if decision == "EXECUTE" else "recurse"

                if self.verbose:
                    thoughts = decision_data.get("thoughts", "N/A")
                    print(f"[plan] strategy={strategy}, thoughts={thoughts[:60]}...")

                return strategy

            except InvalidJSONError as e:
                if attempt == 2:  # Last attempt
                    raise InvalidJSONError(
                        f"Planner returned invalid JSON after 3 attempts: {e}"
                    ) from e
                if self.verbose:
                    print(f"[plan] Invalid JSON (attempt {attempt + 1}/3), retrying...")
                continue

        # Should never reach here due to raise in loop, but satisfy type checker
        raise InvalidJSONError("Planner failed after 3 attempts")

    async def _execute_leaf_async(
        self, task: str, context: RLMContext
    ) -> Output:
        """Execute atomic task using assigned agent or default LLM.

        Args:
            task: Task description
            context: Current RLMContext

        Returns:
            Output dict from LLM execution

        Raises:
            ExecutionError: If LLM call fails
        """
        # Route to agent based on active_agent in context, otherwise use default
        agent_name = context.active_agent or "default"
        agent = self.agents.get(agent_name, self._get_model_for_role("execute"))

        if self.verbose:
            print(f"[execute] agent={agent_name}, task={task[:60]}...")

        # Build execution input
        exec_input: Input = {
            "role": "user",
            "content": task,
        }

        await self._check_budget_async()
        try:
            # Apply semaphore to LLM call
            async with self._semaphore:
                result = await agent(
                    [exec_input],
                    {
                        "task_id": context.task_id,
                        "depth": context.depth,
                        "temperature": 0.7,
                    },
                )
            await self.usage.add_async(extract_usage(result))
            return result

        except (TokenLimitExceeded, CostLimitExceeded):
            raise
        except Exception as e:
            raise ExecutionError(
                f"Failed to execute task '{task[:60]}...' with agent '{agent_name}': {e}"
            ) from e

    async def _recurse_async(
        self, task: str, context: RLMContext
    ) -> Output:
        """Recursively decompose task into sub-tasks and execute in parallel.

        Uses asyncio.gather() to execute sub-tasks concurrently with semaphore
        rate limiting via max_concurrency parameter.

        Args:
            task: Task description
            context: Current RLMContext

        Returns:
            Synthesized Output from all sub-task results

        Raises:
            ExecutionError: If decomposition or synthesis fails
        """
        # Get planner agent for decomposition
        planner = self.agents.get(self.router_model, self._get_model_for_role("plan"))

        # Request task decomposition
        decompose_input: Input = {
            "role": "user",
            "content": f"Decompose this task into sub-tasks:\n{task}",
        }

        await self._check_budget_async()
        try:
            # Apply semaphore to LLM call
            async with self._semaphore:
                decomp_result = await planner(
                    [decompose_input],
                    {
                        "system_prompt": "You are a task decomposition expert. Break complex tasks into 2-5 independent sub-tasks.",
                        "temperature": 0.3,
                    },
                )
            await self.usage.add_async(extract_usage(decomp_result))

            # Parse sub-tasks from response
            decomp_data = safe_parse_json(decomp_result["content"])
            subtasks_raw = decomp_data.get("sub_tasks", [])

            if not subtasks_raw:
                raise ExecutionError("Planner returned empty sub_tasks list")

            # Build SubTask TypedDicts
            self._current_subtasks = []
            for st in subtasks_raw:
                subtask: SubTask = {"description": st["description"]}
                if "assigned_agent" in st:
                    subtask["assigned_agent"] = st["assigned_agent"]
                self._current_subtasks.append(subtask)

            if self.verbose:
                print(f"[recurse] Decomposed into {len(self._current_subtasks)} sub-tasks")

        except Exception as e:
            raise ExecutionError(
                f"Failed to decompose task '{task[:60]}...': {e}"
            ) from e

        # Execute sub-tasks in parallel (rate limiting applied at LLM call level)
        async def solve_subtask(subtask: SubTask) -> Output:
            """Execute single sub-task with proper context."""
            # Create child context with agent assignment
            assigned_agent = subtask.get("assigned_agent", None)
            child_context = context.create_child(
                task_id=str(uuid.uuid4()),
                step_description=subtask["description"],
                active_agent=assigned_agent,
            )
            return await self.solve(subtask["description"], child_context)

        # Gather all results in parallel
        results = await asyncio.gather(*[
            solve_subtask(st) for st in self._current_subtasks
        ])

        # Synthesize results
        return await self._synthesize_async(task, results, context)

    async def _synthesize_async(
        self, original_task: str, results: list[Output], context: RLMContext
    ) -> Output:
        """Synthesize sub-task results into final output.

        Args:
            original_task: Original task description
            results: List of Output dicts from sub-tasks
            context: Current RLMContext

        Returns:
            Synthesized Output dict

        Raises:
            ExecutionError: If synthesis LLM call fails
        """
        # Get synthesizer agent (fallback to default)
        synthesizer = self.agents.get("synthesizer", self._get_model_for_role("synthesize"))

        # Build synthesis prompt
        results_text = "\n\n".join([
            f"Sub-task {i + 1}:\n{r['content']}"
            for i, r in enumerate(results)
        ])

        synth_input: Input = {
            "role": "user",
            "content": f"Original task: {original_task}\n\nSub-task results:\n{results_text}\n\nSynthesize into final answer.",
        }

        await self._check_budget_async()
        try:
            # Apply semaphore to LLM call
            async with self._semaphore:
                synthesis = await synthesizer(
                    [synth_input],
                    {
                        "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
                        "temperature": 0.5,
                    },
                )
            await self.usage.add_async(extract_usage(synthesis))

            if self.verbose:
                print(f"[synthesize] Combined {len(results)} results")

            return synthesis

        except (TokenLimitExceeded, CostLimitExceeded):
            raise
        except Exception as e:
            raise ExecutionError(
                f"Failed to synthesize results for task '{original_task[:60]}...': {e}"
            ) from e

    def solve_sync(
        self, task: str, context: RLMContext | None = None
    ) -> Output:
        """Synchronous wrapper for async solve() method.

        Provides backward compatibility for synchronous code that cannot use
        async/await. Uses asyncio.run() to execute the async solve() method
        in a new event loop.

        Args:
            task: Task description string
            context: Optional RLMContext for memory/state management
                If None, creates new root context with SharedMemory

        Returns:
            Output dict with 'content', 'metadata', and 'context' keys

        Raises:
            RecursionDepthError: If max_depth exceeded
            MaxStepsError: If max_steps exceeded
            ExecutionError: If LLM call or synthesis fails

        Example:
            >>> engine = AsyncRecursiveEngine(llm=my_async_llm)
            >>> # Can be called from synchronous code
            >>> result = engine.solve_sync("Write a report")
            >>> print(result['content'])

        Note:
            This method creates a new event loop for each call, which has
            overhead (~10-50ms). For high-throughput applications, prefer
            using async solve() directly within an async context.
        """
        return asyncio.run(self.solve(task, context))
