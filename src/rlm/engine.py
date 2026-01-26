from __future__ import annotations

import uuid

from rlm.memory import RLMContext, SharedMemory
from rlm.types import (
    ExecutionError,
    Input,
    LLMCaller,
    MaxStepsError,
    Output,
    RecursionDepthError,
)
from rlm.utils import safe_parse_json, validate_planner_decision


class RecursiveEngine:
    """Core recursive execution engine for task decomposition.

    Manages the complete lifecycle of recursive task execution:
    - Decides strategy (execute vs recurse)
    - Enforces depth and step limits
    - Manages RLMContext state
    - Synthesizes results from child agents

    Example:
        >>> def my_llm(inputs, context):
        ...     # Your LLM wrapper here
        ...     return {"content": "result", "metadata": {}}
        >>> engine = RecursiveEngine(llm=my_llm, max_depth=3)
        >>> result = engine.solve("Write a marketing plan")
        >>> print(result['content'])
    """

    def __init__(
        self,
        llm: LLMCaller,
        max_depth: int = 3,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> None:
        """Initialize recursive engine.

        Args:
            llm: Backend LLM caller (must match LLMCaller protocol)
            max_depth: Maximum recursion depth (default 3)
                Prevents infinite recursion by limiting tree depth.
            max_steps: Maximum total steps across all levels (default 100)
                Prevents runaway execution in wide trees.
            verbose: Enable debug logging (default False)
                Logs planner decisions, recursion steps, etc.

        Raises:
            TypeError: If llm does not match LLMCaller protocol
        """
        self.llm = llm
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose
        self._step_count = 0

    def solve(
        self, task: str, context: RLMContext | None = None
    ) -> Output:
        """Solve task through recursive decomposition.

        This is the main entry point for task execution.

        Args:
            task: Natural language description of task to solve
            context: Optional execution context (used for recursive calls)
                If None, creates root context with depth=0.

        Returns:
            Output dict with:
                - content (str): Final result text
                - metadata (dict): Execution metadata (depth, steps, etc.)

        Raises:
            RecursionDepthError: If depth exceeds max_depth
            MaxStepsError: If total steps exceed max_steps
            InvalidJSONError: If LLM returns unparseable JSON
            ExecutionError: If LLM call fails

        Example:
            >>> result = engine.solve("Design a marketing strategy")
            >>> print(result['content'])
            "1. Define target audience..."
            >>> result['metadata']
            {'depth': 2, 'steps': 5, 'duration': 12.3}
        """
        # 1. Initialize context (root or child)
        if context is None:
            # Root call - create initial context
            memory = SharedMemory()
            context = RLMContext(
                task_id=uuid.uuid4().hex,
                parent_id=None,
                depth=0,
                breadcrumbs=(),
                memory_ref=memory,
            )

        # 2. Enforce depth limit
        if context.depth >= self.max_depth:
            raise RecursionDepthError(
                f"Exceeded max_depth={self.max_depth} at depth={context.depth}. "
                f"Task: {task!r}, Breadcrumbs: {context.breadcrumbs}"
            )

        # 3. Enforce step limit
        self._step_count += 1
        if self._step_count > self.max_steps:
            raise MaxStepsError(
                f"Exceeded max_steps={self.max_steps} (current: {self._step_count})"
            )

        # 4. Decide strategy (via planner)
        decision = self._decide_strategy(task, context)

        # 5. Execute or recurse
        if decision == "EXECUTE":
            return self._execute_leaf(task, context)
        else:  # RECURSE
            return self._recurse(task, context)

    def _decide_strategy(
        self, task: str, context: RLMContext
    ) -> str:
        """Decide whether to execute atomically or decompose.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            "EXECUTE" or "RECURSE"

        Raises:
            ExecutionError: If LLM call fails
            InvalidJSONError: If response is invalid JSON
        """
        planner_prompt = self._build_planner_prompt(task, context)
        inputs: list[Input] = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": task},
        ]

        try:
            result = self.llm(inputs, {"mode": "planner"})
        except Exception as e:
            raise ExecutionError(f"LLM call failed for task: {task!r}") from e

        # Parse and validate decision
        data = safe_parse_json(result["content"])
        validate_planner_decision(data)

        decision: str = data["decision"]

        if self.verbose:
            print(
                f"[Depth {context.depth}] Decision: {decision} for task: {task}"
            )

        # Store sub_tasks if RECURSE
        if decision == "RECURSE":
            self._current_subtasks = data["sub_tasks"]
        else:
            self._current_subtasks = []

        return decision

    def _recurse(self, task: str, context: RLMContext) -> Output:
        """Decompose task and recursively solve sub-tasks.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            Synthesized output from all sub-tasks
        """
        sub_tasks: list[str] = self._current_subtasks

        if self.verbose:
            print(
                f"[Depth {context.depth}] Recursing into {len(sub_tasks)} sub-tasks"
            )

        results: list[Output] = []
        for i, sub_task in enumerate(sub_tasks):
            # Create child context
            child_context = context.create_child(
                task_id=uuid.uuid4().hex,
                step_description=f"Sub-task {i + 1}: {sub_task[:50]}",
            )

            # Recursive call
            result = self.solve(sub_task, child_context)
            results.append(result)

        # Synthesize results
        return self._synthesize(task, results, context)

    def _execute_leaf(self, task: str, context: RLMContext) -> Output:
        """Execute task atomically (leaf node).

        Args:
            task: Task description
            context: Current execution context

        Returns:
            Direct output from LLM

        Raises:
            ExecutionError: If LLM call fails
        """
        if self.verbose:
            print(f"[Depth {context.depth}] Executing leaf task: {task}")

        inputs: list[Input] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question directly and concisely.",
            },
            {"role": "user", "content": task},
        ]

        try:
            result = self.llm(inputs, {"mode": "worker"})
        except Exception as e:
            raise ExecutionError(f"LLM call failed for task: {task!r}") from e

        # Add metadata
        result["metadata"]["depth"] = context.depth
        result["metadata"]["task_id"] = context.task_id
        result["metadata"]["breadcrumbs"] = context.breadcrumbs

        return result

    def _synthesize(
        self,
        original_task: str,
        results: list[Output],
        context: RLMContext,
    ) -> Output:
        """Synthesize results from sub-tasks into coherent output.

        Args:
            original_task: Original task description
            results: List of outputs from sub-tasks
            context: Current execution context

        Returns:
            Synthesized output

        Raises:
            ExecutionError: If LLM call fails
        """
        if self.verbose:
            print(f"[Depth {context.depth}] Synthesizing {len(results)} results")

        # Combine results for synthesis
        combined_results = "\n\n".join(
            f"Result {i + 1}:\n{result['content']}"
            for i, result in enumerate(results)
        )

        synthesis_prompt = f"""You are synthesizing results from multiple sub-tasks.

Original task: {original_task}

Sub-task results:
{combined_results}

Synthesize these results into a coherent, comprehensive answer to the original task.
Ensure the answer is well-structured and addresses all aspects of the original task."""

        inputs: list[Input] = [
            {"role": "system", "content": synthesis_prompt},
            {
                "role": "user",
                "content": "Synthesize the results into a final answer.",
            },
        ]

        try:
            result = self.llm(inputs, {"mode": "synthesizer"})
        except Exception as e:
            raise ExecutionError(
                f"Synthesis failed for task: {original_task!r}"
            ) from e

        # Add metadata
        result["metadata"]["depth"] = context.depth
        result["metadata"]["task_id"] = context.task_id
        result["metadata"]["breadcrumbs"] = context.breadcrumbs
        result["metadata"]["sub_results_count"] = len(results)

        return result

    def _build_planner_prompt(
        self, task: str, context: RLMContext
    ) -> str:
        """Build system prompt for planner.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            System prompt for planner LLM
        """
        return f"""You are a task planning assistant. Analyze the task and decide whether to:

1. EXECUTE: Solve the task directly (simple, atomic task)
2. RECURSE: Break down into independent sub-tasks (complex task requiring multiple steps)

Current depth: {context.depth}
Max depth: {self.max_depth}
Remaining depth: {self.max_depth - context.depth}

Guidelines:
- Prefer EXECUTE for simple, direct questions
- Use RECURSE for complex tasks requiring multiple steps or domains
- Each sub-task should be independent and focused
- Aim for 2-4 sub-tasks when decomposing
- If near max depth, prefer EXECUTE

Respond with JSON:
{{
  "thoughts": "Your reasoning about task complexity",
  "decision": "EXECUTE" or "RECURSE",
  "sub_tasks": ["task 1", "task 2", ...]  // Required only if RECURSE
}}"""
