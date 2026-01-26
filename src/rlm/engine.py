from __future__ import annotations

import uuid
from typing import Any

from rlm.exceptions import (
    ExecutionError,
    InvalidJSONError,
    MaxStepsError,
    RecursionDepthError,
)
from rlm.memory import RLMContext, SharedMemory
from rlm.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from rlm.types import Input, LLMCaller, Output, SubTask
from rlm.utils import safe_parse_json, validate_planner_decision


class RecursiveEngine:
    """Core recursive execution engine with multi-agent routing.

    Manages the complete lifecycle of recursive task execution:
    - Decides strategy (execute vs recurse) using planner agent
    - Routes sub-tasks to specialized agents based on assignments
    - Enforces depth and step limits
    - Manages RLMContext state
    - Synthesizes results from child agents

    Example (Single-agent mode):
        >>> def my_llm(inputs, context):
        ...     # Your LLM wrapper here
        ...     return {"content": "result", "metadata": {}}
        >>> engine = RecursiveEngine(llm=my_llm, max_depth=3)
        >>> result = engine.solve("Write a marketing plan")
        >>> print(result['content'])

    Example (Multi-agent mode):
        >>> agents = {
        ...     "planner": planner_llm,
        ...     "researcher": researcher_llm,
        ...     "writer": writer_llm,
        ... }
        >>> engine = RecursiveEngine(
        ...     llm=planner_llm,  # Default/fallback
        ...     agents=agents,
        ...     router_model="planner",
        ...     max_depth=3,
        ... )
        >>> result = engine.solve("Research and write a blog post")
    """

    def __init__(
        self,
        llm: LLMCaller,
        agents: dict[str, LLMCaller] | None = None,
        router_model: str = "planner",
        max_depth: int = 3,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> None:
        """Initialize recursive engine with optional multi-agent support.

        Args:
            llm: Default/fallback LLM caller (must match LLMCaller protocol)
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
            verbose: Enable debug logging (default False)
                Logs planner decisions, agent routing, recursion steps

        Raises:
            TypeError: If llm does not match LLMCaller protocol
            ValueError: If router_model not in agents registry
        """
        self.llm = llm  # Default/fallback LLM
        self.agents = agents.copy() if agents else {}  # Agent registry (copy for immutability)
        self.router_model = router_model
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose
        self._step_count = 0
        self._current_subtasks: list[SubTask] = []

        # Validate router_model exists if using multi-agent mode
        if self.agents and self.router_model not in self.agents:
            raise ValueError(
                f"router_model '{self.router_model}' not found in agents registry. "
                f"Available agents: {list(self.agents.keys())}"
            )

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
                active_agent=None,
            )

        # Type narrowing: context is now guaranteed to be RLMContext
        assert context is not None, "Context should be initialized"

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

    def _get_agent(self, agent_name: str | None) -> LLMCaller:
        """Get agent by name from registry with fallback.

        Args:
            agent_name: Name of agent to retrieve (None = use default)

        Returns:
            LLMCaller instance

        Logs warning if agent not found and falls back to default
        """
        if agent_name is None or not self.agents:
            # No agent specified or single-agent mode
            return self.llm

        if agent_name in self.agents:
            return self.agents[agent_name]

        # Agent not found - log warning and fall back
        if self.verbose:
            print(
                f"⚠️  Agent '{agent_name}' not found in registry. "
                f"Falling back to default LLM. "
                f"Available agents: {list(self.agents.keys())}"
            )

        return self.llm

    def _decide_strategy(
        self, task: str, context: RLMContext
    ) -> str:
        """Decide whether to execute atomically or decompose.

        Uses router agent for planning decisions in multi-agent mode.

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

        # Get router agent for planning
        router_agent = self._get_agent(
            self.router_model if self.agents else None
        )

        # Retry logic for malformed JSON (up to 3 attempts)
        # Validation errors (schema issues) are raised immediately
        max_retries = 3
        data: dict[str, Any] = {}  # Initialize to satisfy type checker

        for attempt in range(max_retries):
            try:
                result = router_agent(inputs, {"mode": "planner"})
            except Exception as e:
                raise ExecutionError(f"LLM call failed for task: {task!r}") from e

            # Try to parse JSON
            try:
                data = safe_parse_json(result["content"])
            except InvalidJSONError as e:
                # Malformed JSON - retry with feedback
                if self.verbose:
                    print(
                        f"[Depth {context.depth}] Malformed JSON on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                if attempt < max_retries - 1:
                    # Add error feedback to inputs for retry
                    inputs.append(
                        {
                            "role": "assistant",
                            "content": result["content"],
                        }
                    )
                    inputs.append(
                        {
                            "role": "user",
                            "content": f"Error parsing JSON: {e}. Please provide valid JSON.",
                        }
                    )
                    continue  # Retry
                else:
                    # Final attempt failed - wrap in ExecutionError
                    raise ExecutionError(
                        f"Failed to get valid JSON after {max_retries} attempts for task: {task!r}"
                    ) from e

            # Validate decision schema (no retry for validation errors)
            validate_planner_decision(data)
            break  # Success

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
        """Decompose task and recursively solve sub-tasks with agent routing.

        Routes each sub-task to its assigned agent from the planning decision.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            Synthesized output from all sub-tasks
        """
        sub_tasks: list[SubTask] = self._current_subtasks

        if self.verbose:
            print(
                f"[Depth {context.depth}] Recursing into {len(sub_tasks)} sub-tasks"
            )

        results: list[Output] = []
        for i, sub_task in enumerate(sub_tasks):
            # Extract task description and assigned agent
            task_desc: str = sub_task["description"]
            assigned_agent: str | None = sub_task.get("assigned_agent")

            if self.verbose and assigned_agent:
                print(
                    f"[Depth {context.depth}] Sub-task {i + 1} "
                    f"routed to agent: {assigned_agent}"
                )

            # Create child context with agent assignment
            child_context = context.create_child(
                task_id=uuid.uuid4().hex,
                step_description=f"Sub-task {i + 1}: {task_desc[:50]}",
                active_agent=assigned_agent,
            )

            # Recursive call (solve will use _get_agent internally for execution)
            result = self.solve(task_desc, child_context)
            results.append(result)

        # Synthesize results
        return self._synthesize(task, results, context)

    def _execute_leaf(self, task: str, context: RLMContext) -> Output:
        """Execute task atomically (leaf node).

        Uses the agent specified in context.active_agent for execution,
        respecting multi-agent routing decisions.

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

        # Get the agent assigned to this task (respects routing)
        agent = self._get_agent(context.active_agent)

        try:
            result = agent(inputs, {"mode": "worker"})
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

        Uses router agent or default LLM for synthesis.

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

        synthesis_prompt = f"""{SYNTHESIZER_SYSTEM_PROMPT}

Original task: {original_task}

Sub-task results:
{combined_results}

Synthesize these results into a coherent, comprehensive answer to the original task."""

        inputs: list[Input] = [
            {"role": "system", "content": synthesis_prompt},
            {
                "role": "user",
                "content": "Synthesize the results into a final answer.",
            },
        ]

        # Use dedicated synthesizer agent if available, otherwise router agent
        synthesizer_name = (
            "synthesizer" if "synthesizer" in self.agents
            else (self.router_model if self.agents else None)
        )
        synthesizer = self._get_agent(synthesizer_name)

        try:
            result = synthesizer(inputs, {"mode": "synthesizer"})
        except Exception as e:
            raise ExecutionError(
                f"Synthesis failed for task: {original_task!r}"
            ) from e

        # Add metadata
        result["metadata"]["depth"] = context.depth
        result["metadata"]["task_id"] = context.task_id
        result["metadata"]["breadcrumbs"] = context.breadcrumbs
        result["metadata"]["sub_results_count"] = len(results)

        # Preserve child results for metadata extraction (cost tracking, etc.)
        result["sub_results"] = results

        return result

    def _build_planner_prompt(
        self, task: str, context: RLMContext
    ) -> str:
        """Build system prompt for planner with agent registry.

        Uses PLANNER_SYSTEM_PROMPT template and injects available agents.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            System prompt for planner LLM with agent information
        """
        # Build agent descriptions
        if self.agents:
            available_agents = ", ".join(self.agents.keys())
            # Generate agent descriptions (for now, just list names)
            # TODO: Could enhance with agent capabilities/descriptions
            agent_descriptions = "\n".join(
                f"- {agent_name}: Specialized agent"
                for agent_name in self.agents.keys()
            )
        else:
            available_agents = "None (single-agent mode)"
            agent_descriptions = "No specialized agents available."

        # Format PLANNER_SYSTEM_PROMPT with agent info
        prompt = PLANNER_SYSTEM_PROMPT.format(
            available_agents=available_agents,
            agent_descriptions=agent_descriptions,
        )

        # Append context-specific information
        prompt += f"""

**Current Context:**
- Current depth: {context.depth}
- Max depth: {self.max_depth}
- Remaining depth: {self.max_depth - context.depth}

**Decision Criteria:**
- EXECUTE if: Task is atomic, simple, or cannot be meaningfully decomposed
- RECURSE if: Task is complex, has distinct sub-components, or benefits from parallel execution

**Quality Standards:**
- Sub-tasks should be independent when possible (enables parallelization)
- Each sub-task should have clear, actionable description
- Decomposition should reduce overall complexity
- Avoid over-decomposition (don't split atomic tasks)"""

        return prompt
