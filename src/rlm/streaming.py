"""Streaming event protocol for progressive results via AsyncGenerator.

This module defines the StreamEvent dataclass and event type system used by
StreamingEngine to emit real-time updates during task execution. Events are
JSON-serializable for transport over SSE (Server-Sent Events) or WebSocket.

Event Types:
    - plan: Emitted after planning decision (decompose vs execute)
    - token: Emitted during LLM token generation (real-time streaming)
    - result: Emitted when a sub-task or leaf task completes
    - error: Emitted when an error occurs during execution

Example:
    >>> from rlm.streaming import StreamEvent
    >>> event = StreamEvent(
    ...     type="token",
    ...     data={"content": "Hello"},
    ...     metadata={"depth": 0, "task_id": "task-123", "timestamp": "2026-01-27T16:30:00Z"}
    ... )
    >>> json_str = event.to_json()
    >>> print(json_str)
    '{"type": "token", "data": {"content": "Hello"}, "metadata": {...}}'
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from rlm.exceptions import ExecutionError
from rlm.memory import RLMContext, SharedMemory
from rlm.prompts import SYNTHESIZER_SYSTEM_PROMPT
from rlm.tools import ToolCallingEngine
from rlm.types import Input

EventType = Literal["plan", "token", "result", "error"]


@dataclass
class StreamEvent:
    """Event emitted during streaming execution.

    Attributes:
        type: Event type (plan, token, result, error)
        data: Event payload (varies by type)
        metadata: Execution metadata (depth, task_id, timestamp)

    Example:
        >>> event = StreamEvent(
        ...     type="plan",
        ...     data={"decision": "decompose", "sub_tasks": ["task1", "task2"]},
        ...     metadata={"depth": 0, "task_id": "root", "timestamp": "2026-01-27T16:30:00Z"}
        ... )
        >>> event.type
        'plan'
    """

    type: EventType
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event structure after initialization.

        Ensures required metadata fields are present and adds timestamp if missing.

        Raises:
            ValueError: If type is not a valid EventType
            ValueError: If required metadata fields are missing
        """
        # Validate event type
        valid_types: tuple[str, ...] = ("plan", "token", "result", "error")
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid event type '{self.type}'. Must be one of: {valid_types}"
            )

        # Ensure required metadata fields
        if "task_id" not in self.metadata:
            raise ValueError("Metadata must include 'task_id'")

        if "depth" not in self.metadata:
            raise ValueError("Metadata must include 'depth'")

        # Add timestamp if not present
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        """Serialize event to JSON string for transport.

        Returns:
            JSON string representation of event

        Example:
            >>> event = StreamEvent(
            ...     type="token",
            ...     data={"content": "Hello"},
            ...     metadata={"depth": 0, "task_id": "task-123"}
            ... )
            >>> json_str = event.to_json()
            >>> isinstance(json_str, str)
            True
        """
        return json.dumps(
            {"type": self.type, "data": self.data, "metadata": self.metadata},
            separators=(",", ":"),  # Compact JSON
        )

    @classmethod
    def from_json(cls, json_str: str) -> StreamEvent:
        """Deserialize event from JSON string.

        Args:
            json_str: JSON string to deserialize

        Returns:
            StreamEvent instance

        Raises:
            ValueError: If JSON is invalid or missing required fields

        Example:
            >>> json_str = '{"type":"token","data":{"content":"Hi"},"metadata":{"depth":0,"task_id":"t1"}}'
            >>> event = StreamEvent.from_json(json_str)
            >>> event.type
            'token'
        """
        try:
            data = json.loads(json_str)
            return cls(
                type=data["type"], data=data["data"], metadata=data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid StreamEvent JSON: {e}") from e

    @classmethod
    def plan_event(
        cls, decision: str, sub_tasks: list[str] | None, task_id: str, depth: int
    ) -> StreamEvent:
        """Create a plan event after planning decision.

        Args:
            decision: Planning decision ("decompose" or "execute")
            sub_tasks: List of sub-task descriptions (None for execute)
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="plan"

        Example:
            >>> event = StreamEvent.plan_event(
            ...     decision="decompose",
            ...     sub_tasks=["task1", "task2"],
            ...     task_id="root",
            ...     depth=0
            ... )
            >>> event.type
            'plan'
            >>> event.data["decision"]
            'decompose'
        """
        return cls(
            type="plan",
            data={
                "decision": decision,
                "sub_tasks": sub_tasks,
            },
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def token_event(cls, content: str, task_id: str, depth: int) -> StreamEvent:
        """Create a token event during LLM generation.

        Args:
            content: Token or partial text content
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="token"

        Example:
            >>> event = StreamEvent.token_event(
            ...     content="Hello",
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'token'
            >>> event.data["content"]
            'Hello'
        """
        return cls(
            type="token",
            data={"content": content},
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def result_event(
        cls, content: str, metadata: dict[str, Any], task_id: str, depth: int
    ) -> StreamEvent:
        """Create a result event when task completes.

        Args:
            content: Final result content
            metadata: Result metadata (tokens, tool_calls, etc.)
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="result"

        Example:
            >>> event = StreamEvent.result_event(
            ...     content="Task complete",
            ...     metadata={"tokens": 150},
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'result'
            >>> event.data["content"]
            'Task complete'
        """
        return cls(
            type="result",
            data={"content": content, "result_metadata": metadata},
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )

    @classmethod
    def error_event(
        cls, error: str, error_type: str, task_id: str, depth: int
    ) -> StreamEvent:
        """Create an error event when execution fails.

        Args:
            error: Error message
            error_type: Error type/class name
            task_id: Unique task identifier
            depth: Recursion depth

        Returns:
            StreamEvent with type="error"

        Example:
            >>> event = StreamEvent.error_event(
            ...     error="Connection timeout",
            ...     error_type="TimeoutError",
            ...     task_id="task-1",
            ...     depth=1
            ... )
            >>> event.type
            'error'
            >>> event.data["error"]
            'Connection timeout'
        """
        return cls(
            type="error",
            data={
                "error": error,
                "error_type": error_type,
            },
            metadata={
                "task_id": task_id,
                "depth": depth,
            },
        )


class StreamingEngine(ToolCallingEngine):
    """Async recursive engine with streaming event emission.

    Extends ToolCallingEngine to emit real-time progress via AsyncGenerator
    of StreamEvent objects. Events are emitted for planning decisions, LLM
    tokens, sub-task results, and errors.

    Events enable progressive UI updates and real-time user feedback during
    long-running task execution.

    Example:
        >>> engine = StreamingEngine(
        ...     llm=my_llm,
        ...     tool_registry=registry,
        ...     max_depth=3
        ... )
        >>>
        >>> async for event in engine.solve_streaming("Complex task"):
        ...     if event.type == "token":
        ...         print(event.data["content"], end="", flush=True)
        ...     elif event.type == "result":
        ...         print(f"\\nCompleted: {event.data['content']}")
    """

    async def solve_streaming(
        self, task: str, context: RLMContext | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Solve task with streaming events emitted during execution.

        Yields StreamEvent objects as execution progresses:
        - plan: After planning decision (execute vs recurse)
        - token: During LLM generation (if LLM supports streaming)
        - result: When sub-tasks or leaf tasks complete
        - error: When execution errors occur

        Args:
            task: Task description to solve
            context: Optional execution context (creates default if None)

        Yields:
            StreamEvent objects tracking execution progress

        Raises:
            ExecutionError: If task execution fails catastrophically

        Example:
            >>> async for event in engine.solve_streaming("Summarize article"):
            ...     match event.type:
            ...         case "plan":
            ...             print(f"Planning: {event.data['decision']}")
            ...         case "token":
            ...             print(event.data["content"], end="")
            ...         case "result":
            ...             print(f"Result: {event.data['content']}")
            ...         case "error":
            ...             print(f"Error: {event.data['error']}")
        """
        if context is None:
            context = RLMContext(
                task_id=str(uuid.uuid4()),
                parent_id=None,
                depth=0,
                breadcrumbs=(),
                memory_ref=SharedMemory(),
                active_agent=None,
            )

        async for event in self._solve_recursive_streaming(task, context):
            yield event

    async def _solve_recursive_streaming(
        self, task: str, context: RLMContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Recursively solve task with streaming events.

        Internal method that handles the recursive decomposition and execution
        with event emission at each step.

        Args:
            task: Task description to solve
            context: Execution context with depth and breadcrumb tracking

        Yields:
            StreamEvent objects during execution
        """
        try:
            # Call planner to decide execute vs recurse
            strategy = await self._plan_async(task, context)

            # Emit plan event after planning decision
            if strategy == "recurse":
                yield StreamEvent.plan_event(
                    decision="decompose",
                    sub_tasks=None,  # Sub-tasks determined after planning
                    task_id=context.task_id,
                    depth=context.depth,
                )
            else:
                yield StreamEvent.plan_event(
                    decision="execute",
                    sub_tasks=None,
                    task_id=context.task_id,
                    depth=context.depth,
                )

            # Execute based on strategy
            if strategy == "recurse":
                # Recursive case: decompose into sub-tasks
                async for event in self._execute_decompose_streaming(task, context):
                    yield event
            else:
                # Leaf case: execute directly
                async for event in self._execute_leaf_streaming(task, context):
                    yield event

        except Exception:
            # Let exception propagate without emitting additional error events
            # Error event is emitted at the source (e.g., _execute_leaf_with_streaming)
            raise

    async def _execute_decompose_streaming(
        self, task: str, context: RLMContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute decomposed sub-tasks with streaming events.

        Args:
            task: Original task description
            context: Current execution context

        Yields:
            StreamEvent objects from sub-task execution
        """
        # Get decomposer agent (fallback to default LLM)
        decomposer = self.agents.get(context.active_agent or "default", self.llm)

        # Get sub-tasks via decomposition
        decompose_input: Input = {
            "role": "user",
            "content": f"Break down this task into 2-4 sub-tasks:\n{task}",
        }

        try:
            async with self._semaphore:
                decompose_result = await decomposer(
                    [decompose_input],
                    {
                        "system_prompt": "You are a task decomposition expert. Break down complex tasks into simpler sub-tasks.",
                        "temperature": 0.0,
                    },
                )

            # Parse sub-tasks from result
            # For simplicity, assume LLM returns numbered list
            sub_tasks = [
                line.strip()
                for line in decompose_result["content"].split("\n")
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))
            ]

            # Execute each sub-task recursively
            sub_results: list[str] = []
            for i, sub_task in enumerate(sub_tasks[:4]):  # Limit to 4 sub-tasks
                # Clean up numbered/bulleted format
                sub_task = sub_task.lstrip("0123456789.-) \t")

                # Create sub-context
                sub_context = RLMContext(
                    task_id=f"{context.task_id}.{i}",
                    parent_id=context.task_id,
                    depth=context.depth + 1,
                    breadcrumbs=context.breadcrumbs + (task,),
                    memory_ref=context.memory_ref,
                    active_agent=context.active_agent,
                )

                # Recursively solve sub-task with streaming
                async for event in self._solve_recursive_streaming(sub_task, sub_context):
                    yield event

                    # Collect result when sub-task completes
                    if event.type == "result" and event.metadata["task_id"] == sub_context.task_id:
                        sub_results.append(event.data["content"])

            # Synthesize final result from sub-results
            synthesis_prompt = self._create_synthesis_prompt(task, sub_results)

            # Get agent for synthesis
            agent = self.agents.get(context.active_agent or "default", self.llm)

            async with self._semaphore:
                final_output = await agent(
                    [{"role": "user", "content": synthesis_prompt}],
                    {
                        "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
                        "task_id": context.task_id,
                        "depth": context.depth,
                    },
                )

            # Emit result event for synthesized output
            yield StreamEvent.result_event(
                content=final_output["content"],
                metadata=final_output.get("metadata", {}),
                task_id=context.task_id,
                depth=context.depth,
            )

        except Exception as e:
            yield StreamEvent.error_event(
                error=f"Decomposition/synthesis failed: {e}",
                error_type=type(e).__name__,
                task_id=context.task_id,
                depth=context.depth,
            )
            raise ExecutionError(f"Failed to execute decomposed task: {e}") from e

    async def _execute_leaf_streaming(
        self, task: str, context: RLMContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute leaf task with token-level streaming events.

        Checks if the agent supports streaming (has a 'stream' method).
        If streaming is supported, yields token events as the LLM generates them.
        Otherwise, falls back to batch mode with a single result event.

        Token-level streaming achieves <500ms time-to-first-token (TTFT) for
        responsive UI updates during long-running LLM generation.

        Args:
            task: Leaf task description
            context: Execution context

        Yields:
            StreamEvent objects:
            - token events during streaming generation
            - result event with final content at the end
            - error events if execution fails

        Note:
            Currently only supports streaming for the final LLM response without
            tool calling. Tool calling loop uses batch mode for simplicity.
            Future enhancement: stream tokens during tool calling iterations.
        """
        try:
            # Get the agent for this task
            agent = self.agents.get(context.active_agent or "default", self.llm)

            # Check if agent supports streaming
            has_streaming = hasattr(agent, "stream") and callable(
                getattr(agent, "stream")
            )

            if has_streaming:
                # Streaming mode: emit token events as they arrive
                async for event in self._execute_leaf_with_streaming(task, context, agent):
                    yield event
            else:
                # Fallback to batch mode
                output = await self._execute_leaf_async(task, context)

                # Emit result event
                yield StreamEvent.result_event(
                    content=output["content"],
                    metadata=output.get("metadata", {}),
                    task_id=context.task_id,
                    depth=context.depth,
                )

        except Exception:
            # Let exception propagate without emitting additional error events
            # Error event is emitted at the source (e.g., _execute_leaf_with_streaming)
            raise

    async def _execute_leaf_with_streaming(
        self, task: str, context: RLMContext, agent: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute leaf task with token-level streaming from LLM.

        Calls agent.stream() to get token-by-token generation and emits
        token events as they arrive. Collects all tokens to emit final
        result event.

        Args:
            task: Leaf task description
            context: Execution context
            agent: LLM agent with streaming support

        Yields:
            StreamEvent objects for tokens and final result
        """
        # Build conversation history
        inputs: list[Input] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use tools when needed to answer questions.",
            },
            {"role": "user", "content": task},
        ]

        # Stream tokens from LLM
        collected_tokens: list[str] = []

        try:
            async with self._semaphore:
                async for token in agent.stream(inputs, {"mode": "worker"}):
                    # Emit token event
                    yield StreamEvent.token_event(
                        content=token,
                        task_id=context.task_id,
                        depth=context.depth,
                    )

                    # Collect token for final content
                    collected_tokens.append(token)

            # Build final content
            final_content = "".join(collected_tokens)

            # Emit result event with complete content
            yield StreamEvent.result_event(
                content=final_content,
                metadata={
                    "depth": context.depth,
                    "task_id": context.task_id,
                    "streaming": True,
                    "token_count": len(collected_tokens),
                },
                task_id=context.task_id,
                depth=context.depth,
            )

        except Exception as e:
            # Handle streaming errors
            # If we collected some tokens, include them in error
            partial_content = "".join(collected_tokens) if collected_tokens else ""

            yield StreamEvent.error_event(
                error=f"Streaming failed: {e}" + (f" (partial content: {partial_content[:100]}...)" if partial_content else ""),
                error_type=type(e).__name__,
                task_id=context.task_id,
                depth=context.depth,
            )
            raise

    def _create_synthesis_prompt(
        self, original_task: str, sub_results: list[str]
    ) -> str:
        """Create prompt for synthesizing sub-results into final answer.

        Args:
            original_task: Original task description
            sub_results: List of sub-task results to synthesize

        Returns:
            Synthesis prompt for LLM
        """
        results_text = "\n\n".join(
            f"Sub-task {i+1} result:\n{result}"
            for i, result in enumerate(sub_results)
        )

        return f"""Original task: {original_task}

Sub-task results:
{results_text}

Synthesize the above sub-task results into a comprehensive answer to the original task.
Provide a coherent, well-structured response that integrates all relevant information."""


__all__ = [
    "EventType",
    "StreamEvent",
    "StreamingEngine",
]
