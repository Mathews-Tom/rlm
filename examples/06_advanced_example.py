"""Advanced Example - Production-Ready Configuration.

This example demonstrates combining multiple features into a production-ready
configuration using direct LLM calls with native APIs.

Features demonstrated:
1. Tool calling using OpenAI's native function calling API
2. Real-time progress tracking with streaming events
3. Checkpoint-based fault tolerance
4. Combined workflow

This approach demonstrates production patterns without recursive decomposition complexity.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


# === Tool Definitions ===


def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        # WARNING: eval() is unsafe in production! Use a proper parser
        result = eval(expression, {"__builtins__": {}}, {})
        return str(float(result))
    except Exception as e:
        return f"Error: {str(e)}"


def search_information(query: str) -> str:
    """Search for information (simulated).

    Args:
        query: Search query

    Returns:
        Search results (simulated)
    """
    # Simulated search results
    if "machine learning" in query.lower() or "ml" in query.lower():
        return "Machine learning is a subset of AI that enables systems to learn and improve from experience. Key applications include image recognition, natural language processing, recommendation systems, and predictive analytics. Recent advances in deep learning have revolutionized the field."
    return f"Search results for '{query}': Information about {query} including key facts, applications, and current trends."


# OpenAI function schemas for tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression (e.g., '2 + 2')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_information",
            "description": "Search for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

FUNCTION_MAP = {
    "get_current_date": get_current_date,
    "calculate": calculate,
    "search_information": search_information,
}


# === Progress Tracking ===


@dataclass
class ProgressEvent:
    """Progress event emitted during execution."""

    type: str
    data: dict[str, Any]
    timestamp: datetime


# === Checkpointing ===


@dataclass
class Checkpoint:
    """Checkpoint data for resuming execution."""

    checkpoint_id: str
    execution_id: str
    task: str
    step: int
    partial_result: str | None
    timestamp: datetime
    metadata: dict[str, Any]


class InMemoryCheckpointStore:
    """In-memory checkpoint storage."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to storage."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    async def get_latest_checkpoint(self, execution_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for an execution."""
        checkpoints = [
            cp for cp in self._checkpoints.values() if cp.execution_id == execution_id
        ]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: cp.timestamp)


# === Advanced LLM with All Features ===


class AdvancedLLM:
    """Production-ready LLM combining all features.

    Features:
    - Tool calling with progress tracking
    - Checkpoint-based fault tolerance
    - Real-time progress events
    """

    def __init__(
        self, checkpoint_store: InMemoryCheckpointStore, api_key: str | None = None
    ) -> None:
        """Initialize advanced LLM.

        Args:
            checkpoint_store: Storage for checkpoints
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.checkpoint_store = checkpoint_store

    async def solve_advanced(
        self,
        task: str,
        execution_id: str,
        use_tools: bool = True,
        model: str = "gpt-4.1",
        max_iterations: int = 10,
    ) -> AsyncGenerator[ProgressEvent, None]:
        """Solve task with all advanced features.

        Args:
            task: Task description
            execution_id: Unique execution identifier
            use_tools: Whether to enable tool calling
            model: OpenAI model to use
            max_iterations: Maximum iterations for tool calling

        Yields:
            ProgressEvent objects tracking execution
        """
        # Create initial checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=f"{execution_id}-start",
            execution_id=execution_id,
            task=task,
            step=0,
            partial_result=None,
            timestamp=datetime.now(),
            metadata={"status": "started", "use_tools": use_tools},
        )
        await self.checkpoint_store.save_checkpoint(checkpoint)

        yield ProgressEvent(
            type="checkpoint",
            data={"checkpoint_id": checkpoint.checkpoint_id, "step": 0},
            timestamp=datetime.now(),
        )

        yield ProgressEvent(
            type="start", data={"task": task, "use_tools": use_tools}, timestamp=datetime.now()
        )

        try:
            if use_tools:
                # Use tool calling with progress tracking
                messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

                for iteration in range(max_iterations):
                    yield ProgressEvent(
                        type="iteration",
                        data={"iteration": iteration + 1},
                        timestamp=datetime.now(),
                    )

                    # Call OpenAI with function calling
                    response = await self.client.chat.completions.create(
                        model=model, messages=messages, tools=TOOLS, tool_choice="auto"
                    )

                    message = response.choices[0].message
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": message.tool_calls,
                        }
                    )

                    # Check for tool calls
                    if message.tool_calls:
                        yield ProgressEvent(
                            type="tool_calls",
                            data={"count": len(message.tool_calls)},
                            timestamp=datetime.now(),
                        )

                        # Execute each tool call
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)

                            yield ProgressEvent(
                                type="tool_execution",
                                data={"name": function_name, "args": function_args},
                                timestamp=datetime.now(),
                            )

                            try:
                                function_to_call = FUNCTION_MAP[function_name]
                                function_response = function_to_call(**function_args)

                                yield ProgressEvent(
                                    type="tool_result",
                                    data={
                                        "name": function_name,
                                        "result": function_response,
                                    },
                                    timestamp=datetime.now(),
                                )

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": function_response,
                                    }
                                )
                            except Exception as e:
                                yield ProgressEvent(
                                    type="tool_error",
                                    data={"name": function_name, "error": str(e)},
                                    timestamp=datetime.now(),
                                )
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"Error: {str(e)}",
                                    }
                                )

                        continue

                    # No tool calls - final answer
                    if message.content:
                        result = message.content
                        break
                else:
                    result = "Maximum iterations reached"

            else:
                # Simple LLM call without tools
                yield ProgressEvent(
                    type="llm_call", data={"status": "calling_api"}, timestamp=datetime.now()
                )

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": task}],
                    temperature=0.7,
                    max_tokens=2000,
                )

                result = response.choices[0].message.content or ""

            # Save completion checkpoint
            checkpoint = Checkpoint(
                checkpoint_id=f"{execution_id}-complete",
                execution_id=execution_id,
                task=task,
                step=1,
                partial_result=result,
                timestamp=datetime.now(),
                metadata={"status": "completed"},
            )
            await self.checkpoint_store.save_checkpoint(checkpoint)

            yield ProgressEvent(
                type="complete", data={"result": result}, timestamp=datetime.now()
            )
            yield ProgressEvent(
                type="checkpoint",
                data={"checkpoint_id": checkpoint.checkpoint_id, "step": 1},
                timestamp=datetime.now(),
            )

        except Exception as e:
            # Save failure checkpoint
            checkpoint = Checkpoint(
                checkpoint_id=f"{execution_id}-failed",
                execution_id=execution_id,
                task=task,
                step=1,
                partial_result=None,
                timestamp=datetime.now(),
                metadata={"status": "failed", "error": str(e)},
            )
            await self.checkpoint_store.save_checkpoint(checkpoint)

            yield ProgressEvent(
                type="error", data={"error": str(e)}, timestamp=datetime.now()
            )
            raise


async def main() -> None:
    """Run advanced example with all features."""
    print("=" * 80)
    print("RLM Advanced Example: Production-Ready Configuration")
    print("=" * 80)
    print("\nThis example combines:")
    print("  • Tool calling (OpenAI function calling)")
    print("  • Real-time progress tracking")
    print("  • Checkpoint-based fault tolerance")
    print()

    # Create checkpoint store
    checkpoint_store = InMemoryCheckpointStore()

    # Create advanced LLM
    llm = AdvancedLLM(checkpoint_store)

    # Example 1: Tool calling with progress tracking
    print("=" * 80)
    print("EXAMPLE 1: Tool Calling with Progress Tracking")
    print("=" * 80)
    print()

    task1 = """
    What is today's date? Then calculate 25 * 8 + 100.

    Use the available tools to get the date and perform the calculation.
    """

    print(f"Task: {task1.strip()}\n")
    print("Execution:\n")

    execution_id = "advanced-demo-001"

    async for event in llm.solve_advanced(task1, execution_id, use_tools=True):
        if event.type == "start":
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] 🚀 Starting execution")
        elif event.type == "iteration":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] 🔄 Iteration {event.data['iteration']}"
            )
        elif event.type == "tool_calls":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] 🔧 Tool calls: {event.data['count']}"
            )
        elif event.type == "tool_execution":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}]    ↳ {event.data['name']}({event.data['args']})"
            )
        elif event.type == "tool_result":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}]    ✓ Result: {event.data['result']}"
            )
        elif event.type == "complete":
            print(f"\n[{event.timestamp.strftime('%H:%M:%S')}] ✅ Complete\n")
            print("=" * 80)
            print("RESULT")
            print("=" * 80)
            print(event.data["result"])
            print()
        elif event.type == "checkpoint":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] 💾 Checkpoint saved: step {event.data['step']}"
            )
        elif event.type == "error":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] ❌ Error: {event.data['error']}"
            )

    # Example 2: Simple task with checkpointing
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Simple Task with Checkpointing")
    print("=" * 80)
    print()

    task2 = """
    Write a 100-word summary about machine learning applications.
    """

    print(f"Task: {task2.strip()}\n")
    print("Execution:\n")

    execution_id2 = "advanced-demo-002"

    async for event in llm.solve_advanced(task2, execution_id2, use_tools=False):
        if event.type == "start":
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] 🚀 Starting execution")
        elif event.type == "llm_call":
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] 📡 Calling LLM...")
        elif event.type == "complete":
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] ✅ Complete\n")
            print("=" * 80)
            print("RESULT")
            print("=" * 80)
            print(event.data["result"])
            print()
        elif event.type == "checkpoint":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] 💾 Checkpoint saved: step {event.data['step']}"
            )
        elif event.type == "error":
            print(
                f"[{event.timestamp.strftime('%H:%M:%S')}] ❌ Error: {event.data['error']}"
            )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  ✓ Tool calling with OpenAI function calling API")
    print("  ✓ Real-time progress tracking with events")
    print("  ✓ Automatic checkpoint saving")
    print("  ✓ Production-ready error handling")
    print("\nFor production:")
    print("  - Use persistent checkpoint storage")
    print("  - Add comprehensive error recovery")
    print("  - Implement token-level streaming")
    print("  - Add monitoring and alerting")
    print("\nFor recursive decomposition:")
    print("  - Use RecursiveEngine for complex task breakdown")
    print("  - Examples 01-02 demonstrate recursive patterns")
    print("  - Combine with tools, streaming, and checkpointing as needed")


if __name__ == "__main__":
    asyncio.run(main())
