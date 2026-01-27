"""Streaming Example - Real-Time Progress Updates.

This example demonstrates how to track execution progress in real-time
using custom progress callbacks with direct LLM calls.

Key concepts:
1. Emit progress events during execution
2. Track different execution stages
3. Provide real-time feedback to users
4. Handle errors with event emission

This approach works with direct LLM calls for maximum simplicity.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


class ProgressEvent:
    """Progress event emitted during execution."""

    def __init__(self, event_type: str, data: dict[str, Any]) -> None:
        self.type = event_type
        self.data = data
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"ProgressEvent(type={self.type}, data={self.data})"


class StreamingLLM:
    """Simple LLM caller with progress tracking.

    Wraps OpenAI API to emit progress events during execution.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize streaming LLM.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def solve_with_progress(
        self, task: str, model: str = "gpt-4.1"
    ) -> AsyncGenerator[ProgressEvent, None]:
        """Solve task with progress events.

        Args:
            task: Task description
            model: OpenAI model to use

        Yields:
            ProgressEvent objects tracking execution progress
        """
        # Emit start event
        yield ProgressEvent("start", {"task": task, "model": model})

        try:
            # Emit LLM call event
            yield ProgressEvent("llm_call", {"status": "calling_api"})

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task}],
                temperature=0.7,
                max_tokens=2000,
            )

            # Extract result
            content = response.choices[0].message.content or ""
            metadata = {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }

            # Emit completion event
            yield ProgressEvent(
                "complete",
                {
                    "result": content,
                    "metadata": metadata,
                },
            )

        except Exception as e:
            # Emit error event
            yield ProgressEvent("error", {"error": str(e)})
            raise


async def process_stream_with_ui(llm: StreamingLLM, task: str) -> None:
    """Process streaming events with UI updates.

    Args:
        llm: Streaming LLM instance
        task: Task to solve
    """
    print("\n" + "=" * 80)
    print("EXECUTION STREAM")
    print("=" * 80 + "\n")

    start_time = datetime.now()

    async for event in llm.solve_with_progress(task):
        elapsed = (datetime.now() - start_time).total_seconds()

        if event.type == "start":
            print(f"[{elapsed:>6.2f}s] 🚀 Starting task execution...")
            print(f"            Task: {event.data['task'][:60]}...")
            print(f"            Model: {event.data['model']}")

        elif event.type == "llm_call":
            print(f"[{elapsed:>6.2f}s] 📡 Calling OpenAI API...")

        elif event.type == "complete":
            print(f"\n[{elapsed:>6.2f}s] ✅ Task completed successfully")
            print(f"\n{'='*80}")
            print("FINAL RESULT")
            print(f"{'='*80}")
            result_content = event.data["result"]
            print(result_content)
            print()

            # Print token usage
            metadata = event.data.get("metadata", {})
            usage = metadata.get("usage", {})
            if usage:
                print(f"{'='*80}")
                print("STATISTICS")
                print(f"{'='*80}")
                print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
                print(f"Completion tokens: {usage.get('completion_tokens', 0)}")
                print(f"Total tokens: {usage.get('total_tokens', 0)}")
                print(f"Execution time: {elapsed:.2f}s")

        elif event.type == "error":
            print(f"\n[{elapsed:>6.2f}s] ❌ Error: {event.data['error']}")


async def main() -> None:
    """Run streaming example with progress tracking."""
    print("=" * 80)
    print("RLM Streaming Example: Real-Time Progress Updates")
    print("=" * 80)
    print("\nThis example demonstrates progress tracking with direct LLM calls.")
    print("Progress events are emitted during execution for real-time feedback.\\n")

    # Create streaming LLM
    llm = StreamingLLM()

    # Example 1: Simple task
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Writing Task")
    print("=" * 80)

    task1 = """
    Write a 150-word summary about artificial intelligence.

    Include: definition, key applications, and future outlook.
    """

    try:
        await process_stream_with_ui(llm, task1.strip())
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    # Example 2: Slightly more complex task
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Analysis Task")
    print("=" * 80)

    task2 = """
    Compare electric vehicles vs gasoline vehicles in 3 key areas:
    cost, environmental impact, and convenience.

    Keep it brief (200 words total).
    """

    try:
        await process_stream_with_ui(llm, task2.strip())
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  ✓ Progress event emission during execution")
    print("  ✓ Real-time status updates")
    print("  ✓ Timestamp tracking")
    print("  ✓ Error handling with events")
    print("\nFor production:")
    print("  - Add more granular progress events (token streaming)")
    print("  - Implement progress bars or UI updates")
    print("  - Add cancellation support")
    print("  - Stream token-by-token for LLM output")
    print("\nFor recursive decomposition with streaming:")
    print("  - Use RecursiveEngine with event hooks")
    print("  - Emit events at key execution points (plan, execute, synthesize)")
    print("  - Track sub-task progress separately")


if __name__ == "__main__":
    asyncio.run(main())
