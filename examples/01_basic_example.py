"""Basic RLM Example - Simple Recursive Task Decomposition.

This example demonstrates the core functionality of RLM:
1. Task decomposition via planner agent
2. Recursive sub-task execution
3. Result synthesis

The engine automatically decides whether to:
- EXECUTE: Solve the task directly (atomic task)
- RECURSE: Decompose into sub-tasks (complex task)
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rlm.engine import RecursiveEngine
from rlm.exceptions import MaxStepsError, RecursionDepthError
from rlm.types import Input, Output

# Load environment variables
load_dotenv()


def create_openai_caller() -> Any:
    """Create an LLM caller using OpenAI API.

    Returns:
        LLMCaller function compatible with RLM.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def llm_caller(inputs: list[Input], context: dict[str, Any]) -> Output:
        """Call OpenAI API with RLM-compatible interface."""
        # Convert RLM Input format to OpenAI messages format
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in inputs]

        response = client.chat.completions.create(
            model=context.get("model", "gpt-4.1"),
            messages=messages,  # type: ignore[arg-type]
            temperature=context.get("temperature", 0.7),
            max_tokens=context.get("max_tokens", 2000),
        )

        return {
            "content": response.choices[0].message.content or "",
            "metadata": {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            },
        }

    return llm_caller


def main() -> None:
    """Run basic example: Analyze a company's market position."""
    # Create LLM caller
    llm_caller = create_openai_caller()

    # Initialize recursive engine
    engine = RecursiveEngine(
        llm=llm_caller,
        max_depth=15,  # High limit to handle over-decomposition
        max_steps=200,  # Allow many steps if needed
    )

    # Define a simple, direct task
    # Note: Adding explicit "do not break down" instructions reduces recursion
    task = """
    Write a 200-word summary about electric vehicles in 2026.

    Include:
    - Top 3 manufacturers and one key strength for each
    - 2 main industry challenges
    - Brief future outlook

    IMPORTANT: This is a simple writing task. Answer directly without breaking it into sub-tasks.
    Just write the summary in one go.
    """

    print("=" * 80)
    print("RLM Basic Example: Recursive Task Decomposition")
    print("=" * 80)
    print(f"\nTask: {task.strip()}\n")
    print("Starting execution...\n")

    try:
        # Solve the task recursively
        result = engine.solve(task)

        print("=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        print(result.get("content", ""))
        print("\n")

        # Print execution statistics
        print("=" * 80)
        print("EXECUTION STATISTICS")
        print("=" * 80)
        metadata = result.get("metadata", {})
        usage = metadata.get("usage", {})
        print(f"Total tokens used: {usage.get('total_tokens', 0)}")
        print(f"Recursion depth: Check trace for details")
        print("\nNote: The engine automatically decided whether to decompose the task")
        print("into sub-tasks (RECURSE) or solve it directly (EXECUTE).")

    except RecursionDepthError as e:
        print(f"\n{'='*80}")
        print("RECURSION DEPTH EXCEEDED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nThis means the LLM decomposed the task too deeply.")
        print("Solutions:")
        print("  1. Increase max_depth parameter (e.g., max_depth=20)")
        print("  2. Add explicit instruction: 'Answer directly without breaking into sub-tasks'")
        print("  3. Use simpler, more atomic tasks")
        print("  4. Try a different model if needed")
        print("\nNote: This is a safety feature to prevent infinite recursion.")

    except MaxStepsError as e:
        print(f"\n{'='*80}")
        print("MAX STEPS EXCEEDED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nThis means the execution took too many steps.")
        print("Solutions:")
        print("  1. Increase max_steps parameter (e.g., max_steps=200)")
        print("  2. Break the task into smaller sub-tasks")

    except Exception as e:
        print(f"\n{'='*80}")
        print("UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
