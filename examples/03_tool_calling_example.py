"""Tool Calling Example - LLMs Using External Tools.

This example demonstrates how to use OpenAI's native function calling
to enable LLMs to use tools during task execution.

Key concepts:
1. Define tools as Python functions
2. Create OpenAI function schemas
3. Let the LLM decide when to call tools
4. Execute tool calls and provide results back to the LLM
5. Iterate until the task is complete

This approach works reliably with OpenAI's function calling API.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


# Define tool functions
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: Timezone identifier (e.g., 'UTC', 'America/New_York')

    Returns:
        Current time as ISO format string
    """
    # Simplified: just return UTC time
    # In production, use pytz for real timezone conversion
    current_time = datetime.now()
    return f"{current_time.isoformat()} ({timezone})"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2')

    Returns:
        Result of the calculation as string

    Raises:
        ValueError: If expression is invalid
    """
    try:
        # WARNING: eval() is unsafe in production! Use a proper parser
        # This is just for demonstration
        result = eval(expression, {"__builtins__": {}}, {})
        return str(float(result))
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information.

    Args:
        query: Search query

    Returns:
        Summary of Wikipedia article (simulated)
    """
    # Simulated Wikipedia search
    # In production, use the Wikipedia API
    return f"Wikipedia summary for '{query}': Electric vehicles (EVs) are automobiles powered by electricity rather than gasoline. Major manufacturers include Tesla, BYD, and Volkswagen. EVs produce zero direct emissions and are increasingly popular due to environmental concerns and government incentives."


# OpenAI function schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a specific timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone identifier (e.g., 'UTC', 'America/New_York')",
                        "default": "UTC",
                    }
                },
                "required": [],
            },
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
                        "description": "Mathematical expression to evaluate (e.g., '15 * 8 + 100 / 4')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for information",
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


# Map function names to actual functions
FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_wikipedia": search_wikipedia,
}


async def solve_with_tools(task: str, max_iterations: int = 10) -> str:
    """Solve a task using tools via OpenAI function calling.

    Args:
        task: Task description
        max_iterations: Maximum conversation iterations

    Returns:
        Final result from the LLM
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task}
    ]

    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*60}")

        # Call OpenAI with function calling enabled
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # Let the model decide when to use tools
        )

        message = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        # Check if the model wants to call functions
        if message.tool_calls:
            print(f"\n🔧 Tool calls requested: {len(message.tool_calls)}")

            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  - {function_name}({function_args})")

                # Execute the function
                try:
                    function_to_call = FUNCTION_MAP[function_name]
                    function_response = function_to_call(**function_args)
                    print(f"    Result: {function_response}")

                    # Add function response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_response,
                    })
                except Exception as e:
                    print(f"    Error: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error executing {function_name}: {str(e)}",
                    })

            # Continue the conversation with tool results
            continue

        # No tool calls - the model has provided a final answer
        if message.content:
            print("\n✅ Final answer received")
            return message.content

    raise RuntimeError(f"Exceeded maximum iterations ({max_iterations})")


async def main() -> None:
    """Run tool calling example."""
    print("=" * 80)
    print("RLM Tool Calling Example: Using OpenAI Function Calling")
    print("=" * 80)
    print("\nThis example demonstrates working tool calling using OpenAI's")
    print("native function calling API (not ToolCallingEngine).\n")

    print("Available Tools:")
    for tool in TOOLS:
        func = tool["function"]
        print(f"  - {func['name']}: {func['description']}")

    # Example 1: Simple calculation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Mathematical Calculation")
    print("=" * 80)

    task1 = "What is 15 * 8 + 100 / 4? Use the calculate tool to find the answer."

    print(f"\nTask: {task1}")

    try:
        result1 = await solve_with_tools(task1, max_iterations=5)
        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(result1)
    except Exception as e:
        print(f"\nError: {e}")

    # Example 2: Information lookup with tool
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Information Lookup")
    print("=" * 80)

    task2 = "Search Wikipedia for information about electric vehicles and tell me the top 3 manufacturers mentioned."

    print(f"\nTask: {task2}")

    try:
        result2 = await solve_with_tools(task2, max_iterations=5)
        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(result2)
    except Exception as e:
        print(f"\nError: {e}")

    # Example 3: Multiple tool calls
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Multiple Tool Calls")
    print("=" * 80)

    task3 = "What is the current time, and what is 50 * 42?"

    print(f"\nTask: {task3}")

    try:
        result3 = await solve_with_tools(task3, max_iterations=5)
        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(result3)
    except Exception as e:
        print(f"\nError: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  ✓ OpenAI's native function calling API")
    print("  ✓ Automatic tool selection by the LLM")
    print("  ✓ Tool execution and result handling")
    print("  ✓ Multi-turn conversation with tools")
    print("\nFor production:")
    print("  - Add proper error handling and retries")
    print("  - Implement tool authentication/authorization")
    print("  - Add logging and monitoring")
    print("  - Use async for concurrent tool calls")


if __name__ == "__main__":
    asyncio.run(main())
