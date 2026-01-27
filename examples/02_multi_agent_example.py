"""Multi-Agent Example - Specialized Agent Routing.

This example demonstrates RLM's multi-agent capabilities:
1. Define specialized agents (researcher, writer, critic)
2. Route sub-tasks to appropriate agents
3. Collaborate across agents to solve complex tasks

Each agent can have:
- Different system prompts/instructions
- Different models (e.g., GPT-4 for reasoning, GPT-3.5 for formatting)
- Specialized capabilities
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


def create_agent_caller(system_prompt: str) -> Any:
    """Create a specialized agent LLM caller.

    Args:
        system_prompt: System prompt defining agent's role

    Returns:
        LLMCaller function for this agent
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def llm_caller(inputs: list[Input], context: dict[str, Any]) -> Output:
        """Call OpenAI with agent-specific system prompt."""
        # Inject system prompt at the beginning
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in inputs])

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
    """Run multi-agent example: Write and critique an article."""
    # Define specialized agents
    researcher_prompt = """
    You are a research specialist. Your job is to:
    - Gather factual information
    - Find credible sources
    - Identify key data points and statistics
    - Provide comprehensive research summaries

    Focus on accuracy and thoroughness.
    """

    writer_prompt = """
    You are a professional writer. Your job is to:
    - Transform research into engaging content
    - Create clear, compelling narratives
    - Use appropriate tone and style
    - Structure information logically

    Focus on clarity and readability.
    """

    critic_prompt = """
    You are a critical editor. Your job is to:
    - Identify weaknesses in arguments
    - Spot factual errors or inconsistencies
    - Suggest improvements
    - Ensure logical flow

    Focus on constructive criticism and quality assurance.
    """

    # Create specialized agent callers
    researcher_caller = create_agent_caller(researcher_prompt)
    writer_caller = create_agent_caller(writer_prompt)
    critic_caller = create_agent_caller(critic_prompt)

    # Create planner/coordinator agent
    planner_caller = create_agent_caller(
        "You are a task coordinator. Break down complex tasks and assign them to specialists."
    )

    # Create agent registry (must include planner as it's the default router_model)
    agents = {
        "planner": planner_caller,  # Router agent for task decomposition
        "researcher": researcher_caller,
        "writer": writer_caller,
        "critic": critic_caller,
    }

    # Initialize engine with multi-agent support
    # The 'planner' agent will handle task decomposition and routing
    engine = RecursiveEngine(
        llm=planner_caller,  # Default/fallback LLM
        agents=agents,
        router_model="planner",  # Use planner agent for decisions
        max_depth=15,  # High limit for multi-agent coordination overhead
        max_steps=200,  # Allow many steps for agent collaboration
    )

    # Define a simple task with minimal steps
    # Note: Multi-agent tasks need very high max_depth due to coordination overhead
    task = """
    Write a 200-word article about AI in healthcare.

    Cover: diagnostic imaging, drug discovery, and patient monitoring.

    IMPORTANT: This is a straightforward writing task.
    Do not over-decompose. Execute as directly as possible.
    """

    print("=" * 80)
    print("RLM Multi-Agent Example: Specialized Agent Routing")
    print("=" * 80)
    print("\nAvailable Agents:")
    print("  - planner: Plans and delegates tasks (router agent)")
    print("  - researcher: Gathers factual information and data")
    print("  - writer: Creates engaging, well-structured content")
    print("  - critic: Reviews and provides constructive feedback\n")
    print(f"Task: {task.strip()}\n")
    print("Starting execution...\n")

    try:
        # Solve the task with multi-agent collaboration
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
        print(
            "\nNote: The planner automatically assigned sub-tasks to specialized agents."
        )
        print(
            "Check the trace to see which agents were used for each step of the process."
        )

    except RecursionDepthError as e:
        print(f"\n{'='*80}")
        print("RECURSION DEPTH EXCEEDED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nMulti-agent tasks have significant coordination overhead causing deep recursion.")
        print("Solutions:")
        print("  1. Increase max_depth significantly (e.g., max_depth=20)")
        print("  2. Simplify the task - remove workflow specifications")
        print("  3. Add 'Execute directly' instruction to discourage over-decomposition")
        print("  4. Consider using single-agent approach for simpler tasks")

    except MaxStepsError as e:
        print(f"\n{'='*80}")
        print("MAX STEPS EXCEEDED")
        print("=" * 80)
        print(f"Error: {e}")

    except Exception as e:
        print(f"\n{'='*80}")
        print("UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
