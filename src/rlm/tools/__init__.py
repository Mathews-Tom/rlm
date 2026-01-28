"""Tool calling framework for RLM.

This module provides the infrastructure for agents to call external tools/functions
during execution, enabling agentic workflows beyond pure text generation.

Example:
    >>> def search_web(query: str, limit: int = 5) -> str:
    ...     '''Search the web for information.'''
    ...     results = api.search(query, limit=limit)
    ...     return json.dumps(results)
    ...
    >>> tool = Tool(
    ...     name="search_web",
    ...     description="Search the web for information",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {
    ...             "query": {"type": "string", "description": "Search query"},
    ...             "limit": {"type": "integer", "default": 5, "description": "Max results"}
    ...         },
    ...         "required": ["query"]
    ...     },
    ...     callable=search_web
    ... )
    ...
    >>> registry = ToolRegistry()
    >>> registry.register(tool)
    >>> result = await registry.execute("search_web", {"query": "AI news"})
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from rlm.async_engine import AsyncRecursiveEngine
from rlm.exceptions import ExecutionError
from rlm.memory import RLMContext
from rlm.types import AsyncLLMCaller, Input, Output, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """External tool/function that agents can call.

    Represents a callable external function with JSON Schema parameter validation.
    Tools enable agents to access external APIs, databases, file systems, and more.

    Attributes:
        name: Unique tool identifier (e.g., "search_web", "calculator")
        description: Human-readable purpose of the tool (shown to LLM)
        parameters: JSON Schema dict describing expected parameters
        callable: Sync or async function to execute (dict[str, Any] -> str)

    Example:
        >>> def get_weather(location: str, units: str = "celsius") -> str:
        ...     '''Get current weather for a location.'''
        ...     data = weather_api.fetch(location, units)
        ...     return json.dumps({"temp": data.temperature, "conditions": data.sky})
        ...
        >>> weather_tool = Tool(
        ...     name="get_weather",
        ...     description="Get current weather conditions for a location",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {"type": "string", "description": "City name or coordinates"},
        ...             "units": {
        ...                 "type": "string",
        ...                 "enum": ["celsius", "fahrenheit"],
        ...                 "default": "celsius",
        ...                 "description": "Temperature units"
        ...             }
        ...         },
        ...         "required": ["location"]
        ...     },
        ...     callable=get_weather
        ... )
    """

    name: str
    description: str
    parameters: dict[str, Any]
    callable: Callable[[dict[str, Any]], str]

    def __post_init__(self) -> None:
        """Validate tool configuration after initialization.

        Raises:
            ValueError: If name is empty or parameters is not a valid JSON Schema object
        """
        if not self.name:
            raise ValueError("Tool name cannot be empty")

        if not isinstance(self.parameters, dict):
            raise ValueError(f"Tool '{self.name}' parameters must be a dict (JSON Schema)")

        if self.parameters.get("type") != "object":
            raise ValueError(
                f"Tool '{self.name}' parameters must have type='object' at root level"
            )


class ToolRegistry:
    """Registry for managing available tools.

    Provides centralized tool registration, validation, and execution.
    Enforces name uniqueness and handles tool execution errors gracefully.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(search_tool)
        >>> registry.register(calculator_tool)
        >>>
        >>> # List available tools
        >>> tools = registry.list_tools()
        >>> print([t.name for t in tools])
        ['search_web', 'calculator']
        >>>
        >>> # Execute tool
        >>> result = await registry.execute("search_web", {"query": "Python"})
        >>> print(result)
        '{"results": [...], "count": 10}'
    """

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' already registered. "
                f"Use a unique name or unregister the existing tool first."
            )

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> Tool | None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to unregister

        Returns:
            Removed tool if found, None otherwise
        """
        tool = self._tools.pop(name, None)
        if tool:
            logger.info(f"Unregistered tool: {name}")
        else:
            logger.warning(f"Attempted to unregister unknown tool: {name}")
        return tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name.

        Args:
            name: Tool name to retrieve

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Get list of all registered tools.

        Returns:
            List of Tool instances sorted by name
        """
        return sorted(self._tools.values(), key=lambda t: t.name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get LLM-compatible tool schemas for all registered tools.

        Returns schemas in OpenAI function calling format.

        Returns:
            List of dicts with name, description, parameters for each tool

        Example:
            >>> schemas = registry.get_tool_schemas()
            >>> print(schemas[0])
            {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.list_tools()
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> str:
        """Execute a tool with given arguments.

        Handles both sync and async tool callables. Enforces timeout
        and provides graceful error handling.

        Args:
            name: Tool name to execute
            arguments: Dict of parameter values (must match tool's JSON Schema)
            timeout: Maximum execution time in seconds (default: 30.0)

        Returns:
            Tool result as string (typically JSON-serialized)

        Raises:
            TimeoutError: If tool execution exceeds timeout
            ValueError: If tool not found or arguments invalid

        Example:
            >>> result = await registry.execute(
            ...     "search_web",
            ...     {"query": "Python tutorials", "limit": 5}
            ... )
            >>> data = json.loads(result)
            >>> print(f"Found {len(data['results'])} results")
            Found 5 results
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(
                f"Tool '{name}' not found. "
                f"Available tools: {', '.join(self._tools.keys())}"
            )

        try:
            # Execute tool with timeout
            logger.debug(f"Executing tool '{name}' with arguments: {arguments}")

            # Check if callable is async
            result_str: str
            if asyncio.iscoroutinefunction(tool.callable):
                result_str = await asyncio.wait_for(
                    tool.callable(arguments), timeout=timeout
                )
            else:
                # Run sync callable in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result_str = await asyncio.wait_for(
                    loop.run_in_executor(None, tool.callable, arguments),
                    timeout=timeout,
                )

            logger.debug(f"Tool '{name}' completed successfully")
            return result_str

        except asyncio.TimeoutError:
            error_msg = f"Tool '{name}' execution timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)

        except Exception as e:
            error_msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


class ToolCallingEngine(AsyncRecursiveEngine):
    """Async recursive engine with tool calling support.

    Extends AsyncRecursiveEngine to enable LLM tool calling workflows.
    When the LLM returns tool_calls in its response, this engine executes
    the requested tools and injects results back into the conversation.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(search_tool)
        >>> registry.register(calculator_tool)
        >>>
        >>> async def my_llm(inputs, context):
        ...     # Your LLM that supports tool calling
        ...     return {
        ...         "content": "",
        ...         "metadata": {},
        ...         "tool_calls": [{"name": "search", "arguments": {"query": "AI news"}}]
        ...     }
        >>>
        >>> engine = ToolCallingEngine(
        ...     llm=my_llm,
        ...     tool_registry=registry,
        ...     max_depth=3,
        ...     verbose=True
        ... )
        >>> result = await engine.solve("Find latest AI news")
    """

    def __init__(
        self,
        llm: AsyncLLMCaller,
        tool_registry: ToolRegistry,
        agents: dict[str, AsyncLLMCaller] | None = None,
        router_model: str = "planner",
        max_depth: int = 3,
        max_steps: int = 100,
        max_concurrency: int = 10,
        tool_timeout: float = 30.0,
        max_tool_retries: int = 2,
        max_tool_iterations: int = 5,
        verbose: bool = False,
    ) -> None:
        """Initialize tool calling engine.

        Args:
            llm: Async LLM caller that supports tool calling
            tool_registry: ToolRegistry instance with registered tools
            agents: Optional multi-agent registry
            router_model: Agent name for planning decisions
            max_depth: Maximum recursion depth
            max_steps: Maximum total execution steps
            max_concurrency: Max concurrent tasks
            tool_timeout: Timeout for tool execution in seconds (default: 30.0)
            max_tool_retries: Max retries for transient tool failures (default: 2)
            max_tool_iterations: Max iterations in tool calling loop (default: 5)
                Prevents infinite loops when LLM continuously requests tools
            verbose: Enable debug logging
        """
        super().__init__(
            llm=llm,
            agents=agents,
            router_model=router_model,
            max_depth=max_depth,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.tool_registry = tool_registry
        self.tool_timeout = tool_timeout
        self.max_tool_retries = max_tool_retries
        self.max_tool_iterations = max_tool_iterations

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[tuple[str, str]]:
        """Execute list of tool calls and return results.

        Executes tools in sequence (not parallel) to maintain deterministic
        ordering and handle dependencies between tool calls.

        Args:
            tool_calls: List of ToolCall dicts with name and arguments

        Returns:
            List of (tool_name, result_string) tuples

        Example:
            >>> tool_calls = [
            ...     {"name": "search", "arguments": {"query": "Python"}},
            ...     {"name": "calculator", "arguments": {"expr": "2+2"}}
            ... ]
            >>> results = await engine._execute_tool_calls(tool_calls)
            >>> results
            [("search", '{"results": [...]}'), ("calculator", "4")]
        """
        results: list[tuple[str, str]] = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]

            if self.verbose:
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")

            # Check if tool exists
            tool = self.tool_registry.get(tool_name)
            if tool is None:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(t.name for t in self.tool_registry.list_tools())}"
                )
                logger.warning(error_msg)
                results.append((tool_name, f"ERROR: {error_msg}"))
                continue

            # Execute tool with retries
            for attempt in range(self.max_tool_retries + 1):
                try:
                    result = await self.tool_registry.execute(
                        tool_name,
                        arguments,
                        timeout=self.tool_timeout,
                    )
                    results.append((tool_name, result))
                    if self.verbose:
                        logger.info(
                            f"Tool '{tool_name}' completed successfully on attempt {attempt + 1}"
                        )
                    break  # Success

                except TimeoutError:
                    # Timeout errors are not retried
                    error_msg = f"Tool '{tool_name}' timed out after {self.tool_timeout}s"
                    logger.error(error_msg)
                    results.append((tool_name, f"ERROR: {error_msg}"))
                    break

                except Exception as e:
                    if attempt < self.max_tool_retries:
                        # Retry on transient failures
                        if self.verbose:
                            logger.warning(
                                f"Tool '{tool_name}' failed on attempt {attempt + 1}/{self.max_tool_retries + 1}: {e}"
                            )
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed
                        error_msg = f"Tool '{tool_name}' failed after {self.max_tool_retries + 1} attempts: {type(e).__name__}: {e}"
                        logger.error(error_msg, exc_info=True)
                        results.append((tool_name, f"ERROR: {error_msg}"))

        return results

    async def _execute_leaf_async(
        self, task: str, context: RLMContext
    ) -> Output:
        """Execute leaf task with iterative tool calling support.

        Overrides parent AsyncRecursiveEngine._execute_leaf_async to add
        tool calling support. If the LLM returns tool_calls, execute them
        and inject results back into the conversation until LLM returns
        a final answer without tool calls.

        Args:
            task: Task description
            context: Current execution context

        Returns:
            Final Output after tool calling loop completes

        Raises:
            ExecutionError: If LLM call fails
        """
        if self.verbose:
            logger.info(f"[Depth {context.depth}] Executing leaf task with tools: {task}")

        # Get the agent for this task
        agent = self.agents.get(context.active_agent or "default", self.llm)

        # Build conversation history
        inputs: list[Input] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use tools when needed to answer questions.",
            },
            {"role": "user", "content": task},
        ]

        # Iterative tool calling loop
        # Supports "read-before-write" pattern where tool can call another tool
        # by having the LLM request multiple tool calls in sequence
        result: Output | None = None

        for iteration in range(self.max_tool_iterations):
            if self.verbose:
                logger.info(
                    f"[Depth {context.depth}] Tool calling iteration {iteration + 1}/{self.max_tool_iterations}"
                )

            # Call LLM
            try:
                async with self._semaphore:
                    result = await agent(inputs, {"mode": "worker"})
            except Exception as e:
                logger.error(f"LLM call failed: {e}", exc_info=True)
                raise ExecutionError(f"LLM call failed for task: {task!r}") from e

            # Check if LLM requested tool calls
            tool_calls = result.get("tool_calls", [])
            if not tool_calls:
                # No tool calls - return final answer
                result["metadata"]["depth"] = context.depth
                result["metadata"]["task_id"] = context.task_id
                result["metadata"]["tool_iterations"] = iteration
                return result

            # Execute requested tools
            tool_results = await self._execute_tool_calls(tool_calls)

            # Inject tool results into conversation
            # Add assistant's tool call request
            inputs.append({
                "role": "assistant",
                "content": result.get("content", ""),  # May be empty for pure tool calls
            })

            # Add tool results as user messages
            tool_results_text = "\n\n".join(
                f"Tool: {name}\nResult: {result_str}"
                for name, result_str in tool_results
            )
            inputs.append({
                "role": "user",
                "content": f"Tool results:\n{tool_results_text}\n\nPlease use these results to answer the original question.",
            })

        # Max iterations reached - return last response
        if result is None:
            raise ExecutionError(f"Tool calling loop failed to produce result for task: {task!r}")

        logger.warning(
            f"Max tool calling iterations ({self.max_tool_iterations}) reached for task: {task}"
        )
        result["metadata"]["depth"] = context.depth
        result["metadata"]["task_id"] = context.task_id
        result["metadata"]["tool_iterations"] = self.max_tool_iterations
        result["metadata"]["max_iterations_reached"] = True
        return result


__all__ = ["Tool", "ToolRegistry", "ToolCallingEngine"]
