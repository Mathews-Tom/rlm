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
            if asyncio.iscoroutinefunction(tool.callable):
                result = await asyncio.wait_for(
                    tool.callable(arguments), timeout=timeout
                )
            else:
                # Run sync callable in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, tool.callable, arguments),
                    timeout=timeout,
                )

            logger.debug(f"Tool '{name}' completed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = f"Tool '{name}' execution timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)

        except Exception as e:
            error_msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            # Re-raise to allow caller to handle
            raise RuntimeError(error_msg) from e


__all__ = ["Tool", "ToolRegistry"]
