"""Example tool implementations demonstrating common use cases.

This module provides reference implementations of tools for:
- Web search (DuckDuckGo)
- Mathematical calculations
- File system operations (read/write)
- HTTP requests
- Current time retrieval

Each tool demonstrates:
- Comprehensive docstrings with examples
- Parameter validation
- Graceful error handling
- Type hints (dict[str, Any] -> str)

Example:
    >>> from rlm.tools import Tool, ToolRegistry
    >>> from rlm.tools.examples import create_calculator_tool
    >>>
    >>> calculator = create_calculator_tool()
    >>> registry = ToolRegistry()
    >>> registry.register(calculator)
    >>>
    >>> result = await registry.execute("calculator", {"expression": "2 + 2"})
    >>> print(result)
    "4"
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from rlm.tools import Tool


def create_search_web_tool() -> Tool:
    """Create web search tool using DuckDuckGo API.

    Returns:
        Tool instance for web search

    Example:
        >>> tool = create_search_web_tool()
        >>> result = tool.callable({"query": "Python programming", "max_results": 3})
        >>> data = json.loads(result)
        >>> print(len(data['results']))
        3
    """

    def search_web(params: dict[str, Any]) -> str:
        """Search the web using DuckDuckGo Instant Answer API.

        Args:
            params: Dict with:
                - query (str): Search query string
                - max_results (int): Maximum number of results (default: 5)

        Returns:
            JSON string with search results

        Raises:
            ValueError: If query is missing or empty
            RuntimeError: If API request fails
        """
        # Validate parameters
        query = params.get("query", "").strip()
        if not query:
            raise ValueError("Query parameter is required and cannot be empty")

        max_results = params.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")

        try:
            # Use DuckDuckGo Instant Answer API
            # Note: This is a simple example. For production, use requests library
            import urllib.request

            encoded_query = quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())

            # Extract relevant results
            results = []

            # Abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Result"),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:50] + "...",
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    })
                if len(results) >= max_results:
                    break

            return json.dumps({
                "query": query,
                "results": results,
                "count": len(results),
            }, indent=2)

        except Exception as e:
            raise RuntimeError(f"Web search failed: {type(e).__name__}: {e}") from e

    return Tool(
        name="search_web",
        description="Search the web for information using DuckDuckGo. Returns relevant results with titles, snippets, and URLs.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (e.g., 'Python async programming')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        callable=search_web,
    )


def create_calculator_tool() -> Tool:
    """Create safe mathematical expression evaluator tool.

    Returns:
        Tool instance for calculations

    Example:
        >>> tool = create_calculator_tool()
        >>> result = tool.callable({"expression": "(2 + 3) * 4"})
        >>> print(result)
        "20"
    """

    def calculator(params: dict[str, Any]) -> str:
        """Evaluate mathematical expressions safely.

        Args:
            params: Dict with:
                - expression (str): Math expression to evaluate

        Returns:
            String representation of result

        Raises:
            ValueError: If expression is invalid or contains forbidden operations
        """
        # Validate parameters
        expression = params.get("expression", "").strip()
        if not expression:
            raise ValueError("Expression parameter is required and cannot be empty")

        # Whitelist allowed characters (numbers, operators, parentheses, spaces)
        allowed_pattern = r'^[0-9+\-*/().%\s]+$'
        if not re.match(allowed_pattern, expression):
            raise ValueError(
                f"Expression contains forbidden characters. "
                f"Allowed: numbers, +, -, *, /, %, (, ), spaces. "
                f"Got: {expression!r}"
            )

        try:
            # Use eval with restricted globals/locals for safety
            # Only allow __builtins__ to be empty (no dangerous functions)
            result = eval(expression, {"__builtins__": {}}, {})

            # Ensure result is numeric
            if not isinstance(result, (int, float)):
                raise ValueError(f"Expression must evaluate to a number, got {type(result).__name__}")

            return str(result)

        except SyntaxError as e:
            raise ValueError(f"Invalid mathematical expression: {e}") from e
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Calculation error: {type(e).__name__}: {e}") from e

    return Tool(
        name="calculator",
        description="Evaluate mathematical expressions safely. Supports basic arithmetic operations (+, -, *, /, %, parentheses).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '(2 + 3) * 4')",
                },
            },
            "required": ["expression"],
        },
        callable=calculator,
    )


def create_file_read_tool(base_dir: str | None = None) -> Tool:
    """Create file reading tool with path validation.

    Args:
        base_dir: Base directory for file operations (default: current directory)
            All file paths must be within this directory for security.

    Returns:
        Tool instance for file reading

    Example:
        >>> tool = create_file_read_tool(base_dir="/tmp/sandbox")
        >>> result = tool.callable({"path": "data.txt"})
        >>> print(result)
        "File contents here..."
    """

    def file_read(params: dict[str, Any]) -> str:
        """Read file contents with path validation.

        Args:
            params: Dict with:
                - path (str): Relative path to file within base_dir

        Returns:
            File contents as string

        Raises:
            ValueError: If path is invalid or outside base_dir
            FileNotFoundError: If file doesn't exist
            RuntimeError: If read fails
        """
        # Validate parameters
        path_str = params.get("path", "").strip()
        if not path_str:
            raise ValueError("Path parameter is required and cannot be empty")

        # Resolve paths
        base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        file_path = (base / path_str).resolve()

        # Security check: ensure path is within base_dir
        if not file_path.is_relative_to(base):
            raise ValueError(
                f"Path '{path_str}' resolves outside base directory. "
                f"Only files within {base} are accessible."
            )

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # Read file contents
            content = file_path.read_text(encoding="utf-8")
            return json.dumps({
                "path": str(file_path.relative_to(base)),
                "size_bytes": len(content),
                "content": content,
            }, indent=2)

        except UnicodeDecodeError:
            raise RuntimeError(f"File is not valid UTF-8 text: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {type(e).__name__}: {e}") from e

    return Tool(
        name="file_read",
        description="Read contents of a text file. Path must be within configured base directory for security.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to file (e.g., 'data/input.txt')",
                },
            },
            "required": ["path"],
        },
        callable=file_read,
    )


def create_file_write_tool(base_dir: str | None = None) -> Tool:
    """Create file writing tool with safety checks.

    Args:
        base_dir: Base directory for file operations (default: current directory)
            All file paths must be within this directory for security.

    Returns:
        Tool instance for file writing

    Example:
        >>> tool = create_file_write_tool(base_dir="/tmp/sandbox")
        >>> result = tool.callable({"path": "output.txt", "content": "Hello, World!"})
        >>> data = json.loads(result)
        >>> print(data['status'])
        "success"
    """

    def file_write(params: dict[str, Any]) -> str:
        """Write content to file with safety checks.

        Args:
            params: Dict with:
                - path (str): Relative path to file within base_dir
                - content (str): Content to write

        Returns:
            JSON string with write status

        Raises:
            ValueError: If path is invalid or outside base_dir
            RuntimeError: If write fails
        """
        # Validate parameters
        path_str = params.get("path", "").strip()
        if not path_str:
            raise ValueError("Path parameter is required and cannot be empty")

        content = params.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Content parameter must be a string")

        # Resolve paths
        base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        file_path = (base / path_str).resolve()

        # Security check: ensure path is within base_dir
        if not file_path.is_relative_to(base):
            raise ValueError(
                f"Path '{path_str}' resolves outside base directory. "
                f"Only files within {base} are writable."
            )

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            file_path.write_text(content, encoding="utf-8")

            return json.dumps({
                "status": "success",
                "path": str(file_path.relative_to(base)),
                "bytes_written": len(content.encode("utf-8")),
            }, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to write file: {type(e).__name__}: {e}") from e

    return Tool(
        name="file_write",
        description="Write content to a text file. Creates parent directories if needed. Path must be within configured base directory for security.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to file (e.g., 'output/result.txt')",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to file",
                },
            },
            "required": ["path", "content"],
        },
        callable=file_write,
    )


def create_http_request_tool() -> Tool:
    """Create HTTP request tool for GET/POST operations.

    Returns:
        Tool instance for HTTP requests

    Example:
        >>> tool = create_http_request_tool()
        >>> result = tool.callable({
        ...     "url": "https://api.example.com/data",
        ...     "method": "GET"
        ... })
        >>> data = json.loads(result)
        >>> print(data['status_code'])
        200
    """

    def http_request(params: dict[str, Any]) -> str:
        """Make HTTP GET or POST request.

        Args:
            params: Dict with:
                - url (str): URL to request
                - method (str): HTTP method (GET or POST)
                - data (dict): Optional POST body data
                - headers (dict): Optional HTTP headers

        Returns:
            JSON string with response status, headers, and body

        Raises:
            ValueError: If URL or method is invalid
            RuntimeError: If request fails
        """
        # Validate parameters
        url = params.get("url", "").strip()
        if not url:
            raise ValueError("URL parameter is required and cannot be empty")

        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        method = params.get("method", "GET").upper()
        if method not in ("GET", "POST"):
            raise ValueError("Method must be GET or POST")

        post_data = params.get("data")
        headers = params.get("headers", {})

        try:
            import urllib.request

            # Prepare request
            if method == "POST" and post_data:
                if isinstance(post_data, dict):
                    body = json.dumps(post_data).encode("utf-8")
                    headers["Content-Type"] = "application/json"
                else:
                    body = str(post_data).encode("utf-8")

                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            else:
                req = urllib.request.Request(url, headers=headers, method="GET")

            # Execute request
            with urllib.request.urlopen(req, timeout=10) as response:
                response_body = response.read().decode("utf-8")
                response_headers = dict(response.headers)

                return json.dumps({
                    "status_code": response.status,
                    "headers": response_headers,
                    "body": response_body,
                }, indent=2)

        except Exception as e:
            raise RuntimeError(f"HTTP request failed: {type(e).__name__}: {e}") from e

    return Tool(
        name="http_request",
        description="Make HTTP GET or POST requests. Supports custom headers and JSON body for POST.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to request (must start with http:// or https://)",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method (GET or POST)",
                    "default": "GET",
                },
                "data": {
                    "type": "object",
                    "description": "Optional POST body data (JSON object)",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers",
                },
            },
            "required": ["url"],
        },
        callable=http_request,
    )


def create_current_time_tool() -> Tool:
    """Create current time retrieval tool.

    Returns:
        Tool instance for getting current time

    Example:
        >>> tool = create_current_time_tool()
        >>> result = tool.callable({"format": "iso"})
        >>> data = json.loads(result)
        >>> print(data['timezone'])
        "UTC"
    """

    def current_time(params: dict[str, Any]) -> str:
        """Get current date and time.

        Args:
            params: Dict with:
                - format (str): Output format ('iso' or 'unix'), default 'iso'
                - timezone (str): Timezone name, default 'UTC'

        Returns:
            JSON string with current time information

        Raises:
            ValueError: If format or timezone is invalid
        """
        # Validate parameters
        format_type = params.get("format", "iso").lower()
        if format_type not in ("iso", "unix"):
            raise ValueError("Format must be 'iso' or 'unix'")

        timezone = params.get("timezone", "UTC")
        # For simplicity, only support UTC in this example
        if timezone != "UTC":
            raise ValueError("Only UTC timezone is supported in this example")

        try:
            now = datetime.utcnow()

            result: dict[str, Any] = {
                "timezone": "UTC",
            }

            if format_type == "iso":
                result["datetime"] = now.isoformat() + "Z"
                result["date"] = now.date().isoformat()
                result["time"] = now.time().isoformat()
            else:  # unix
                result["timestamp"] = int(now.timestamp())

            return json.dumps(result, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to get current time: {type(e).__name__}: {e}") from e

    return Tool(
        name="current_time",
        description="Get current date and time. Supports ISO 8601 and Unix timestamp formats.",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "unix"],
                    "description": "Output format ('iso' for ISO 8601, 'unix' for Unix timestamp)",
                    "default": "iso",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (currently only 'UTC' supported)",
                    "default": "UTC",
                },
            },
            "required": [],
        },
        callable=current_time,
    )


__all__ = [
    "create_search_web_tool",
    "create_calculator_tool",
    "create_file_read_tool",
    "create_file_write_tool",
    "create_http_request_tool",
    "create_current_time_tool",
]
