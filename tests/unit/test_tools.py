"""Unit tests for Tool dataclass and ToolRegistry.

Tests cover:
- Tool dataclass validation
- ToolRegistry operations (register, unregister, get, list_tools)
- Tool schema generation
- Tool execution (success and failure cases)
- Error handling, retries, timeouts
- Unknown tool handling
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any

from rlm.tools import Tool, ToolRegistry


class TestToolDataclass:
    """Test Tool dataclass validation."""

    def test_tool_creation_valid(self) -> None:
        """Test creating a valid Tool instance."""
        def my_tool(params: dict[str, Any]) -> str:
            return "result"

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"param": {"type": "string"}},
                "required": ["param"],
            },
            callable=my_tool,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.callable == my_tool
        assert tool.parameters["type"] == "object"

    def test_tool_empty_name_fails(self) -> None:
        """Test that empty tool name raises ValueError."""
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            Tool(
                name="",
                description="Test",
                parameters={"type": "object", "properties": {}},
                callable=lambda x: "result",
            )

    def test_tool_invalid_parameters_type_fails(self) -> None:
        """Test that non-dict parameters raise ValueError."""
        with pytest.raises(ValueError, match="parameters must be a dict"):
            Tool(
                name="test",
                description="Test",
                parameters="invalid",  # type: ignore
                callable=lambda x: "result",
            )

    def test_tool_missing_object_type_fails(self) -> None:
        """Test that parameters without type='object' raise ValueError."""
        with pytest.raises(ValueError, match="must have type='object'"):
            Tool(
                name="test",
                description="Test",
                parameters={"type": "string"},  # Wrong type
                callable=lambda x: "result",
            )


class TestToolRegistry:
    """Test ToolRegistry operations."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create empty tool registry."""
        return ToolRegistry()

    @pytest.fixture
    def sample_tool(self) -> Tool:
        """Create sample tool for testing."""
        def echo(params: dict[str, Any]) -> str:
            return params.get("message", "")

        return Tool(
            name="echo",
            description="Echo the message",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            callable=echo,
        )

    def test_register_tool(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test registering a tool."""
        registry.register(sample_tool)
        assert registry.get("echo") == sample_tool

    def test_register_duplicate_tool_fails(
        self, registry: ToolRegistry, sample_tool: Tool
    ) -> None:
        """Test that registering duplicate tool name raises ValueError."""
        registry.register(sample_tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(sample_tool)

    def test_unregister_tool(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test unregistering a tool."""
        registry.register(sample_tool)
        removed = registry.unregister("echo")

        assert removed == sample_tool
        assert registry.get("echo") is None

    def test_unregister_unknown_tool(self, registry: ToolRegistry) -> None:
        """Test unregistering unknown tool returns None."""
        result = registry.unregister("nonexistent")
        assert result is None

    def test_get_tool(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test getting tool by name."""
        registry.register(sample_tool)
        retrieved = registry.get("echo")

        assert retrieved == sample_tool

    def test_get_unknown_tool(self, registry: ToolRegistry) -> None:
        """Test getting unknown tool returns None."""
        result = registry.get("nonexistent")
        assert result is None

    def test_list_tools(self, registry: ToolRegistry) -> None:
        """Test listing all registered tools."""
        tool1 = Tool("tool1", "First", {"type": "object", "properties": {}}, lambda x: "1")
        tool2 = Tool("tool2", "Second", {"type": "object", "properties": {}}, lambda x: "2")
        tool3 = Tool("tool3", "Third", {"type": "object", "properties": {}}, lambda x: "3")

        registry.register(tool2)
        registry.register(tool1)
        registry.register(tool3)

        tools = registry.list_tools()

        # Should be sorted by name
        assert len(tools) == 3
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"
        assert tools[2].name == "tool3"

    def test_get_tool_schemas(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test getting LLM-compatible tool schemas."""
        registry.register(sample_tool)
        schemas = registry.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"
        assert schemas[0]["description"] == "Echo the message"
        assert schemas[0]["parameters"]["type"] == "object"


class TestToolExecution:
    """Test tool execution with ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create tool registry with test tools."""
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self, registry: ToolRegistry) -> None:
        """Test executing synchronous tool."""
        def add_numbers(params: dict[str, Any]) -> str:
            a = params["a"]
            b = params["b"]
            return str(a + b)

        tool = Tool(
            "add",
            "Add two numbers",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            add_numbers,
        )

        registry.register(tool)
        result = await registry.execute("add", {"a": 5, "b": 3})

        assert result == "8"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self, registry: ToolRegistry) -> None:
        """Test executing async tool."""
        async def async_echo(params: dict[str, Any]) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return params["message"]

        tool = Tool(
            "async_echo",
            "Async echo",
            {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            async_echo,
        )

        registry.register(tool)
        result = await registry.execute("async_echo", {"message": "hello"})

        assert result == "hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_fails(self, registry: ToolRegistry) -> None:
        """Test executing unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool 'unknown' not found"):
            await registry.execute("unknown", {})

    @pytest.mark.asyncio
    async def test_execute_timeout(self, registry: ToolRegistry) -> None:
        """Test tool execution timeout."""
        async def slow_tool(params: dict[str, Any]) -> str:
            await asyncio.sleep(10)  # Longer than timeout
            return "done"

        tool = Tool(
            "slow",
            "Slow tool",
            {"type": "object", "properties": {}},
            slow_tool,
        )

        registry.register(tool)

        with pytest.raises(TimeoutError, match="timed out"):
            await registry.execute("slow", {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, registry: ToolRegistry) -> None:
        """Test tool execution error handling."""
        def failing_tool(params: dict[str, Any]) -> str:
            raise RuntimeError("Tool failed intentionally")

        tool = Tool(
            "failing",
            "Failing tool",
            {"type": "object", "properties": {}},
            failing_tool,
        )

        registry.register(tool)

        with pytest.raises(RuntimeError, match="Tool 'failing' failed"):
            await registry.execute("failing", {})


class TestMockTools:
    """Test with mock tools for deterministic testing."""

    @pytest.fixture
    def mock_success_tool(self) -> Tool:
        """Mock tool that always succeeds."""
        def success(params: dict[str, Any]) -> str:
            return "success"

        return Tool(
            "mock_success",
            "Always succeeds",
            {"type": "object", "properties": {}},
            success,
        )

    @pytest.fixture
    def mock_failure_tool(self) -> Tool:
        """Mock tool that always fails."""
        def failure(params: dict[str, Any]) -> str:
            raise ValueError("Mock failure")

        return Tool(
            "mock_failure",
            "Always fails",
            {"type": "object", "properties": {}},
            failure,
        )

    @pytest.fixture
    def mock_counter_tool(self) -> tuple[Tool, list[int]]:
        """Mock tool that counts invocations."""
        call_count = [0]

        def counter(params: dict[str, Any]) -> str:
            call_count[0] += 1
            return str(call_count[0])

        tool = Tool(
            "mock_counter",
            "Counts calls",
            {"type": "object", "properties": {}},
            counter,
        )

        return tool, call_count

    @pytest.mark.asyncio
    async def test_mock_success_tool(
        self, registry: ToolRegistry, mock_success_tool: Tool
    ) -> None:
        """Test mock tool that always succeeds."""
        registry = ToolRegistry()
        registry.register(mock_success_tool)

        result = await registry.execute("mock_success", {})
        assert result == "success"

    @pytest.mark.asyncio
    async def test_mock_failure_tool(
        self, registry: ToolRegistry, mock_failure_tool: Tool
    ) -> None:
        """Test mock tool that always fails."""
        registry = ToolRegistry()
        registry.register(mock_failure_tool)

        with pytest.raises(RuntimeError, match="Mock failure"):
            await registry.execute("mock_failure", {})

    @pytest.mark.asyncio
    async def test_mock_counter_tool(self, mock_counter_tool: tuple[Tool, list[int]]) -> None:
        """Test mock tool that tracks call count."""
        tool, call_count = mock_counter_tool
        registry = ToolRegistry()
        registry.register(tool)

        # First call
        result1 = await registry.execute("mock_counter", {})
        assert result1 == "1"
        assert call_count[0] == 1

        # Second call
        result2 = await registry.execute("mock_counter", {})
        assert result2 == "2"
        assert call_count[0] == 2


@pytest.fixture
def registry() -> ToolRegistry:
    """Shared fixture for tool registry."""
    return ToolRegistry()
