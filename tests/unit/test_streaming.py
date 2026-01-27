"""Unit tests for streaming event protocol.

Tests cover:
- StreamEvent dataclass validation
- Event type enforcement
- Required metadata fields
- JSON serialization/deserialization
- Factory methods for each event type
- Timestamp auto-generation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from rlm.streaming import StreamEvent


class TestStreamEventValidation:
    """Test StreamEvent dataclass validation."""

    def test_valid_stream_event(self) -> None:
        """Test creating valid StreamEvent."""
        event = StreamEvent(
            type="token",
            data={"content": "Hello"},
            metadata={"task_id": "task-1", "depth": 0},
        )

        assert event.type == "token"
        assert event.data == {"content": "Hello"}
        assert event.metadata["task_id"] == "task-1"
        assert event.metadata["depth"] == 0
        assert "timestamp" in event.metadata

    def test_invalid_event_type_fails(self) -> None:
        """Test that invalid event type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid event type"):
            StreamEvent(
                type="invalid",  # type: ignore
                data={},
                metadata={"task_id": "task-1", "depth": 0},
            )

    def test_missing_task_id_fails(self) -> None:
        """Test that missing task_id in metadata raises ValueError."""
        with pytest.raises(ValueError, match="task_id"):
            StreamEvent(
                type="token",
                data={},
                metadata={"depth": 0},  # Missing task_id
            )

    def test_missing_depth_fails(self) -> None:
        """Test that missing depth in metadata raises ValueError."""
        with pytest.raises(ValueError, match="depth"):
            StreamEvent(
                type="token",
                data={},
                metadata={"task_id": "task-1"},  # Missing depth
            )

    def test_timestamp_auto_generated(self) -> None:
        """Test that timestamp is automatically added if not provided."""
        event = StreamEvent(
            type="token",
            data={"content": "test"},
            metadata={"task_id": "task-1", "depth": 0},
        )

        assert "timestamp" in event.metadata
        # Verify timestamp is valid ISO 8601 format
        timestamp = event.metadata["timestamp"]
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_timestamp_preserved_if_provided(self) -> None:
        """Test that provided timestamp is not overwritten."""
        custom_timestamp = "2026-01-01T00:00:00Z"
        event = StreamEvent(
            type="token",
            data={},
            metadata={"task_id": "task-1", "depth": 0, "timestamp": custom_timestamp},
        )

        assert event.metadata["timestamp"] == custom_timestamp


class TestStreamEventSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_json(self) -> None:
        """Test StreamEvent serialization to JSON."""
        event = StreamEvent(
            type="token",
            data={"content": "Hello"},
            metadata={"task_id": "task-1", "depth": 0, "timestamp": "2026-01-27T12:00:00Z"},
        )

        json_str = event.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["type"] == "token"
        assert data["data"]["content"] == "Hello"
        assert data["metadata"]["task_id"] == "task-1"

    def test_from_json(self) -> None:
        """Test StreamEvent deserialization from JSON."""
        json_str = '{"type":"token","data":{"content":"Hi"},"metadata":{"task_id":"t1","depth":0}}'

        event = StreamEvent.from_json(json_str)

        assert event.type == "token"
        assert event.data["content"] == "Hi"
        assert event.metadata["task_id"] == "t1"
        assert event.metadata["depth"] == 0

    def test_from_json_invalid_fails(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid StreamEvent JSON"):
            StreamEvent.from_json("not valid json")

    def test_from_json_missing_fields_fails(self) -> None:
        """Test that JSON missing required fields raises ValueError."""
        with pytest.raises(ValueError, match="Invalid StreamEvent JSON"):
            StreamEvent.from_json('{"type":"token"}')  # Missing data

    def test_roundtrip_serialization(self) -> None:
        """Test that serialize -> deserialize preserves event."""
        original = StreamEvent(
            type="result",
            data={"content": "Task complete", "result_metadata": {"tokens": 150}},
            metadata={"task_id": "task-1", "depth": 1, "timestamp": "2026-01-27T12:00:00Z"},
        )

        json_str = original.to_json()
        restored = StreamEvent.from_json(json_str)

        assert restored.type == original.type
        assert restored.data == original.data
        assert restored.metadata["task_id"] == original.metadata["task_id"]


class TestStreamEventFactoryMethods:
    """Test factory methods for creating events."""

    def test_plan_event_decompose(self) -> None:
        """Test creating plan event with decompose decision."""
        event = StreamEvent.plan_event(
            decision="decompose",
            sub_tasks=["task1", "task2"],
            task_id="root",
            depth=0,
        )

        assert event.type == "plan"
        assert event.data["decision"] == "decompose"
        assert event.data["sub_tasks"] == ["task1", "task2"]
        assert event.metadata["task_id"] == "root"
        assert event.metadata["depth"] == 0

    def test_plan_event_execute(self) -> None:
        """Test creating plan event with execute decision."""
        event = StreamEvent.plan_event(
            decision="execute",
            sub_tasks=None,
            task_id="task-1",
            depth=1,
        )

        assert event.type == "plan"
        assert event.data["decision"] == "execute"
        assert event.data["sub_tasks"] is None
        assert event.metadata["depth"] == 1

    def test_token_event(self) -> None:
        """Test creating token event."""
        event = StreamEvent.token_event(
            content="Hello",
            task_id="task-1",
            depth=1,
        )

        assert event.type == "token"
        assert event.data["content"] == "Hello"
        assert event.metadata["task_id"] == "task-1"
        assert event.metadata["depth"] == 1

    def test_result_event(self) -> None:
        """Test creating result event."""
        event = StreamEvent.result_event(
            content="Task complete",
            metadata={"tokens": 150, "duration": 2.5},
            task_id="task-1",
            depth=1,
        )

        assert event.type == "result"
        assert event.data["content"] == "Task complete"
        assert event.data["result_metadata"]["tokens"] == 150
        assert event.metadata["task_id"] == "task-1"

    def test_error_event(self) -> None:
        """Test creating error event."""
        event = StreamEvent.error_event(
            error="Connection timeout",
            error_type="TimeoutError",
            task_id="task-1",
            depth=1,
        )

        assert event.type == "error"
        assert event.data["error"] == "Connection timeout"
        assert event.data["error_type"] == "TimeoutError"
        assert event.metadata["task_id"] == "task-1"
        assert event.metadata["depth"] == 1


class TestEventTypes:
    """Test different event type scenarios."""

    def test_all_event_types_valid(self) -> None:
        """Test that all event types are valid."""
        valid_types = ["plan", "token", "result", "error"]

        for event_type in valid_types:
            event = StreamEvent(
                type=event_type,  # type: ignore
                data={},
                metadata={"task_id": "task-1", "depth": 0},
            )
            assert event.type == event_type

    def test_event_metadata_extensible(self) -> None:
        """Test that metadata can include additional fields."""
        event = StreamEvent(
            type="token",
            data={"content": "test"},
            metadata={
                "task_id": "task-1",
                "depth": 0,
                "custom_field": "custom_value",
                "breadcrumbs": ("root", "task-1"),
            },
        )

        assert event.metadata["custom_field"] == "custom_value"
        assert event.metadata["breadcrumbs"] == ("root", "task-1")

    def test_event_data_flexible(self) -> None:
        """Test that data dict can contain various types."""
        event = StreamEvent(
            type="result",
            data={
                "content": "Result",
                "result_metadata": {
                    "tokens": 100,
                    "duration": 1.5,
                    "cached": True,
                    "sub_results": ["r1", "r2"],
                },
            },
            metadata={"task_id": "task-1", "depth": 0},
        )

        assert isinstance(event.data["result_metadata"]["tokens"], int)
        assert isinstance(event.data["result_metadata"]["duration"], float)
        assert isinstance(event.data["result_metadata"]["cached"], bool)
        assert isinstance(event.data["result_metadata"]["sub_results"], list)
