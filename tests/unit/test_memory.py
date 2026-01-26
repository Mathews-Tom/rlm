from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rlm.memory import RLMContext, SharedMemory


def test_shared_memory_store() -> None:
    """Test content storage and reference ID generation."""
    memory = SharedMemory()
    content = "Large document content"

    ref_id = memory.store(content)

    assert ref_id.startswith("ref::")
    assert len(ref_id) == 13  # "ref::" (5 chars) + 8 hex chars


def test_shared_memory_resolve() -> None:
    """Test content retrieval by reference ID."""
    memory = SharedMemory()
    content = "Test content"

    ref_id = memory.store(content)
    retrieved = memory.resolve(ref_id)

    assert retrieved == content


def test_shared_memory_missing_ref() -> None:
    """Test handling of missing reference ID."""
    memory = SharedMemory()

    result = memory.resolve("ref::invalid")
    assert result == ""


def test_shared_memory_multiple_items() -> None:
    """Test storing and resolving multiple items."""
    memory = SharedMemory()

    ref1 = memory.store("Content 1")
    ref2 = memory.store("Content 2")
    ref3 = memory.store("Content 3")

    assert ref1 != ref2 != ref3
    assert memory.resolve(ref1) == "Content 1"
    assert memory.resolve(ref2) == "Content 2"
    assert memory.resolve(ref3) == "Content 3"


def test_context_immutability() -> None:
    """Verify RLMContext is frozen and immutable."""
    memory = SharedMemory()
    context = RLMContext(
        task_id="abc",
        parent_id=None,
        depth=0,
        breadcrumbs=(),
        memory_ref=memory,
    )

    with pytest.raises(FrozenInstanceError):
        context.depth = 1  # type: ignore[misc]  # Should raise error


def test_context_create_child() -> None:
    """Test child context creation."""
    memory = SharedMemory()
    parent = RLMContext(
        task_id="parent",
        parent_id=None,
        depth=0,
        breadcrumbs=(),
        memory_ref=memory,
    )

    child = parent.create_child("child", "Sub-task")

    assert child.task_id == "child"
    assert child.parent_id == "parent"
    assert child.depth == 1
    assert child.breadcrumbs == ("Sub-task",)
    assert child.memory_ref is memory  # Same reference


def test_context_nested_children() -> None:
    """Test creating nested child contexts."""
    memory = SharedMemory()
    root = RLMContext(
        task_id="root",
        parent_id=None,
        depth=0,
        breadcrumbs=(),
        memory_ref=memory,
    )

    child1 = root.create_child("child1", "Step 1")
    child2 = child1.create_child("child2", "Step 2")
    child3 = child2.create_child("child3", "Step 3")

    assert child3.depth == 3
    assert child3.breadcrumbs == ("Step 1", "Step 2", "Step 3")
    assert child3.parent_id == "child2"
    assert child3.memory_ref is memory


def test_context_parent_unchanged() -> None:
    """Verify parent context remains unchanged when creating child."""
    memory = SharedMemory()
    parent = RLMContext(
        task_id="parent",
        parent_id=None,
        depth=0,
        breadcrumbs=(),
        memory_ref=memory,
    )

    # Create child
    child = parent.create_child("child", "Sub-task")

    # Parent should be unchanged
    assert parent.depth == 0
    assert parent.breadcrumbs == ()
    assert parent.task_id == "parent"
    assert parent.parent_id is None

    # Child should have updated values
    assert child.depth == 1
    assert child.breadcrumbs == ("Sub-task",)


def test_large_document_offloading() -> None:
    """Test memory offloading with large documents (50k+ characters).

    Verifies that SharedMemory can handle large content without issues,
    simulating variable offloading to prevent context overflow.
    """
    memory = SharedMemory()

    # Create 50k+ character document
    large_content = "x" * 50_000 + "\n" + "Large document test content" * 100

    # Store large document
    ref_id = memory.store(large_content)

    # Verify reference ID format
    assert ref_id.startswith("ref::")
    assert len(ref_id) == 13

    # Verify full content can be retrieved
    retrieved = memory.resolve(ref_id)
    assert retrieved == large_content
    assert len(retrieved) > 50_000

    # Verify multiple large documents can coexist
    large_content2 = "y" * 60_000
    ref_id2 = memory.store(large_content2)

    assert ref_id != ref_id2
    assert memory.resolve(ref_id) == large_content
    assert memory.resolve(ref_id2) == large_content2
