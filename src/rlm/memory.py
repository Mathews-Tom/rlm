from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SharedMemory:
    """Shared memory store for variable offloading.

    Stores large content blobs and passes by reference to prevent
    context overflow. Uses prefix pattern "ref::{uuid}" for reference IDs.
    """

    def __init__(self) -> None:
        """Initialize empty memory store."""
        self._store: dict[str, str] = {}

    def store(self, content: str) -> str:
        """Store content and return reference ID.

        Args:
            content: Content to store (typically large documents)

        Returns:
            Reference ID with format "ref::{uuid}"

        Example:
            >>> memory = SharedMemory()
            >>> ref_id = memory.store("Large document...")
            >>> ref_id
            'ref::abc12345'
        """
        doc_id = f"ref::{uuid.uuid4().hex[:8]}"
        self._store[doc_id] = content
        return doc_id

    def resolve(self, doc_id: str) -> str:
        """Retrieve content by reference ID.

        Args:
            doc_id: Reference ID returned from store()

        Returns:
            Stored content, or empty string if not found

        Example:
            >>> memory = SharedMemory()
            >>> ref_id = memory.store("Test content")
            >>> memory.resolve(ref_id)
            'Test content'
            >>> memory.resolve("ref::invalid")
            ''
        """
        return self._store.get(doc_id, "")


@dataclass(frozen=True)
class RLMContext:
    """Immutable execution context for recursion tree.

    Tracks state through recursion levels. Each level creates
    a new context with updated depth and breadcrumbs.

    Attributes:
        task_id: Unique identifier for current task (UUID)
        parent_id: ID of parent task (None for root)
        depth: Current recursion depth (0 = root)
        breadcrumbs: Path from root to current node (task descriptions)
        memory_ref: Shared memory store for variable offloading

    Example:
        Root context:
        >>> memory = SharedMemory()
        >>> context = RLMContext(
        ...     task_id="abc123",
        ...     parent_id=None,
        ...     depth=0,
        ...     breadcrumbs=(),
        ...     memory_ref=memory
        ... )

        Child context:
        >>> child = context.create_child("def456", "Research topic")
        >>> child.depth
        1
        >>> child.breadcrumbs
        ('Research topic',)
    """

    task_id: str
    parent_id: str | None
    depth: int
    breadcrumbs: tuple[str, ...]
    memory_ref: SharedMemory

    def create_child(self, task_id: str, step_description: str) -> RLMContext:
        """Create child context for recursive call.

        Args:
            task_id: Unique ID for child task
            step_description: Description of sub-task (added to breadcrumbs)

        Returns:
            New RLMContext with incremented depth

        Example:
            >>> parent = RLMContext("parent", None, 0, (), memory)
            >>> child = parent.create_child("child", "Sub-task")
            >>> child.parent_id
            'parent'
            >>> child.depth
            1
            >>> child.breadcrumbs
            ('Sub-task',)
        """
        return RLMContext(
            task_id=task_id,
            parent_id=self.task_id,
            depth=self.depth + 1,
            breadcrumbs=self.breadcrumbs + (step_description,),
            memory_ref=self.memory_ref,
        )
