from __future__ import annotations

import uuid
from dataclasses import dataclass


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

    def store_named(self, name: str, content: str) -> None:
        """Store content under a specific named key.

        Args:
            name: Key to store content under (must start with '__')
            content: Content to store

        Raises:
            ValueError: If name does not start with '__'
        """
        if not name.startswith("__"):
            raise ValueError("Named keys must start with '__' to avoid collision with ref:: IDs")
        self._store[name] = content

    def resolve_named(self, name: str) -> str | None:
        """Retrieve content by named key, None if not found."""
        return self._store.get(name)

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
        active_agent: Name of agent executing current task (None = default LLM)

    Example:
        Root context:
        >>> memory = SharedMemory()
        >>> context = RLMContext(
        ...     task_id="abc123",
        ...     parent_id=None,
        ...     depth=0,
        ...     breadcrumbs=(),
        ...     memory_ref=memory,
        ...     active_agent=None
        ... )

        Child context with agent:
        >>> child = context.create_child("def456", "Research topic", active_agent="researcher")
        >>> child.depth
        1
        >>> child.breadcrumbs
        ('Research topic',)
        >>> child.active_agent
        'researcher'
    """

    task_id: str
    parent_id: str | None
    depth: int
    breadcrumbs: tuple[str, ...]
    memory_ref: SharedMemory
    active_agent: str | None = None

    def create_child(
        self,
        task_id: str,
        step_description: str,
        active_agent: str | None = None,
    ) -> RLMContext:
        """Create child context for recursive call.

        Args:
            task_id: Unique ID for child task
            step_description: Description of sub-task (added to breadcrumbs)
            active_agent: Name of agent for child task (None = inherit from parent)

        Returns:
            New RLMContext with incremented depth

        Example:
            >>> parent = RLMContext("parent", None, 0, (), memory, None)
            >>> child = parent.create_child("child", "Sub-task")
            >>> child.parent_id
            'parent'
            >>> child.depth
            1
            >>> child.breadcrumbs
            ('Sub-task',)
            >>> child.active_agent  # Inherited from parent
            None

            With agent assignment:
            >>> child2 = parent.create_child("child2", "Research", active_agent="researcher")
            >>> child2.active_agent
            'researcher'
        """
        return RLMContext(
            task_id=task_id,
            parent_id=self.task_id,
            depth=self.depth + 1,
            breadcrumbs=self.breadcrumbs + (step_description,),
            memory_ref=self.memory_ref,
            active_agent=active_agent if active_agent is not None else self.active_agent,
        )
