"""Checkpoint Example - Fault Tolerance and Recovery.

This example demonstrates how to implement checkpointing with direct LLM calls
for fault tolerance and recovery from failures.

Key concepts:
1. Custom checkpoint store for saving execution state
2. Periodic checkpoint creation during long-running tasks
3. Simulated failure to demonstrate recovery
4. Resume execution from last successful checkpoint

This approach works with direct LLM calls for maximum simplicity.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


@dataclass
class Checkpoint:
    """Checkpoint data for resuming execution."""

    checkpoint_id: str
    execution_id: str
    task: str
    step: int
    partial_result: str | None
    timestamp: datetime
    metadata: dict[str, Any]


class InMemoryCheckpointStore:
    """In-memory checkpoint storage.

    In production, replace with persistent storage (file system, database, S3).
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to storage."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        print(
            f"💾 Checkpoint saved: {checkpoint.checkpoint_id} (step {checkpoint.step})"
        )

    async def load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from storage."""
        return self._checkpoints.get(checkpoint_id)

    async def list_checkpoints(self, execution_id: str) -> list[str]:
        """List all checkpoint IDs for an execution."""
        return [
            cp.checkpoint_id
            for cp in self._checkpoints.values()
            if cp.execution_id == execution_id
        ]

    async def get_latest_checkpoint(self, execution_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for an execution."""
        checkpoints = [
            cp for cp in self._checkpoints.values() if cp.execution_id == execution_id
        ]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: cp.timestamp)


class CheckpointableLLM:
    """LLM caller with checkpoint support.

    Wraps OpenAI API to add periodic checkpointing during execution.
    """

    def __init__(
        self, checkpoint_store: InMemoryCheckpointStore, api_key: str | None = None
    ) -> None:
        """Initialize checkpointable LLM.

        Args:
            checkpoint_store: Storage for checkpoints
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.checkpoint_store = checkpoint_store

    async def solve_with_checkpoints(
        self,
        task: str,
        execution_id: str,
        fail_on_step: int | None = None,
        model: str = "gpt-4.1",
    ) -> str:
        """Solve task with automatic checkpointing.

        Args:
            task: Task description
            execution_id: Unique identifier for this execution
            fail_on_step: Step number to simulate failure (for demo)
            model: OpenAI model to use

        Returns:
            Final result

        Raises:
            RuntimeError: If simulated failure occurs
        """
        print(f"🚀 Starting execution: {execution_id}\n")

        # Step 1: Save initial checkpoint
        checkpoint_id = f"{execution_id}-step-0"
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            execution_id=execution_id,
            task=task,
            step=0,
            partial_result=None,
            timestamp=datetime.now(),
            metadata={"status": "started"},
        )
        await self.checkpoint_store.save_checkpoint(checkpoint)

        # Simulate failure at specific step
        if fail_on_step == 1:
            raise RuntimeError(f"Simulated failure at step 1")

        # Step 2: Call LLM
        print("📡 Calling LLM...")
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task}],
                temperature=0.7,
                max_tokens=2000,
            )

            result = response.choices[0].message.content or ""

            # Step 3: Save completion checkpoint
            checkpoint_id = f"{execution_id}-step-1"
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                task=task,
                step=1,
                partial_result=result,
                timestamp=datetime.now(),
                metadata={
                    "status": "completed",
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage
                        else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens
                        if response.usage
                        else 0,
                    },
                },
            )
            await self.checkpoint_store.save_checkpoint(checkpoint)

            print("✅ Execution completed\n")
            return result

        except Exception as e:
            # Save failure checkpoint
            checkpoint_id = f"{execution_id}-step-1-failed"
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                execution_id=execution_id,
                task=task,
                step=1,
                partial_result=None,
                timestamp=datetime.now(),
                metadata={"status": "failed", "error": str(e)},
            )
            await self.checkpoint_store.save_checkpoint(checkpoint)
            raise

    async def resume_from_checkpoint(
        self, execution_id: str, model: str = "gpt-4.1"
    ) -> str:
        """Resume execution from last checkpoint.

        Args:
            execution_id: Execution ID to resume
            model: OpenAI model to use

        Returns:
            Final result

        Raises:
            ValueError: If no checkpoint found
        """
        # Load last checkpoint
        checkpoint = await self.checkpoint_store.get_latest_checkpoint(execution_id)

        if not checkpoint:
            raise ValueError(f"No checkpoint found for execution {execution_id}")

        print(f"🔄 Resuming from checkpoint: {checkpoint.checkpoint_id}")
        print(f"   Step: {checkpoint.step}")
        print(f"   Status: {checkpoint.metadata.get('status', 'unknown')}\n")

        # If already completed, return the result
        if checkpoint.metadata.get("status") == "completed":
            print("✅ Execution already completed\n")
            return checkpoint.partial_result or ""

        # Otherwise, re-run the task (in real implementation, would use checkpoint state)
        print("🔄 Re-running task from checkpoint...\n")
        return await self.solve_with_checkpoints(
            checkpoint.task, execution_id, fail_on_step=None, model=model
        )


async def main() -> None:
    """Run checkpoint example with fault tolerance and recovery."""
    print("=" * 80)
    print("RLM Checkpoint Example: Fault Tolerance and Recovery")
    print("=" * 80)
    print("\nThis example demonstrates checkpointing with direct LLM calls.")
    print("Progress is automatically saved, allowing recovery from failures.\n")

    # Create checkpoint store
    checkpoint_store = InMemoryCheckpointStore()

    # Create checkpointable LLM
    llm = CheckpointableLLM(checkpoint_store)

    execution_id = "demo-execution-001"

    # Define a task
    task = """
    Write a 150-word summary about cloud computing benefits.

    Include: scalability, cost savings, and flexibility.
    """

    print(f"Task: {task.strip()}\n")

    # PART 1: Run with simulated failure
    print("=" * 80)
    print("PART 1: Initial Execution (will fail at step 1)")
    print("=" * 80)
    print()

    try:
        await llm.solve_with_checkpoints(
            task=task,
            execution_id=execution_id,
            fail_on_step=1,  # Simulate failure
        )
    except RuntimeError as e:
        print(f"\n✓ Expected failure occurred: {e}")
        print("✓ Checkpoints were saved before the failure\n")

    # Check saved checkpoints
    checkpoints = await checkpoint_store.list_checkpoints(execution_id)
    print(f"Checkpoints saved: {len(checkpoints)}")
    for checkpoint_id in checkpoints:
        checkpoint = await checkpoint_store.load_checkpoint(checkpoint_id)
        if checkpoint:
            status = checkpoint.metadata.get("status", "unknown")
            print(f"  • {checkpoint_id} - Step {checkpoint.step} - {status}")

    # PART 2: Recover from checkpoint
    print("\n" + "=" * 80)
    print("PART 2: Recovery from Last Checkpoint")
    print("=" * 80)
    print()

    try:
        # Resume from last checkpoint
        result = await llm.resume_from_checkpoint(execution_id=execution_id)

        print("=" * 80)
        print("FINAL RESULT (after recovery)")
        print("=" * 80)
        print(result)
        print()

        # Print final checkpoint info
        final_checkpoint = await checkpoint_store.get_latest_checkpoint(execution_id)
        if final_checkpoint:
            usage = final_checkpoint.metadata.get("usage", {})
            if usage:
                print("=" * 80)
                print("STATISTICS")
                print("=" * 80)
                print(f"Total tokens: {usage.get('total_tokens', 0)}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nThis example demonstrates:")
        print("  ✓ Automatic checkpoint saving during execution")
        print("  ✓ Simulated failure at step 1")
        print("  ✓ Checkpoint preserved before failure")
        print("  ✓ Successful recovery from last checkpoint")
        print("  ✓ Task completion after recovery")
        print("\nFor production:")
        print("  - Use persistent storage (file system, database, S3)")
        print("  - Add checkpoint cleanup/expiration policies")
        print("  - Implement incremental state saving")
        print("  - Add checkpoint versioning and rollback")
        print("\nFor recursive decomposition with checkpointing:")
        print("  - Save checkpoints at each recursion level")
        print("  - Store sub-task results in checkpoints")
        print("  - Resume from deepest successful checkpoint")

    except Exception as e:
        print(f"\n❌ Error during recovery: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
