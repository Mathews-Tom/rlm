from __future__ import annotations

import rlm.async_types as async_types
import rlm.types as rlm_types
from rlm.async_engine import AsyncRecursiveEngine
from rlm.budget import TokenUsage, UsageAccumulator, extract_usage
from rlm.caching import CachedAsyncEngine
from rlm.checkpoints import (
    Checkpoint,
    CheckpointStore,
    CheckpointableEngine,
    InMemoryCheckpointStore,
)
from rlm.engine import RecursiveEngine
from rlm.exceptions import (
    BudgetExceeded,
    CodeExecutionError,
    CostLimitExceeded,
    ExecutionError,
    InvalidJSONError,
    MaxREPLIterationsError,
    MaxStepsError,
    RecursionDepthError,
    RLMError,
    TokenLimitExceeded,
)
from rlm.memory import RLMContext, SharedMemory
from rlm.observability import InstrumentedAsyncEngine
from rlm.prompts import PLANNER_SYSTEM_PROMPT, REPL_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from rlm.repl_engine import CodeExecutionEngine, ContextProxy, REPLConfig
from rlm.streaming import StreamEvent, StreamingEngine
from rlm.tools import Tool, ToolCallingEngine, ToolRegistry
from rlm.types import (
    Input,
    Item,
    LLMCaller,
    Output,
    PlannerDecision,
    SubTask,
    TraceObject,
)
from rlm.types import AsyncStreamingLLMCaller, REPLIteration, ToolCall, UsageInfo

AsyncInput = async_types.AsyncInput
AsyncItem = async_types.AsyncItem
AsyncOutput = async_types.AsyncOutput
AsyncLLMCaller = rlm_types.AsyncLLMCaller
AsyncToolCaller = rlm_types.AsyncToolCaller

__all__ = [
    # Engines
    "RecursiveEngine",
    "AsyncRecursiveEngine",
    "CachedAsyncEngine",
    "InstrumentedAsyncEngine",
    "ToolCallingEngine",
    "StreamingEngine",
    "CodeExecutionEngine",
    # Checkpoints
    "Checkpoint",
    "CheckpointStore",
    "CheckpointableEngine",
    "InMemoryCheckpointStore",
    # Types
    "Input",
    "Output",
    "Item",
    "LLMCaller",
    "AsyncLLMCaller",
    "AsyncToolCaller",
    "AsyncStreamingLLMCaller",
    "AsyncInput",
    "AsyncItem",
    "AsyncOutput",
    "SubTask",
    "PlannerDecision",
    "TraceObject",
    "ToolCall",
    "UsageInfo",
    "REPLIteration",
    # Budget
    "TokenUsage",
    "UsageAccumulator",
    "extract_usage",
    # REPL
    "ContextProxy",
    "REPLConfig",
    # Tools
    "Tool",
    "ToolRegistry",
    # Streaming
    "StreamEvent",
    # Memory
    "RLMContext",
    "SharedMemory",
    # Prompts
    "PLANNER_SYSTEM_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "REPL_SYSTEM_PROMPT",
    # Exceptions
    "RLMError",
    "RecursionDepthError",
    "MaxStepsError",
    "InvalidJSONError",
    "ExecutionError",
    "BudgetExceeded",
    "TokenLimitExceeded",
    "CostLimitExceeded",
    "CodeExecutionError",
    "MaxREPLIterationsError",
]

__version__ = "0.1.0"
