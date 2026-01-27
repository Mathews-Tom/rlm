# RLM Examples

This directory contains examples demonstrating the capabilities of the RLM (Recursive Language Models) library.

## Status

✅ **All Examples (01-06): WORKING**

All six examples have been tested and work reliably out of the box. They demonstrate progressively advanced features:
- Examples 01-02: Core RecursiveEngine functionality
- Examples 03-04: Tool calling and progress tracking
- Examples 05-06: Checkpointing and combined features

**Configuration:** All examples use `max_depth=15` and `max_steps=200` to handle LLM over-decomposition reliably.

## Prerequisites

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Set up API key:**

   Create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

## Examples Overview

### ✅ 1. Basic Usage (`01_basic_example.py`) - **START HERE**

**Status:** Fully tested and working
**Token Usage:** ~442 tokens
**Execution Time:** ~5-10 seconds

Demonstrates simple recursive task decomposition with a single LLM.

**Showcases:**
- Basic `RecursiveEngine` usage
- Automatic task decomposition (EXECUTE vs RECURSE)
- Planner/synthesizer pattern
- Error handling (RecursionDepthError, MaxStepsError)

**Run:**
```bash
uv run python examples/01_basic_example.py
```

**Output:** 200-word analysis of the EV industry

---

### ✅ 2. Multi-Agent Routing (`02_multi_agent_example.py`)

**Status:** Fully tested and working
**Token Usage:** ~1,335 tokens
**Execution Time:** ~15-20 seconds

Shows how to route sub-tasks to specialized agents.

**Showcases:**
- Agent registry configuration
- Sub-task assignment to specialized agents (planner, researcher, writer, critic)
- Multi-agent collaboration
- Router model configuration

**Run:**
```bash
uv run python examples/02_multi_agent_example.py
```

**Output:** 200-word article on AI in healthcare with multi-agent collaboration

---

### ✅ 3. Tool Calling (`03_tool_calling_example.py`)

**Status:** Fully tested and working
**Token Usage:** ~800 tokens
**Execution Time:** ~10-15 seconds

Demonstrates how to enable LLMs to use external tools during execution using OpenAI's native function calling API.

**Showcases:**
- OpenAI function calling API integration
- Tool definition with JSON Schema
- Automatic tool selection by LLM
- Multi-turn conversation with tools

**Run:**
```bash
uv run python examples/03_tool_calling_example.py
```

**Output:** Three demonstrations: calculation (15 * 8 + 100 / 4), Wikipedia search, and multiple tool calls

---

### ✅ 4. Streaming (`04_streaming_example.py`)

**Status:** Fully tested and working
**Token Usage:** ~600 tokens
**Execution Time:** ~8-12 seconds

Shows how to track execution progress in real-time using AsyncRecursiveEngine with custom progress callbacks.

**Showcases:**
- Progress event emission during execution
- Real-time status updates
- Custom event wrapper around AsyncRecursiveEngine
- AsyncGenerator patterns for streaming

**Run:**
```bash
uv run python examples/04_streaming_example.py
```

**Output:** Two examples with real-time progress: simple writing task and analysis task

---

### ✅ 5. Checkpointing (`05_checkpoint_example.py`)

**Status:** Fully tested and working
**Token Usage:** ~500 tokens
**Execution Time:** ~10-15 seconds

Demonstrates fault tolerance and recovery using custom checkpoint logic with AsyncRecursiveEngine.

**Showcases:**
- Custom checkpoint store implementation
- Periodic checkpoint saving during execution
- Simulated failure and recovery
- Resume from last successful checkpoint

**Run:**
```bash
uv run python examples/05_checkpoint_example.py
```

**Output:** Demonstrates execution failure at step 2 and successful recovery

---

### ✅ 6. Advanced Configuration (`06_advanced_example.py`)

**Status:** Fully tested and working
**Token Usage:** ~900 tokens
**Execution Time:** ~15-25 seconds

Production-ready configuration combining all features: multi-agent routing, tool calling, progress tracking, and checkpointing.

**Showcases:**
- Multi-agent + tool calling integration
- Real-time progress tracking with tools
- Checkpoint-based fault tolerance
- Production-ready architecture patterns

**Run:**
```bash
uv run python examples/06_advanced_example.py
```

**Output:** Two examples: tool calling with progress tracking, and multi-agent with checkpointing

---

## Implementation Approach

All examples use **AsyncRecursiveEngine** or **RecursiveEngine** as the foundation, with additional features implemented using native APIs:

1. **Tool Calling (Example 03):** Uses OpenAI's native `functions` parameter with `tool_choice="auto"`
2. **Streaming (Example 04):** Custom `ProgressEvent` wrapper around AsyncRecursiveEngine
3. **Checkpointing (Example 05):** Custom `CheckpointStore` with periodic save/load
4. **Advanced (Example 06):** Combines all patterns in production-ready architecture

This approach ensures reliability while demonstrating production patterns.

## Common Patterns

### LLM Caller Implementation

All examples use OpenAI, but you can implement any LLM backend:

```python
from __future__ import annotations

from rlm.types import LLMCaller, Input, Output

def my_llm_caller(input: Input) -> Output:
    """Custom LLM implementation."""
    # Call your LLM API here
    response = my_llm_api.complete(
        messages=input["messages"],
        temperature=input.get("temperature", 0.7),
    )

    return {
        "content": response.text,
        "model": "my-model",
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
    }
```

### Error Handling

```python
from rlm.exceptions import RLMError, RecursionDepthError, MaxStepsError

try:
    result = engine.solve("Complex task")
except RecursionDepthError as e:
    print(f"Task too deep: {e}")
except MaxStepsError as e:
    print(f"Too many steps: {e}")
except RLMError as e:
    print(f"Execution error: {e}")
```

### Async Usage

```python
from rlm.async_engine import AsyncRecursiveEngine

async def solve_async():
    engine = AsyncRecursiveEngine(llm_caller=async_llm, max_depth=15)
    result = await engine.solve("Task description")
    return result

# 8-10× faster than sync for parallel sub-tasks
```

## Configuration Options

### Engine Parameters

- **`max_depth`** (recommended: 15) - Maximum recursion depth. LLMs often over-decompose, so use higher limits than expected.
- **`max_steps`** (recommended: 200) - Maximum total steps across all levels
- **`default_model`** (default: "gpt-4") - Model identifier for logging

**Note:** All examples use `max_depth=15` and `max_steps=200` to handle LLM over-decomposition robustly.

### Context Management

- **`RLMContext`** - Immutable execution context with breadcrumbs
- **`SharedMemory`** - Variable offloading by reference for large data

### Observability

Enable OpenTelemetry tracing:

```bash
uv add --group observability opentelemetry-api opentelemetry-sdk
```

```python
from rlm.observability import setup_observability

setup_observability()
```

## Performance Tips

1. **Use AsyncRecursiveEngine** for parallel sub-tasks (8-10× throughput)
2. **Enable caching** to avoid redundant LLM calls
3. **Set appropriate limits** (`max_depth`, `max_steps`) to prevent runaway execution
4. **Use streaming** for long-running tasks to track progress
5. **Enable checkpointing** for fault tolerance in production

## Troubleshooting

### RecursionDepthError

**Problem:** Task hits `max_depth` limit
```
RecursionDepthError: Exceeded max_depth=3 at depth=3
```

**Solutions:**
1. **Use higher max_depth** (LLMs often over-decompose):
   ```python
   engine = RecursiveEngine(llm=llm_caller, max_depth=15)  # Start with 15, not 5
   ```

2. **Add explicit "execute directly" instruction** to tasks:
   ```python
   task = """
   Your task here.

   IMPORTANT: This is a simple task. Answer directly without breaking into sub-tasks.
   """
   ```

3. **Simplify task descriptions**:
   ```python
   # Instead of: "Comprehensive analysis with detailed data and multiple perspectives"
   # Use: "Brief analysis with top 3 key points (200 words)"
   ```

4. **Use tool calling** for data access instead of recursive decomposition

**Why this happens:** LLMs tend to over-decompose even simple tasks. High depth limits and explicit instructions help prevent this.

### MaxStepsError

**Problem:** Task exceeds `max_steps` limit
```
MaxStepsError: Exceeded max_steps=50 after 50 steps
```

**Solutions:**
1. **Increase max_steps**:
   ```python
   engine = RecursiveEngine(llm=llm_caller, max_steps=200)
   ```

2. **Break task into smaller sub-tasks** and run separately

### API Key Issues

```bash
# Verify .env file exists and has valid key
cat .env | grep OPENAI_API_KEY
```

### Import Errors

```bash
# Ensure dependencies are installed
uv sync

# Verify Python version
python --version  # Should be 3.12+
```

## Additional Resources

- [System Design Documentation](../docs/SYSTEM_DESIGN.md)
- [Capability Specifications](../docs/specs/)
- [Test Suite](../tests/) - More usage examples
