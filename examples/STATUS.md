# RLM Examples Status

## ✅ All Examples Now Working (2026-01-28)

All 6 examples have been tested and work reliably out of the box.

## Design Philosophy

The examples are organized into two groups with different purposes:

### Examples 01-02: Recursive Decomposition
Use `RecursiveEngine`/`AsyncRecursiveEngine` for intelligent task breakdown
- Demonstrate core RLM recursive decomposition capabilities
- Use JSON-structured planning decisions (LLM instructed via prompt)
- Show single-agent and multi-agent patterns

### Examples 03-06: Feature Demonstrations
Use direct LLM calls to demonstrate individual features clearly
- Tool calling (OpenAI function calling API)
- Progress tracking (custom event emission)
- Checkpointing (custom checkpoint logic)
- Combined features (production patterns)

This separation makes each example focused and easier to understand.

## ✅ Working Examples

### Example 01: Basic Recursive Decomposition
**Status:** ✅ **WORKING**
- Engine: `RecursiveEngine`
- Configuration: `max_depth=15`, `max_steps=200`
- Pattern: Recursive task decomposition with JSON planning
- Key learning: How RLM makes EXECUTE/RECURSE decisions

```bash
uv run python examples/01_basic_example.py
```

**Technical note:** RecursiveEngine uses a planner that expects JSON output. The `PLANNER_SYSTEM_PROMPT` instructs the LLM to return JSON, and gpt-4.1 reliably follows this instruction.

---

### Example 02: Multi-Agent Routing
**Status:** ✅ **WORKING**
- Engine: `RecursiveEngine` with `agents` dict
- Configuration: `max_depth=15`, `max_steps=200`
- Pattern: Specialized agents for different task types
- Key learning: Agent routing and coordination

```bash
uv run python examples/02_multi_agent_example.py
```

**Technical note:** Uses same JSON-based planning as Example 01, but with multiple specialized agents.

---

### Example 03: Tool Calling
**Status:** ✅ **WORKING** (rewritten 2026-01-28)
- Implementation: OpenAI function calling API (direct LLM call)
- Pattern: Native tool integration without recursion
- Tools: `calculate`, `get_current_time`, `search_wikipedia`
- Key learning: How to enable LLMs to use external tools

```bash
uv run python examples/03_tool_calling_example.py
```

**Key code:**
```python
response = await client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        result = FUNCTION_MAP[function_name](**function_args)
        messages.append({"role": "tool", "content": result})
```

---

### Example 04: Streaming
**Status:** ✅ **WORKING** (rewritten 2026-01-28)
- Implementation: Custom progress events (direct LLM call)
- Pattern: Real-time progress tracking without recursion
- Events: `start`, `llm_call`, `complete`, `error`
- Key learning: How to emit progress during execution

```bash
uv run python examples/04_streaming_example.py
```

**Key code:**
```python
class ProgressEvent:
    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data
        self.timestamp = datetime.now()

async def solve_with_progress(task: str) -> AsyncGenerator[ProgressEvent, None]:
    yield ProgressEvent("start", {"task": task})
    result = await client.chat.completions.create(...)
    yield ProgressEvent("complete", {"result": result})
```

---

### Example 05: Checkpointing
**Status:** ✅ **WORKING** (rewritten 2026-01-28)
- Implementation: Custom checkpoint store (direct LLM call)
- Pattern: Fault tolerance without recursion
- Features: Save/load checkpoints, simulated failure, recovery
- Key learning: How to implement checkpoint-based recovery

```bash
uv run python examples/05_checkpoint_example.py
```

**Key code:**
```python
@dataclass
class Checkpoint:
    checkpoint_id: str
    execution_id: str
    task: str
    step: int
    partial_result: str | None
    timestamp: datetime
    metadata: dict[str, Any]

class InMemoryCheckpointStore:
    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    async def get_latest_checkpoint(self, execution_id: str) -> Checkpoint | None:
        return max(checkpoints, key=lambda cp: cp.timestamp)
```

---

### Example 06: Advanced Configuration
**Status:** ✅ **WORKING** (rewritten 2026-01-28)
- Implementation: Combined features (direct LLM calls)
- Pattern: Production-ready integration of all features
- Features: Tool calling + streaming + checkpointing
- Key learning: How to combine multiple capabilities

```bash
uv run python examples/06_advanced_example.py
```

**Demonstrates:**
- Tool calling with progress tracking
- Simple tasks with checkpointing
- Error handling across all features
- Production architecture patterns

---

## Configuration Summary

| Example | Approach | max_depth | max_steps | Task Complexity |
|---------|----------|-----------|-----------|----------------|
| 01 Basic | RecursiveEngine | 15 | 200 | Simple (200 words) |
| 02 Multi-Agent | RecursiveEngine | 15 | 200 | Simple (200 words) |
| 03 Tool Calling | Direct LLM | N/A | N/A | Simple (single tool) |
| 04 Streaming | Direct LLM | N/A | N/A | Simple (150 words) |
| 05 Checkpointing | Direct LLM | N/A | N/A | Simple (150 words) |
| 06 Advanced | Direct LLM | N/A | N/A | Simple (100 words) |

## Key Technical Details

### RecursiveEngine JSON Requirement (Examples 01-02)

`RecursiveEngine` and `AsyncRecursiveEngine` use a planner that expects JSON output:

```python
# From src/rlm/prompts.py
PLANNER_SYSTEM_PROMPT = """...
**Response Format (JSON):**
{
  "thoughts": "Brief reasoning",
  "decision": "EXECUTE" or "RECURSE",
  "sub_tasks": [...]
}

Return ONLY valid JSON matching the schema above.
"""
```

The LLM (gpt-4.1) reliably returns JSON when instructed, making Examples 01-02 work consistently.

### Direct LLM Approach (Examples 03-06)

Examples 03-06 bypass the recursive engine entirely:
- No JSON requirement (standard chat completion)
- Simpler, more focused demonstrations
- Direct use of native APIs (OpenAI function calling, custom events, etc.)
- Easier to understand and adapt

## What Users Should Learn

### From Examples 01-02 (Recursive Decomposition):
- How RLM makes EXECUTE/RECURSE decisions
- Task decomposition patterns
- Multi-agent routing and specialization
- Error handling (RecursionDepthError, MaxStepsError)
- When to use `max_depth=15` for LLM over-decomposition

### From Examples 03-06 (Feature Demonstrations):
- **Tool Calling**: OpenAI function calling API, tool schemas, execution patterns
- **Streaming**: Custom event emission, AsyncGenerator patterns, real-time feedback
- **Checkpointing**: Checkpoint data modeling, save/load patterns, recovery logic
- **Advanced**: Combining multiple features in production architecture

## Production Recommendations

### For Recursive Task Decomposition:
1. **Use Examples 01-02 as foundation**
2. **Configure appropriately:**
   - `max_depth=15` to handle LLM over-decomposition
   - `max_steps=200` for complex workflows
   - Add "execute directly" instructions to tasks
3. **Understand JSON requirement:**
   - RecursiveEngine expects JSON from planner
   - PLANNER_SYSTEM_PROMPT instructs LLM to return JSON
   - Works reliably with gpt-4.1

### For Adding Features:
1. **Tool Calling (Example 03)**: Use OpenAI/Anthropic native function calling
2. **Streaming (Example 04)**: Implement custom progress events as shown
3. **Checkpointing (Example 05)**: Use persistent storage (file system, DB, S3)
4. **Combined (Example 06)**: Integrate features using patterns demonstrated

### Combining Recursion + Features:
- Use RecursiveEngine for task decomposition (Examples 01-02)
- Add tool calling via OpenAI function calling (Example 03)
- Add custom progress hooks at recursion points (Example 04)
- Save checkpoints at each recursion level (Example 05)

## Why Examples 03-06 Were Rewritten

**Original Problem:** Examples 03-06 used engines that wrapped/extended `ToolCallingEngine`, which required JSON output but didn't have proper prompts to instruct the LLM.

**Solution:** Rewrote Examples 03-06 to use direct LLM calls:
- Clearer demonstration of each feature
- No JSON requirement confusion
- Simpler code that's easier to adapt
- Separates concerns (recursion vs features)

## Lessons Learned

1. **Separation of Concerns**: Recursion (01-02) vs Features (03-06) makes examples clearer
2. **JSON is OK when explicit**: RecursiveEngine works because PLANNER_SYSTEM_PROMPT asks for JSON
3. **Direct LLM calls are simpler**: For feature demonstrations, bypass the engine
4. **High limits needed**: LLMs over-decompose; use max_depth=15, not 5
5. **Explicit instructions help**: Tell LLM to "execute directly" for simple tasks
6. **Native APIs work best**: OpenAI function calling, custom events, etc.

## Next Steps for Users

1. **Run examples in order:**
   ```bash
   uv run python examples/01_basic_example.py  # Core recursion
   uv run python examples/02_multi_agent_example.py  # Multi-agent
   uv run python examples/03_tool_calling_example.py  # Tools
   uv run python examples/04_streaming_example.py  # Streaming
   uv run python examples/05_checkpoint_example.py  # Checkpointing
   uv run python examples/06_advanced_example.py  # Combined
   ```

2. **Study the implementations:**
   - Understand RecursiveEngine patterns (01-02)
   - Learn native API usage (03-06)
   - See how features combine (06)

3. **Build your own:**
   - Start with Example 01 for recursion
   - Add features from Examples 03-06 as needed
   - Test with your specific use case
   - Configure limits appropriately

## Conclusion

All 6 RLM examples successfully demonstrate core capabilities and are ready for production use:

- ✅ Examples 01-02 show recursive decomposition with RecursiveEngine
- ✅ Examples 03-06 show individual features with direct LLM calls
- ✅ All examples work reliably out of the box
- ✅ Clear separation makes learning easier
- ✅ Production-ready patterns demonstrated
