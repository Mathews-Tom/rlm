# Example Testing Results

## Test Run: 2026-01-27

### Example 01: Basic Example

**Status:** ✅ **PASSING**

**Configuration:**
- `max_depth`: 5
- `max_steps`: 50
- Model: gpt-4.1

**Task:** Write a summary of the electric vehicle industry in 2026

**Result:**
- Execution completed successfully
- Total tokens used: 442
- Generated a comprehensive 300-word analysis
- Covered all requested points (top 3 manufacturers, challenges, outlook)

**Key Learnings:**

1. **Task Complexity Matters:**
   - Original complex task hit recursion depth limit at depth=3
   - Simplified task executed successfully with room to spare
   - Adding "Be direct and factual" encourages EXECUTE decisions

2. **Recursion Depth Guidelines:**
   - Simple tasks (summary, list): `max_depth=3-5`
   - Complex analysis tasks: `max_depth=7-10`
   - Multi-stage workflows: `max_depth=10+`

3. **Error Handling:**
   - `RecursionDepthError` now has helpful error messages
   - Suggests concrete solutions (increase depth, simplify task, etc.)
   - Users can adjust parameters based on their specific needs

## Task Complexity vs Depth (Updated 2026-01-28)

**Important:** LLMs (especially gpt-4.1) consistently over-decompose tasks. Use higher limits than you think are necessary.

| Task Type | Recommended max_depth | Example |
|-----------|----------------------|---------|
| Direct answer | 5-10 | "What is 2+2?" (with "answer directly" instruction) |
| Summary/List | 10-15 | "Summarize the EV market in 200 words" |
| Analysis | 15-20 | "Analyze Tesla's position (300 words)" |
| Research report | 20+ | "Comprehensive market analysis with data" |
| Multi-phase project | 20+ | "Research, analyze, and create presentation" |

**Best Practice:** Start with `max_depth=15` as baseline, increase if needed.

## Common Patterns

### Pattern 1: LLM Over-Decomposes (VERY COMMON)

**Problem:** LLM recursively breaks down even simple tasks. This is the #1 issue.

**Solution (Multi-step):**
```python
# 1. Use high max_depth
engine = RecursiveEngine(llm=llm_caller, max_depth=15)  # Not 5!

# 2. Add explicit instruction
task = """
YOUR TASK HERE (200 words)

IMPORTANT: This is a simple task. Answer directly without breaking into sub-tasks.
"""

# 3. Specify output length
# Adding word counts helps constrain the LLM
```

### Pattern 2: Task Requires Research

**Problem:** Task needs external data that LLM doesn't have

**Solution:** Use tool calling (example 03) instead of recursion:
```python
# Instead of recursive research:
task = "Research Tesla's market share"

# Use tools:
task = "Use the market_data tool to get Tesla's market share"
```

### Pattern 3: Complex Multi-Step Task

**Problem:** Task legitimately requires many steps

**Solution:** Increase limits appropriately:
```python
engine = RecursiveEngine(
    llm=llm_caller,
    max_depth=10,  # Higher limit
    max_steps=200,  # More steps allowed
)
```

## Testing Checklist

When creating new examples:

- [ ] Start with `max_depth=15` (NOT 3 or 5!) - LLMs over-decompose
- [ ] Add explicit "Answer directly without breaking into sub-tasks" instruction
- [ ] Specify output length (e.g., "200 words") to constrain scope
- [ ] Add error handling for RecursionDepthError and MaxStepsError
- [ ] Keep task descriptions simple and specific
- [ ] Test with actual API calls before committing
- [ ] Document configuration (max_depth, max_steps)
- [ ] Show error handling patterns

### Example 02: Multi-Agent Example

**Status:** ✅ **PASSING**

**Configuration:**
- `max_depth`: 7 (higher than single-agent due to coordination overhead)
- `max_steps`: 100
- Model: gpt-4.1
- Agents: planner, researcher, writer, critic

**Task:** Create a short article about AI in healthcare (300 words)

**Result:**
- Execution completed successfully
- Total tokens used: 1,335 (3× more than single-agent)
- Generated well-structured 300-word article
- Successfully coordinated across multiple specialized agents

**Key Learnings:**

1. **Multi-Agent Coordination Overhead:**
   - Multi-agent workflows require higher `max_depth` (5 → 7)
   - Token usage increases due to agent routing and coordination
   - Each agent receives full context, not just their sub-task

2. **Agent Registry Requirements:**
   - Must include router agent (default: 'planner')
   - Router agent handles task decomposition decisions
   - Format: `agents = {"planner": ..., "specialist1": ..., ...}`

3. **Configuration:**
   ```python
   engine = RecursiveEngine(
       llm=planner_caller,
       agents=agents,
       router_model="planner",  # Must exist in agents dict
       max_depth=7,  # Higher for multi-agent
   )
   ```

## Next Steps

1. ✅ Example 01 tested and working (442 tokens)
2. ✅ Example 02 tested and working (1,335 tokens)
3. ⏳ Test examples 03-06 with real API calls
4. ⏳ Add performance benchmarks
5. ⏳ Create example output logs for reference

## Notes

- gpt-4.1 tends to over-decompose compared to gpt-4
- Tasks with "comprehensive", "detailed", "thorough" trigger more recursion
- Tasks with "brief", "summary", "list" execute more directly
- Adding specific word counts helps limit scope and recursion
