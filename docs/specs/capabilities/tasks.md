# Tasks: Advanced Capabilities (CAP-001)

**From:** `capabilities/spec.md` + `capabilities/plan.md`
**Timeline:** 12 weeks (Weeks 5-16, depends on CORE-001, INTEL-001, PERF-001)
**Team:** 1-2 backend engineers
**Created:** 2026-01-25

## Summary

- Total tasks: 15 stories
- Estimated effort: 48-60 story points
- Critical path duration: 12 weeks
- Key risks: Tool execution failures, streaming latency, checkpoint serialization complexity

## Phase Breakdown

### Phase 1: Tool Calling Framework (Weeks 5-8, 16 SP)

**Goal:** Enable agents to call external tools/APIs during execution
**Deliverable:** ToolCallingEngine with 5+ example integrations

#### Tasks

**[CAP-002] Tool Protocol and Registry Implementation**

- **Description:** Implement Tool dataclass, tool registration system, and LLM protocol extensions for tool calling support
- **Acceptance:**
  - [ ] Tool dataclass with name, description, parameters (JSON Schema), callable fields
  - [ ] ToolRegistry class for managing available tools
  - [ ] Tools registered via engine initialization (tools parameter)
  - [ ] LLMCaller protocol extended to support tool_calls in Output
  - [ ] Tool validation: name uniqueness, valid JSON Schema parameters
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] All public types exported in **all**
  - [ ] Google-style docstrings with examples
- **Effort:** 3 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-001, PERF-001 (async engine)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/tools.py (create)

**[CAP-003] Tool Execution Engine**

- **Description:** Implement tool call execution loop with error handling, retries, and result injection
- **Acceptance:**
  - [ ] ToolCallingEngine extends AsyncRecursiveEngine
  - [ ] \_execute_tool_calls(tool_calls) method executes tools in sequence
  - [ ] Tool execution errors caught and returned as error strings
  - [ ] Unknown tools handled gracefully (error message, not crash)
  - [ ] Tool results injected into conversation context
  - [ ] Retry logic for transient failures (up to 2 retries)
  - [ ] Timeout handling for long-running tools (configurable, default 30s)
  - [ ] Verbose logging when verbose=True
  - [ ] All methods have comprehensive docstrings
- **Effort:** 5 story points (3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-002
- **Priority:** P0 (Critical)
- **Files:** src/rlm/tools.py (modify)

**[CAP-004] Iterative Tool Calling Loop**

- **Description:** Implement multi-turn tool calling where LLM can request multiple tool calls before final answer
- **Acceptance:**
  - [ ] \_execute_leaf_with_tools(task, context) implements iterative loop
  - [ ] Loop continues until LLM returns response without tool_calls
  - [ ] Maximum iterations limit (default 5, configurable)
  - [ ] Conversation history preserved across iterations
  - [ ] Tool results formatted as tool role messages
  - [ ] Loop termination when max iterations reached (with warning)
  - [ ] Read-before-write pattern supported (tool can call another tool)
  - [ ] State tracking: track number of tool calls per task
  - [ ] Integration with existing \_execute_leaf method
- **Effort:** 5 story points (3-4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-003
- **Priority:** P0 (Critical)
- **Files:** src/rlm/tools.py (modify), src/rlm/engine.py (integrate)

**[CAP-005] Example Tool Implementations**

- **Description:** Create 5+ example tool integrations demonstrating common use cases
- **Acceptance:**
  - [ ] search_web tool: Web search API integration (example: DuckDuckGo)
  - [ ] calculator tool: Safe mathematical expression evaluation
  - [ ] file_read tool: Read file contents with path validation
  - [ ] file_write tool: Write file contents with safety checks
  - [ ] http_request tool: Make HTTP GET/POST requests
  - [ ] current_time tool: Return current datetime
  - [ ] Each tool has comprehensive docstring with examples
  - [ ] Each tool has parameter validation
  - [ ] Each tool handles errors gracefully
  - [ ] All tools use type hints (dict[str, Any] -> str)
- **Effort:** 2 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-003
- **Priority:** P1 (Important)
- **Files:** src/rlm/tools/examples.py (create)

**[CAP-006] Tool Calling Test Suite**

- **Description:** Comprehensive unit and integration tests for tool calling functionality
- **Acceptance:**
  - [ ] test_tools.py: Tool dataclass validation, registry operations
  - [ ] test_tool_execution.py: Tool execution, error handling, retries
  - [ ] test_tool_loop.py: Iterative tool calling, max iterations
  - [ ] Mock tools for deterministic testing
  - [ ] Test tool execution success and failure cases
  - [ ] Test unknown tool handling
  - [ ] Test max iterations enforcement
  - [ ] Test read-before-write pattern (tool calling another tool)
  - [ ] Integration test with real OpenAI tool calling
  - [ ] All tests pass with pytest
  - [ ] Coverage ≥90% for tools module
- **Effort:** 3 story points (3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-005
- **Priority:** P0 (Critical)
- **Files:** tests/unit/test_tools.py (create), tests/integration/test_tool_calling.py (create)

### Phase 2: Streaming Responses (Weeks 9-11, 12 SP)

**Goal:** Progressive results via AsyncGenerator for improved UX
**Deliverable:** StreamingEngine with <500ms TTFT and SSE integration example

#### Tasks

**[CAP-007] Streaming Event Protocol**

- **Description:** Implement StreamEvent dataclass and event type system for streaming responses
- **Acceptance:**
  - [ ] StreamEvent dataclass with type, data, metadata fields
  - [ ] Event types: plan, token, result, error (Literal type)
  - [ ] Metadata includes depth, task_id, timestamp
  - [ ] StreamEvent serialization to JSON for transport
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] All types exported in **all**
  - [ ] Google-style docstrings with examples
- **Effort:** 2 story points (1 hour)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-001 (async engine)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/streaming.py (create)

**[CAP-008] Streaming Engine Implementation**

- **Description:** Implement solve_streaming() method returning AsyncGenerator of StreamEvent
- **Acceptance:**
  - [ ] StreamingEngine extends ToolCallingEngine
  - [ ] solve_streaming(task, context=None) -> AsyncGenerator[StreamEvent, None]
  - [ ] Emits plan event after planning decision
  - [ ] Emits token events during LLM execution
  - [ ] Emits result events when sub-tasks complete
  - [ ] Emits error events on failures
  - [ ] Recursive streaming: sub-tasks stream events through parent
  - [ ] Depth and breadcrumb tracking in metadata
  - [ ] All events are JSON-serializable
  - [ ] Comprehensive docstrings with usage examples
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-007
- **Priority:** P0 (Critical)
- **Files:** src/rlm/streaming.py (modify)

**[CAP-009] Token-Level Streaming**

- **Description:** Implement LLM token streaming integration for real-time response generation
- **Acceptance:**
  - [ ] \_execute_leaf_streaming(task, context) -> AsyncGenerator[str, None]
  - [ ] Integration with LLM streaming API (OpenAI, Anthropic)
  - [ ] Token-by-token yield from LLM
  - [ ] Buffer management for partial tokens
  - [ ] Fallback to batch mode if streaming not supported
  - [ ] Time-to-first-token (TTFT) <500ms demonstrated
  - [ ] Streaming maintains same accuracy as batch mode
  - [ ] Error handling during streaming (partial results)
- **Effort:** 3 story points (2-3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-008
- **Priority:** P0 (Critical)
- **Files:** src/rlm/streaming.py (modify)

**[CAP-010] Streaming Integration and Tests**

- **Description:** SSE transport example and comprehensive streaming tests
- **Acceptance:**
  - [ ] FastAPI SSE endpoint example using sse-starlette
  - [ ] EventSourceResponse integration documented
  - [ ] test_streaming.py: Event emission, types, metadata
  - [ ] test_streaming_sse.py: SSE transport end-to-end test
  - [ ] Mock LLM with streaming support for deterministic tests
  - [ ] Test plan → token → result event sequence
  - [ ] Test error event emission on failures
  - [ ] Integration test with real OpenAI streaming
  - [ ] Performance test: TTFT <500ms verified
  - [ ] All tests pass with pytest
  - [ ] Coverage ≥90% for streaming module
- **Effort:** 3 story points (2-3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-009
- **Priority:** P0 (Critical)
- **Files:** examples/streaming_sse.py (create), tests/unit/test_streaming.py (create), tests/integration/test_streaming_sse.py (create)

### Phase 3: Checkpoints & Resume (Weeks 12-16, 20 SP)

**Goal:** Fault-tolerant long-running tasks with state persistence
**Deliverable:** CheckpointableEngine with 95% recovery success and multiple storage backends

#### Tasks

**[CAP-011] Checkpoint Data Model**

- **Description:** Implement Checkpoint dataclass and serialization for execution state
- **Acceptance:**
  - [ ] Checkpoint dataclass with checkpoint_id, task, context, completed_steps, pending_steps, results, timestamp
  - [ ] RLMContext serialization to dict (asdict support)
  - [ ] SharedMemory serialization support
  - [ ] Checkpoint.to_json() and from_json() methods
  - [ ] Timestamp stored as ISO 8601 string
  - [ ] Checkpoint validation: required fields, types
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] All types exported in **all**
  - [ ] Google-style docstrings with examples
- **Effort:** 3 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-001 (RLMContext), PERF-001 (async)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/checkpoints.py (create)

**[CAP-012] Checkpoint Store Protocol**

- **Description:** Define CheckpointStore protocol and in-memory implementation
- **Acceptance:**
  - [ ] CheckpointStore Protocol with save, load, delete, list methods
  - [ ] All methods are async (awaitable)
  - [ ] InMemoryCheckpointStore implementation for testing
  - [ ] Store uses dict[str, Checkpoint] for storage
  - [ ] Thread-safe operations (asyncio.Lock)
  - [ ] Automatic cleanup of old checkpoints (configurable TTL)
  - [ ] List method returns checkpoints sorted by timestamp
  - [ ] Protocol has comprehensive docstrings
- **Effort:** 3 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-011
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/checkpoints.py (modify)

**[CAP-013] Periodic Checkpoint Saving**

- **Description:** Implement automatic checkpoint creation during execution
- **Acceptance:**
  - [ ] CheckpointableEngine extends StreamingEngine
  - [ ] **init** accepts checkpoint_store and checkpoint_interval parameters
  - [ ] Checkpoint saved every N steps (configurable, default 5)
  - [ ] \_save_checkpoint(task, context, decision) method
  - [ ] Checkpoint includes serialized context and memory
  - [ ] Checkpoint tracks completed vs pending sub-tasks
  - [ ] Checkpoint overhead <1% execution time (benchmarked)
  - [ ] Checkpoint save time <100ms (measured)
  - [ ] Failed checkpoint saves logged but don't crash execution
  - [ ] Verbose logging when checkpoints saved
- **Effort:** 5 story points (3-4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-012
- **Priority:** P0 (Critical)
- **Files:** src/rlm/checkpoints.py (modify)

**[CAP-014] Checkpoint Resume Logic**

- **Description:** Implement recovery from saved checkpoints
- **Acceptance:**
  - [ ] solve_with_checkpoints(task, checkpoint_id=None) method
  - [ ] Attempts to load checkpoint by ID if provided
  - [ ] Resumes from checkpoint if found
  - [ ] Falls back to fresh execution if no checkpoint
  - [ ] \_resume_from_checkpoint(checkpoint) -> Output method
  - [ ] Reconstructs RLMContext from checkpoint
  - [ ] Reconstructs SharedMemory state from checkpoint
  - [ ] Re-executes only pending sub-tasks (skips completed)
  - [ ] Synthesizes final result from cached + new results
  - [ ] 95%+ successful recovery rate (tested)
  - [ ] Recovery time proportional to pending work only
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-013
- **Priority:** P0 (Critical)
- **Files:** src/rlm/checkpoints.py (modify)

**[CAP-015] Storage Backend Implementations**

- **Description:** Implement FileCheckpointStore, RedisCheckpointStore, and S3CheckpointStore
- **Acceptance:**
  - [ ] FileCheckpointStore: JSON files in .checkpoints/ directory
  - [ ] FileCheckpointStore uses aiofiles for async I/O
  - [ ] RedisCheckpointStore: Uses redis-py with async support
  - [ ] RedisCheckpointStore handles connection failures gracefully
  - [ ] S3CheckpointStore: Uses aioboto3 for S3 storage
  - [ ] S3CheckpointStore configurable bucket and prefix
  - [ ] All stores implement CheckpointStore protocol
  - [ ] All stores handle serialization/deserialization
  - [ ] All stores have error handling and retries
  - [ ] Configuration examples for each backend
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-012
- **Priority:** P1 (Important)
- **Files:** src/rlm/checkpoints/file_store.py (create), src/rlm/checkpoints/redis_store.py (create), src/rlm/checkpoints/s3_store.py (create)

**[CAP-016] Checkpoint Test Suite and Integration**

- **Description:** Comprehensive checkpoint tests including fault tolerance scenarios
- **Acceptance:**
  - [ ] test_checkpoints.py: Checkpoint serialization, store protocol
  - [ ] test_checkpoint_stores.py: File, Redis, S3 backend tests
  - [ ] test_checkpoint_recovery.py: Resume from checkpoint scenarios
  - [ ] Mock failures during execution to test recovery
  - [ ] Test checkpoint save overhead (<1% verified)
  - [ ] Test checkpoint save time (<100ms verified)
  - [ ] Test 95%+ recovery success rate
  - [ ] Test partial completion recovery (some tasks done)
  - [ ] Test checkpoint cleanup (old checkpoints removed)
  - [ ] Integration test with real Redis (requires Redis server)
  - [ ] Integration test with real S3 (requires AWS credentials)
  - [ ] All tests pass with pytest
  - [ ] Coverage ≥90% for checkpoints module
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CAP-015
- **Priority:** P0 (Critical)
- **Files:** tests/unit/test_checkpoints.py (create), tests/unit/test_checkpoint_stores.py (create), tests/integration/test_checkpoint_recovery.py (create)

## Critical Path

```plaintext
Tool Calling Track (Weeks 5-8):
CAP-002 → CAP-003 → CAP-004 → CAP-005 → CAP-006
  (2h)      (3h)      (3-4h)    (2h)      (3h)
                    [13-14 hours ≈ 4 weeks with research/design]

Streaming Track (Weeks 9-11):
CAP-007 → CAP-008 → CAP-009 → CAP-010
  (1h)      (4h)      (2-3h)    (2-3h)
                    [9-11 hours ≈ 3 weeks with integration/testing]

Checkpoints Track (Weeks 12-16):
CAP-011 → CAP-012 → CAP-013 → CAP-014 → CAP-015 → CAP-016
  (2h)      (2h)      (3-4h)    (4h)      (4h)      (4h)
                    [19-20 hours ≈ 5 weeks with backend integrations]
```

**Bottlenecks:**

- CAP-004: Iterative tool calling loop (most complex tool logic)
- CAP-009: Token streaming integration (LLM backend-specific)
- CAP-014: Checkpoint recovery (state reconstruction complexity)
- CAP-015: Multiple storage backends (requires external services)

**Parallel Tracks:**

- Tool Calling (Weeks 5-8) can be fully independent
- Streaming (Weeks 9-11) starts after tools complete
- Checkpoints (Weeks 12-16) builds on streaming

## Quick Wins (Weeks 5-6)

1. **CAP-002 Tool Protocol** - Enables tool registration and testing (2 hours)
2. **CAP-005 Example Tools** - Demonstrates capabilities early (2 hours)
3. **CAP-007 Streaming Events** - Foundation for streaming (1 hour)

## Risk Mitigation

| Task    | Risk                                 | Mitigation                                   | Contingency                                |
| ------- | ------------------------------------ | -------------------------------------------- | ------------------------------------------ |
| CAP-003 | Tool execution failures crash engine | Comprehensive try/except, error strings      | Fallback to error message instead of crash |
| CAP-004 | Infinite tool calling loops          | Max iterations limit (5), timeout per tool   | Circuit breaker after max iterations       |
| CAP-009 | Streaming adds latency               | Benchmark TTFT <500ms, optimize buffers      | Fallback to batch mode if streaming slow   |
| CAP-013 | Checkpoint serialization fails       | Graceful failure logging, continue execution | Skip checkpoints, rely on retries          |
| CAP-014 | Checkpoint recovery bugs             | Extensive testing, 95% success requirement   | Fresh execution if recovery fails          |
| CAP-015 | External storage failures (Redis/S3) | Retry logic, connection pooling, timeouts    | Fallback to FileCheckpointStore            |

## Testing Strategy

### Automated Testing Tasks

- **CAP-006 Tool Tests** (3 SP) - Mocked tools, deterministic behavior
- **CAP-010 Streaming Tests** (3 SP) - Event sequences, SSE integration
- **CAP-016 Checkpoint Tests** (5 SP) - Recovery scenarios, storage backends

### Quality Gates

- ≥90% code coverage for all capabilities modules
- mypy --strict passes with zero errors
- All unit tests pass in <10 seconds
- Integration tests pass with real services (manual run)
- Performance benchmarks met: TTFT <500ms, checkpoint overhead <1%

### Integration Testing Requirements

- **Tool Calling:** Test with real OpenAI function calling API
- **Streaming:** Test with real SSE transport (FastAPI + browser EventSource)
- **Checkpoints:** Test with Redis (local) and S3 (test bucket)

## Team Allocation

**Backend Engineer #1 (Primary, 1 FTE)**

- Tool calling framework (CAP-002, CAP-003, CAP-004)
- Streaming implementation (CAP-007, CAP-008, CAP-009)
- Checkpoint core logic (CAP-011, CAP-012, CAP-013, CAP-014)

**Backend Engineer #2 (Supporting, 0.5 FTE for Weeks 13-16)**

- Example tools (CAP-005)
- Storage backends (CAP-015)
- Testing support (CAP-006, CAP-010, CAP-016)

## Sprint Planning

**Weeks 5-8: Tool Calling (16 SP capacity, 4 weeks)**

| Week   | Focus            | Story Points | Key Deliverables                              |
| ------ | ---------------- | ------------ | --------------------------------------------- |
| Week 5 | Tool Protocol    | 3 SP         | Tool dataclass, registry, protocol extensions |
| Week 6 | Tool Execution   | 5 SP         | Tool calling engine, error handling           |
| Week 7 | Iterative Loop   | 5 SP         | Multi-turn tool calls, read-before-write      |
| Week 8 | Examples & Tests | 5 SP         | 5+ example tools, comprehensive tests         |

**Weeks 9-11: Streaming (12 SP capacity, 3 weeks)**

| Week    | Focus              | Story Points | Key Deliverables                    |
| ------- | ------------------ | ------------ | ----------------------------------- |
| Week 9  | Streaming Protocol | 2 SP         | StreamEvent types, serialization    |
| Week 10 | Streaming Engine   | 5 SP         | solve_streaming, event emission     |
| Week 11 | Integration        | 6 SP         | Token streaming, SSE example, tests |

**Weeks 12-16: Checkpoints (20 SP capacity, 5 weeks)**

| Week    | Focus               | Story Points | Key Deliverables                     |
| ------- | ------------------- | ------------ | ------------------------------------ |
| Week 12 | Checkpoint Protocol | 6 SP         | Checkpoint dataclass, store protocol |
| Week 13 | Periodic Saving     | 5 SP         | Auto-save during execution           |
| Week 14 | Recovery Logic      | 5 SP         | Resume from checkpoint               |
| Week 15 | Storage Backends    | 5 SP         | File, Redis, S3 implementations      |
| Week 16 | Testing & Polish    | 5 SP         | Comprehensive tests, documentation   |

## Task Sequencing for /implement

Tasks organized by feature for parallel development:

**Tool Calling (Sequential):**

1. CAP-002 (no dependencies)
2. CAP-003 (depends on CAP-002)
3. CAP-004 (depends on CAP-003)
4. CAP-005 (depends on CAP-003, can parallel with CAP-004)
5. CAP-006 (depends on CAP-005)

**Streaming (Sequential):**

1. CAP-007 (depends on PERF-001)
2. CAP-008 (depends on CAP-007)
3. CAP-009 (depends on CAP-008)
4. CAP-010 (depends on CAP-009)

**Checkpoints (Mixed):**

1. CAP-011 (depends on CORE-001, PERF-001)
2. CAP-012 (depends on CAP-011)
3. CAP-013 (depends on CAP-012)
4. CAP-014 (depends on CAP-013)
5. CAP-015 (depends on CAP-012, can parallel with CAP-013/CAP-014)
6. CAP-016 (depends on CAP-015)

## Estimation Method

**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)

**Mapping:**

- 1 SP = ~30 min (simple, well-defined)
- 2 SP = ~1 hour (moderate complexity)
- 3 SP = ~1.5-2 hours (complex, requires design)
- 5 SP = ~3-4 hours (very complex, multiple components)
- 8 SP = ~5-6 hours (epic-level, needs breakdown)

**Assumptions:**

- Developer familiar with async Python patterns
- Developer has experience with tool calling APIs (OpenAI functions)
- Developer understands streaming protocols (AsyncGenerator, SSE)
- External services available for testing (Redis, S3)

## Definition of Done

For each story ticket to be marked COMPLETE:

- [ ] Code written following Python 3.12 standards
- [ ] from **future** import annotations at top of all files
- [ ] All type hints use built-in generics (list[T], dict[K,V], T | None)
- [ ] mypy --strict passes with zero errors
- [ ] Google-style docstrings for all public APIs
- [ ] Unit tests written and passing
- [ ] Test coverage ≥90% for new code
- [ ] Integration tests passing (where applicable)
- [ ] Performance benchmarks met (TTFT, checkpoint overhead)
- [ ] Code reviewed (self-review minimum)
- [ ] No FIXMEs or TODOs in committed code
- [ ] Documentation updated (README examples)

## Appendix

### CSV Export for Project Management

```csv
ID,Title,Description,Estimate_SP,Priority,Assignee,Dependencies,Week
CAP-002,Tool Protocol,Tool dataclass and registry implementation,3,P0,Backend,"CORE-001,PERF-001",5
CAP-003,Tool Execution,Tool call execution engine with error handling,5,P0,Backend,CAP-002,6
CAP-004,Iterative Tool Loop,Multi-turn tool calling implementation,5,P0,Backend,CAP-003,7
CAP-005,Example Tools,5+ example tool integrations,2,P1,Backend,CAP-003,8
CAP-006,Tool Tests,Comprehensive tool calling test suite,3,P0,Backend,CAP-005,8
CAP-007,Streaming Protocol,StreamEvent dataclass and types,2,P0,Backend,PERF-001,9
CAP-008,Streaming Engine,solve_streaming implementation,5,P0,Backend,CAP-007,10
CAP-009,Token Streaming,Token-level LLM streaming integration,3,P0,Backend,CAP-008,11
CAP-010,Streaming Tests,SSE integration and streaming tests,3,P0,Backend,CAP-009,11
CAP-011,Checkpoint Model,Checkpoint dataclass and serialization,3,P0,Backend,"CORE-001,PERF-001",12
CAP-012,Checkpoint Store,CheckpointStore protocol and in-memory impl,3,P0,Backend,CAP-011,12
CAP-013,Periodic Saving,Automatic checkpoint creation,5,P0,Backend,CAP-012,13
CAP-014,Checkpoint Resume,Recovery from saved checkpoints,5,P0,Backend,CAP-013,14
CAP-015,Storage Backends,File/Redis/S3 checkpoint stores,5,P1,Backend,CAP-012,15
CAP-016,Checkpoint Tests,Checkpoint recovery and storage tests,5,P0,Backend,CAP-015,16
```

### Success Metrics

**Technical:**

- Zero mypy errors in strict mode
- ≥90% test coverage across all capabilities
- 80% tool call success rate (real usage)
- <500ms TTFT for streaming responses
- <1% checkpoint overhead in execution time
- 95%+ successful checkpoint recovery

**Business:**

- 30% adoption of tool calling within 3 months
- 80% of web apps enable streaming responses
- 20% adoption of checkpoints for long-running tasks (>5 min)
- 20x perceived responsiveness improvement from streaming
- 5+ example tool integrations available

**Performance Benchmarks:**

- Tool execution timeout: 30s (configurable)
- Max tool iterations: 5 per task
- TTFT streaming: <500ms
- Checkpoint save time: <100ms
- Checkpoint overhead: <1% execution time
- Recovery success rate: ≥95%

### Feature Adoption Targets

| Feature      | 1 Month | 3 Months | 6 Months |
| ------------ | ------- | -------- | -------- |
| Tool Calling | 10%     | 30%      | 50%      |
| Streaming    | 40%     | 80%      | 95%      |
| Checkpoints  | 5%      | 20%      | 35%      |

### External Dependencies

**Required:**

- aiofiles (async file I/O for FileCheckpointStore)
- sse-starlette (SSE transport for FastAPI example)

**Optional:**

- redis (RedisCheckpointStore)
- aioboto3 (S3CheckpointStore)
- httpx (HTTP request tool example)
- requests (fallback for sync tools)

### Documentation Deliverables

- [ ] Tool calling API reference
- [ ] Example tools cookbook
- [ ] Streaming integration guide
- [ ] SSE transport example (FastAPI)
- [ ] Checkpoint configuration guide
- [ ] Storage backend setup (Redis, S3)
- [ ] Performance tuning guide
- [ ] Migration guide from batch to streaming
