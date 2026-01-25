# Tasks: Core Foundation (CORE-001)

**From:** `core/spec.md` + `core/plan.md`
**Timeline:** 2 days (Sprint 1)
**Team:** 1 backend engineer
**Created:** 2026-01-25

## Summary

- Total tasks: 7 stories
- Estimated effort: 19 story points
- Critical path duration: 2 days
- Key risks: Infinite recursion, invalid JSON from LLM, context overflow

## Phase Breakdown

### Phase 1: Foundation Setup (Day 1 Morning, 8 SP)

**Goal:** Establish type system, memory management, and utilities
**Deliverable:** Complete types.py, memory.py, utils.py modules

#### Tasks

**[CORE-002] Types and Protocol Implementation**

- **Description:** Implement OpenResponses protocol types (Input, Item, Output), LLMCaller protocol, PlannerDecision TypedDict, and custom exception hierarchy
- **Acceptance:**
  - [ ] Input, Item, Output TypedDicts defined with proper field types
  - [ ] LLMCaller Protocol with **call** signature
  - [ ] PlannerDecision TypedDict with decision and sub_tasks fields
  - [ ] Custom exceptions: RLMError, RecursionDepthError, ExecutionError, InvalidJSONError
  - [ ] All types use Python 3.12+ syntax (list[T], dict[K,V], T | None)
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] All public types exported in **all**
- **Effort:** 3 story points (1.5 hours)
- **Owner:** Backend Engineer
- **Dependencies:** None
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/types.py (create)

**[CORE-003] Memory and Context Implementation**

- **Description:** Implement SharedMemory class for variable offloading and RLMContext frozen dataclass for execution state tracking
- **Acceptance:**
  - [ ] SharedMemory class with store(content) -> ref_id method
  - [ ] SharedMemory.resolve(ref_id) -> content method
  - [ ] Reference IDs use format "ref::{uuid8}" for readability
  - [ ] RLMContext frozen dataclass with task_id, parent_id, depth, breadcrumbs, memory_ref
  - [ ] RLMContext.create_child(task_id, step_description) -> RLMContext method
  - [ ] Breadcrumbs are immutable tuple (not list)
  - [ ] Dataclass is frozen (mutation raises FrozenInstanceError)
  - [ ] Google-style docstrings for all public methods
- **Effort:** 3 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-002 (needs types for type hints)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/memory.py (create)

**[CORE-004] Utility Functions**

- **Description:** Implement safe JSON parsing with markdown stripping, schema validation, and optional MIMIR-compatible trace logging
- **Acceptance:**
  - [ ] safe_parse_json(content: str) -> dict function
  - [ ] Strips markdown code blocks (`json ... `)
  - [ ] Uses json.loads exclusively (never eval)
  - [ ] Raises InvalidJSONError with descriptive message on failure
  - [ ] TraceObject TypedDict for MIMIR compatibility
  - [ ] create_trace(context, decision, result) -> TraceObject helper
  - [ ] All functions have comprehensive docstrings with examples
  - [ ] Edge cases tested: empty string, invalid JSON, nested objects
- **Effort:** 2 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-002 (needs InvalidJSONError exception)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/utils.py (create)

### Phase 2: Core Engine Implementation (Day 1 Afternoon + Day 2 Morning, 6 SP)

**Goal:** Implement recursive execution engine with depth limits and error handling
**Deliverable:** Complete RecursiveEngine class with solve() method

#### Tasks

**[CORE-005] Recursive Engine Core Logic**

- **Description:** Implement RecursiveEngine class with solve(), \_decide_strategy(), \_plan(), \_recurse(), \_execute_leaf(), and \_synthesize() methods
- **Acceptance:**
  - [ ] RecursiveEngine **init** with llm, max_depth, max_steps, verbose parameters
  - [ ] solve(task, context=None) -> Output public method
  - [ ] \_decide_strategy(task, context) calls planner LLM with structured prompt
  - [ ] \_plan(task, context) returns PlannerDecision from JSON output
  - [ ] \_recurse(task, context, decision) executes sub-tasks sequentially
  - [ ] \_execute_leaf(task, context) calls LLM for atomic task execution
  - [ ] \_synthesize(results) aggregates sub-task results
  - [ ] RecursionDepthError raised when depth >= max_depth
  - [ ] ExecutionError raised on LLM call failures with context
  - [ ] Retry logic for invalid JSON (up to 3 attempts)
  - [ ] Verbose logging when verbose=True
  - [ ] All methods have comprehensive docstrings
- **Effort:** 5 story points (3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-002, CORE-003, CORE-004
- **Priority:** P0 (Critical)
- **Files:** src/rlm/engine.py (create)

**[CORE-006] Package Exports and Integration**

- **Description:** Configure package exports in **init**.py for public API surface
- **Acceptance:**
  - [ ] Exports RecursiveEngine from engine module
  - [ ] Exports RLMContext, SharedMemory from memory module
  - [ ] Exports Input, Output, Item, LLMCaller from types module
  - [ ] Exports safe_parse_json from utils module
  - [ ] **all** list defines public API
  - [ ] Package can be imported with: from rlm import RecursiveEngine
  - [ ] No internal modules exposed (types, memory, utils are internal)
- **Effort:** 1 story point (30 min)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-005
- **Priority:** P0 (Critical)
- **Files:** src/rlm/**init**.py (create)

### Phase 3: Testing and Validation (Day 2 Afternoon, 5 SP)

**Goal:** Achieve ≥90% test coverage with unit and integration tests
**Deliverable:** Complete test suite passing with mypy --strict

#### Tasks

**[CORE-007] Unit Test Suite**

- **Description:** Implement comprehensive unit tests for types, memory, utils, and engine with mocked LLM
- **Acceptance:**
  - [ ] test_types.py: TypedDict validation, Protocol adherence
  - [ ] test_memory.py: SharedMemory store/resolve, RLMContext immutability, create_child
  - [ ] test_utils.py: JSON parsing safety (valid, invalid, edge cases), markdown stripping
  - [ ] test_engine.py: Recursion logic, depth limits, error handling, retry logic
  - [ ] Mock LLM for deterministic tests
  - [ ] Test infinite recursion detection (always-recurse mock)
  - [ ] Test invalid JSON handling (malformed JSON mock)
  - [ ] Test state immutability (frozen dataclass mutation attempt)
  - [ ] Test large document offloading (50k+ character strings)
  - [ ] All tests pass with pytest
  - [ ] Coverage ≥90% for core module (pytest --cov)
- **Effort:** 3 story points (3 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-006
- **Priority:** P0 (Critical)
- **Files:** tests/unit/test_types.py, tests/unit/test_memory.py, tests/unit/test_utils.py, tests/unit/test_engine.py (all create)

**[CORE-008] Integration Test Suite**

- **Description:** Implement integration tests with real OpenAI API to validate end-to-end execution
- **Acceptance:**
  - [ ] test_openai.py: Real OpenAI integration (requires API key)
  - [ ] Test simple task execution (EXECUTE decision)
  - [ ] Test recursive task decomposition (RECURSE with 2-3 sub-tasks)
  - [ ] Test depth limit enforcement (task requiring depth > max_depth)
  - [ ] Test variable offloading with large documents
  - [ ] Test result synthesis from multiple sub-tasks
  - [ ] Tests use pytest markers (@pytest.mark.integration)
  - [ ] Tests can be skipped if OPENAI_API_KEY not set
  - [ ] All integration tests pass with real LLM
  - [ ] Results logged for manual inspection
- **Effort:** 2 story points (1.5 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-007
- **Priority:** P0 (Critical)
- **Files:** tests/integration/test_openai.py (create)

## Critical Path

```plaintext
CORE-002 → CORE-003 → CORE-004 → CORE-005 → CORE-006 → CORE-007 → CORE-008
  (1.5h)    (2h)       (2h)       (3h)       (0.5h)     (3h)       (1.5h)
                            [13.5 hours ≈ 2 days with buffer]
```

**Bottlenecks:**

- CORE-005: Most complex implementation (recursion logic)
- CORE-007: Comprehensive test coverage required

**Parallel Tracks:**

- None - all tasks sequential due to dependencies

## Quick Wins (Day 1)

1. **CORE-002 Types Implementation** - Unblocks all other work (1.5 hours)
2. **CORE-003 Memory** - Core data structures complete (2 hours)
3. **CORE-004 Utils** - JSON safety demonstrated (2 hours)

## Risk Mitigation

| Task     | Risk                    | Mitigation                                      | Contingency                                  |
| -------- | ----------------------- | ----------------------------------------------- | -------------------------------------------- |
| CORE-005 | Infinite recursion bugs | Hard max_depth enforcement, comprehensive tests | Add emergency circuit breaker in production  |
| CORE-005 | Invalid JSON from LLM   | Retry logic (3 attempts), safe parsing          | Fallback to EXECUTE mode on parse failure    |
| CORE-007 | Test coverage < 90%     | Focus on critical paths first, add edge cases   | Document uncovered code as known limitation  |
| CORE-008 | OpenAI API rate limits  | Use exponential backoff, mark tests as flaky    | Mock OpenAI for CI, run integration manually |

## Testing Strategy

### Automated Testing Tasks

- **CORE-007 Unit Tests** (3 SP) - Mocked LLM, deterministic behavior
- **CORE-008 Integration Tests** (2 SP) - Real OpenAI, end-to-end validation

### Quality Gates

- ≥90% code coverage (pytest --cov=src/rlm --cov-fail-under=90)
- mypy --strict passes with zero errors
- All unit tests pass in <5 seconds
- Integration tests pass with real OpenAI (manual run)

## Team Allocation

**Backend Engineer (1 FTE)**

- Types and protocols (CORE-002)
- Memory management (CORE-003)
- Utilities (CORE-004)
- Recursive engine (CORE-005)
- Package integration (CORE-006)
- Testing (CORE-007, CORE-008)

## Sprint Planning

**Sprint 1 (2 days, 19 SP capacity)**

| Day   | Focus      | Story Points | Key Deliverables                                  |
| ----- | ---------- | ------------ | ------------------------------------------------- |
| Day 1 | Foundation | 8 SP         | types.py, memory.py, utils.py, engine.py skeleton |
| Day 2 | Completion | 11 SP        | Complete engine.py, **init**.py, full test suite  |

## Task Sequencing for /implement

Tasks must be executed in strict dependency order:

1. CORE-002 (no dependencies)
2. CORE-003 (depends on CORE-002)
3. CORE-004 (depends on CORE-002)
4. CORE-005 (depends on CORE-002, CORE-003, CORE-004)
5. CORE-006 (depends on CORE-005)
6. CORE-007 (depends on CORE-006)
7. CORE-008 (depends on CORE-007)

## Estimation Method

**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)

**Mapping:**

- 1 SP = ~30 min (simple, well-defined)
- 2 SP = ~1 hour (moderate complexity)
- 3 SP = ~1.5-2 hours (complex, requires design)
- 5 SP = ~3-4 hours (very complex, multiple components)

**Assumptions:**

- Developer familiar with Python 3.12 typing
- Developer has experience with LLM APIs
- Development environment already configured (uv installed)

## Definition of Done

For each story ticket to be marked COMPLETE:

- [ ] Code written following Python 3.12 standards
- [ ] from **future** import annotations at top of all files
- [ ] All type hints use built-in generics (list[T], dict[K,V], T | None)
- [ ] mypy --strict passes with zero errors
- [ ] Google-style docstrings for all public APIs
- [ ] Unit tests written and passing
- [ ] Test coverage ≥90% for new code
- [ ] Integration tests passing (for CORE-008)
- [ ] Code reviewed (self-review minimum)
- [ ] No FIXMEs or TODOs in committed code

## Appendix

### CSV Export for Project Management

```csv
ID,Title,Description,Estimate_SP,Priority,Assignee,Dependencies,Day
CORE-002,Types and Protocol,OpenResponses types and LLMCaller protocol,3,P0,Backend,,1
CORE-003,Memory and Context,SharedMemory and RLMContext implementation,3,P0,Backend,CORE-002,1
CORE-004,Utility Functions,Safe JSON parsing and trace logging,2,P0,Backend,CORE-002,1
CORE-005,Recursive Engine,Core recursion logic implementation,5,P0,Backend,"CORE-002,CORE-003,CORE-004",1-2
CORE-006,Package Exports,__init__.py public API configuration,1,P0,Backend,CORE-005,2
CORE-007,Unit Test Suite,Comprehensive unit tests with mocks,3,P0,Backend,CORE-006,2
CORE-008,Integration Tests,OpenAI integration end-to-end tests,2,P0,Backend,CORE-007,2
```

### Success Metrics

**Technical:**

- Zero mypy errors in strict mode
- ≥90% test coverage
- All recursion depth limits respected
- Zero infinite loops in production

**Business:**

- Enables INTEL-001, PERF-001, CAP-001 work to begin
- Foundation stable enough for multi-week development on top
- Backend-agnostic design validated (works with OpenAI, Anthropic, Ollama)
