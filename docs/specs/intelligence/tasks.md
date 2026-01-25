# Tasks: Intelligence Layer (INTEL-001)

**From:** `intelligence/spec.md` + `intelligence/plan.md`
**Timeline:** 2 days (Sprint 2, Day 3-4)
**Team:** 1 backend engineer
**Created:** 2026-01-25

## Summary

- Total tasks: 5 stories
- Estimated effort: 13 story points
- Critical path duration: 2 days
- Key risks: Invalid agent assignments, cost optimization not achieved, planner assigns all tasks to expensive agent

## Phase Breakdown

### Phase 1: Prompts and Planning TypedDicts (Day 3 Morning, 2 SP)

**Goal:** Establish prompt templates and enhanced type system for multi-agent routing
**Deliverable:** Complete prompts.py module and extended types.py

#### Tasks

**[INTEL-002] Prompts and Planning TypedDicts**

- **Description:** Implement PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT, SubTask TypedDict, enhanced PlannerDecision, and AgentConfig dataclass
- **Acceptance:**
  - [ ] PLANNER_SYSTEM_PROMPT with {available_agents} placeholder
  - [ ] Prompt includes EXECUTE vs RECURSE decision guidelines
  - [ ] Prompt includes agent assignment instructions
  - [ ] Prompt includes JSON response format with sub_tasks[].assigned_agent field
  - [ ] SYNTHESIZER_SYSTEM_PROMPT with {results} placeholder
  - [ ] SubTask TypedDict with description: str and assigned_agent: str | None fields
  - [ ] PlannerDecision TypedDict with thoughts, decision, sub_tasks fields
  - [ ] AgentConfig dataclass with name, llm_callable, description, system_prompt fields
  - [ ] validate_planner_decision() function validates JSON against schema
  - [ ] Validation raises JSONValidationError for missing/invalid fields
  - [ ] from **future** import annotations at top of files
  - [ ] mypy --strict passes with zero errors
  - [ ] Google-style docstrings for all public functions
- **Effort:** 2 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-001 (complete foundation required)
- **Priority:** P0 (Blocker)
- **Files:** src/rlm/prompts.py (create), src/rlm/types.py (extend)

### Phase 2: Agent Registry and Routing Logic (Day 3 Afternoon + Day 4 Morning, 7 SP)

**Goal:** Implement multi-agent registry and intelligent routing in RecursiveEngine
**Deliverable:** Extended RecursiveEngine with agent-aware execution

#### Tasks

**[INTEL-003] Agent Registry and Routing Logic**

- **Description:** Extend RecursiveEngine with agents parameter, router_model selection, \_get_agent() fallback, \_plan_with_agents(), and \_recurse_with_agents() methods
- **Acceptance:**
  - [ ] RecursiveEngine.**init** accepts agents: dict[str, LLMCaller] parameter
  - [ ] RecursiveEngine.**init** accepts router_model: str parameter (default: "planner")
  - [ ] Backward compatibility: llm parameter creates single-agent registry
  - [ ] Validate router_model exists in agents registry on init
  - [ ] \_get_agent(agent_name) retrieves agent by name
  - [ ] \_get_agent() falls back to router_model if agent_name not found
  - [ ] \_get_agent() logs warning when fallback occurs
  - [ ] \_plan_with_agents() formats PLANNER_SYSTEM_PROMPT with agent descriptions
  - [ ] \_plan_with_agents() calls router_model with planner prompt
  - [ ] \_plan_with_agents() retries up to 3 times on InvalidJSONError
  - [ ] \_plan_with_agents() falls back to EXECUTE mode after max retries
  - [ ] \_recurse_with_agents() routes sub-tasks to assigned agents
  - [ ] \_recurse_with_agents() creates child contexts with assigned_agent
  - [ ] solve() initializes context with router_model as active_agent
  - [ ] solve() uses context.active_agent for EXECUTE mode
  - [ ] All methods have comprehensive docstrings with examples
  - [ ] Verbose logging when verbose=True
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** INTEL-002
- **Priority:** P0 (Critical)
- **Files:** src/rlm/engine.py (extend)

**[INTEL-004] Context Extension for Active Agent**

- **Description:** Extend RLMContext with active_agent field and update create_child() to support agent assignment
- **Acceptance:**
  - [ ] RLMContext.active_agent: str field added
  - [ ] RLMContext.create_child() accepts assigned_agent: str | None parameter
  - [ ] create_child() uses assigned_agent if provided, inherits active_agent if None
  - [ ] Dataclass remains frozen (mutation raises FrozenInstanceError)
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] Google-style docstrings updated
  - [ ] **init**.py exports updated with new types
- **Effort:** 2 story points (1.5 hours)
- **Owner:** Backend Engineer
- **Dependencies:** INTEL-002
- **Priority:** P0 (Critical)
- **Files:** src/rlm/memory.py (extend), src/rlm/**init**.py (update exports)

### Phase 3: Testing and Validation (Day 4 Afternoon, 4 SP)

**Goal:** Achieve ≥90% test coverage and validate cost optimization claims
**Deliverable:** Complete test suite demonstrating 40%+ cost reduction

#### Tasks

**[INTEL-005] Planning Unit Tests**

- **Description:** Implement unit tests for prompts, JSON validation, agent registry, and routing logic with mocked agents
- **Acceptance:**
  - [ ] test_prompts.py: Validate prompt templates format correctly
  - [ ] test_prompts.py: Test {available_agents} placeholder substitution
  - [ ] test_validation.py: Test validate_planner_decision() with valid JSON
  - [ ] test_validation.py: Test validate_planner_decision() raises on missing fields
  - [ ] test_validation.py: Test validate_planner_decision() raises on invalid types
  - [ ] test_agent_registry.py: Test multi-agent routing with mock planner + mock worker
  - [ ] test_agent_registry.py: Verify correct agents called for sub-tasks
  - [ ] test_agent_registry.py: Test agent fallback when assigned_agent missing
  - [ ] test_agent_registry.py: Test backward compatibility with single llm parameter
  - [ ] test_agent_registry.py: Test router_model validation on init
  - [ ] Mock agents for deterministic tests
  - [ ] All tests pass with pytest
  - [ ] Coverage ≥90% for prompts, validation, routing logic
- **Effort:** 2 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** INTEL-003, INTEL-004
- **Priority:** P0 (Critical)
- **Files:** tests/unit/test_prompts.py, tests/unit/test_validation.py, tests/unit/test_agent_registry.py (all create)

**[INTEL-006] Multi-Agent Integration Tests**

- **Description:** Implement integration tests with real OpenAI API to validate multi-agent routing and cost optimization
- **Acceptance:**
  - [ ] test_multi_agent.py: Real multi-provider test (OpenAI planner + worker)
  - [ ] test_multi_agent.py: Test task decomposition with assigned agents
  - [ ] test_multi_agent.py: Verify correct agents execute sub-tasks
  - [ ] test_multi_agent.py: Test synthesis of results from multiple agents
  - [ ] test_cost_optimization.py: Demonstrate 40%+ cost reduction
  - [ ] test_cost_optimization.py: Track expensive_cost vs cheap_cost
  - [ ] test_cost_optimization.py: Calculate baseline cost (all expensive model)
  - [ ] test_cost_optimization.py: Assert savings_pct >= 40%
  - [ ] Tests use pytest markers (@pytest.mark.integration)
  - [ ] Tests can be skipped if OPENAI_API_KEY not set
  - [ ] All integration tests pass with real LLM
  - [ ] Cost metrics logged for manual inspection
- **Effort:** 2 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** INTEL-005
- **Priority:** P0 (Critical)
- **Files:** tests/integration/test_multi_agent.py, tests/integration/test_cost_optimization.py (both create)

## Critical Path

```plaintext
INTEL-002 → INTEL-003 → INTEL-004 → INTEL-005 → INTEL-006
  (2h)       (4h)        (1.5h)       (2h)        (2h)
                      [11.5 hours ≈ 2 days with buffer]
```

**Bottlenecks:**

- INTEL-003: Most complex implementation (agent registry + routing)
- INTEL-006: Real API tests for cost validation

**Parallel Tracks:**

- INTEL-003 and INTEL-004 can run in parallel after INTEL-002

## Quick Wins (Day 3 Morning)

1. **INTEL-002 Prompts and Types** - Unblocks routing implementation (2 hours)

## Risk Mitigation

| Task      | Risk                                   | Mitigation                                                         | Contingency                                   |
| --------- | -------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------- |
| INTEL-003 | Invalid agent assignments              | Validate assigned_agent against registry, fallback to router_model | Log warnings and degrade gracefully           |
| INTEL-003 | Planner assigns all to expensive agent | Prompt engineering: explicitly prefer cheap agents for execution   | Add examples in prompt showing proper routing |
| INTEL-006 | Cost optimization not achieved         | Clear planner prompts, integration tests validating cost reduction | Document actual savings, adjust expectations  |
| INTEL-006 | OpenAI API rate limits                 | Use exponential backoff, mark tests as flaky                       | Mock OpenAI for CI, run integration manually  |

## Testing Strategy

### Automated Testing Tasks

- **INTEL-005 Unit Tests** (2 SP) - Mocked agents, deterministic routing
- **INTEL-006 Integration Tests** (2 SP) - Real OpenAI, cost validation

### Quality Gates

- ≥90% code coverage (pytest --cov=src/rlm --cov-fail-under=90)
- mypy --strict passes with zero errors
- All unit tests pass in <5 seconds
- Integration tests demonstrate 40%+ cost reduction
- Planner produces valid JSON 95%+ of time

## Team Allocation

**Backend Engineer (1 FTE)**

- Prompts and enhanced types (INTEL-002)
- Agent registry and routing (INTEL-003)
- Context extension (INTEL-004)
- Testing (INTEL-005, INTEL-006)

## Sprint Planning

**Sprint 2 (2 days, 13 SP capacity)**

| Day   | Focus              | Story Points | Key Deliverables                                           |
| ----- | ------------------ | ------------ | ---------------------------------------------------------- |
| Day 3 | Prompts + Registry | 7 SP         | prompts.py, extended types.py, agent registry in engine.py |
| Day 4 | Context + Testing  | 6 SP         | extended RLMContext, complete test suite, cost validation  |

## Task Sequencing for /implement

Tasks must be executed in strict dependency order:

1. INTEL-002 (depends on CORE-001 complete)
2. INTEL-003 (depends on INTEL-002)
3. INTEL-004 (depends on INTEL-002) - Can run parallel with INTEL-003
4. INTEL-005 (depends on INTEL-003, INTEL-004)
5. INTEL-006 (depends on INTEL-005)

## Estimation Method

**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)

**Mapping:**

- 1 SP = ~30 min (simple, well-defined)
- 2 SP = ~1-2 hours (moderate complexity)
- 3 SP = ~2-3 hours (complex, requires design)
- 5 SP = ~4 hours (very complex, multiple components)

**Assumptions:**

- CORE-001 is complete and passing all tests
- Developer familiar with multi-agent architectures
- Developer has OpenAI API access for integration tests
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
- [ ] Integration tests passing (for INTEL-006)
- [ ] Cost optimization demonstrated (40%+ reduction)
- [ ] Code reviewed (self-review minimum)
- [ ] No FIXMEs or TODOs in committed code

## Appendix

### CSV Export for Project Management

```csv
ID,Title,Description,Estimate_SP,Priority,Assignee,Dependencies,Day
INTEL-002,Prompts and Planning TypedDicts,PLANNER_SYSTEM_PROMPT and enhanced types,2,P0,Backend,CORE-001,3
INTEL-003,Agent Registry and Routing Logic,Multi-agent RecursiveEngine extensions,5,P0,Backend,INTEL-002,3-4
INTEL-004,Context Extension for Active Agent,RLMContext.active_agent field,2,P0,Backend,INTEL-002,4
INTEL-005,Planning Unit Tests,Unit tests for prompts and routing,2,P0,Backend,"INTEL-003,INTEL-004",4
INTEL-006,Multi-Agent Integration Tests,Real API tests and cost validation,2,P0,Backend,INTEL-005,4
```

### Success Metrics

**Technical:**

- Zero mypy errors in strict mode
- ≥90% test coverage
- Planner produces valid JSON 95%+ of time
- All agent routing fallbacks logged

**Business:**

- 40-60% API cost reduction demonstrated
- Supports heterogeneous providers (OpenAI + Anthropic + Ollama)
- Enables PERF-001 and CAP-001 work to begin
- Multi-agent foundation stable for future enhancements

### Integration with CORE-001

**Extends existing modules:**

- types.py: Add SubTask, enhanced PlannerDecision, AgentConfig
- engine.py: Add agents parameter, router_model, \_plan_with_agents(), \_recurse_with_agents()
- memory.py: Add active_agent field to RLMContext
- **init**.py: Export new types

**New modules:**

- prompts.py: PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT

**Backward compatibility:**

- Single llm parameter still works (creates single-agent registry)
- Existing tests continue to pass
