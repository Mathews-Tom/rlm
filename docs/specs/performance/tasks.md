# Tasks: Performance Enhancements (PERF-001)

**From:** `performance/spec.md` + `performance/plan.md`
**Timeline:** 4 weeks (Sprint 2-5)
**Team:** 1 backend engineer
**Created:** 2026-01-25

## Summary

- Total tasks: 9 stories
- Estimated effort: 44 story points
- Critical path duration: 4 weeks
- Key risks: Async breaking changes, Redis unavailability, OpenTelemetry overhead

## Phase Breakdown

### Phase 1: Async Refactor - Core Infrastructure (Week 1, 13 SP)

**Goal:** Convert synchronous engine to async/await with backward compatibility
**Deliverable:** AsyncRecursiveEngine with asyncio.gather() parallel execution

#### Tasks

**[PERF-002] AsyncLLMCaller Protocol and Type System**

- **Description:** Define async protocol for LLM callers, update type system to support async operations, and create async version of core types
- **Acceptance:**
  - [ ] AsyncLLMCaller Protocol with async **call** signature
  - [ ] Protocol accepts (messages: list[dict], \*\*kwargs) -> str
  - [ ] Async variants of Input, Output, Item TypedDicts (if needed)
  - [ ] Backward compatibility with synchronous LLMCaller protocol
  - [ ] Type hints use async/await syntax (Awaitable, Coroutine)
  - [ ] from **future** import annotations at top of file
  - [ ] mypy --strict passes with zero errors
  - [ ] All public types exported in **all**
  - [ ] Google-style docstrings with async usage examples
- **Effort:** 3 story points (2 hours)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-001 (complete)
- **Priority:** P1 (Blocker)
- **Files:** src/rlm/types.py (edit), src/rlm/async_types.py (create)

**[PERF-003] AsyncRecursiveEngine Core Implementation**

- **Description:** Implement async version of RecursiveEngine with async solve(), \_plan_async(), \_execute_leaf_async(), and \_recurse_async() methods
- **Acceptance:**
  - [ ] AsyncRecursiveEngine class inherits core patterns from RecursiveEngine
  - [ ] async solve(task, context) -> Output main entry point
  - [ ] async \_plan_async(task, context) calls async LLM for planning
  - [ ] async \_execute_leaf_async(task, context) for atomic execution
  - [ ] async \_recurse_async(task, context, decision) for decomposition
  - [ ] async \_synthesize_async(results) aggregates sub-task outputs
  - [ ] All internal methods use async/await syntax
  - [ ] asyncio.Semaphore for max_concurrency rate limiting
  - [ ] RecursionDepthError raised when depth >= max_depth
  - [ ] ExecutionError raised on async LLM failures
  - [ ] Retry logic for invalid JSON (up to 3 attempts)
  - [ ] Verbose logging with async-safe print statements
  - [ ] Google-style async docstrings for all methods
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-002
- **Priority:** P1 (Critical)
- **Files:** src/rlm/async_engine.py (create)

**[PERF-004] Parallel Execution with asyncio.gather()**

- **Description:** Implement concurrent sub-task execution using asyncio.gather() for 8-10x throughput improvement
- **Acceptance:**
  - [ ] \_recurse_async() uses asyncio.gather() for parallel execution
  - [ ] All sub-tasks execute concurrently (not sequentially)
  - [ ] Semaphore rate limiting prevents resource exhaustion
  - [ ] max_concurrency parameter controls parallel task limit (default: 10)
  - [ ] Exceptions in sub-tasks propagate correctly via gather()
  - [ ] solve_sync(task) -> Output wrapper for backward compatibility
  - [ ] solve_sync() uses asyncio.run() to execute async code
  - [ ] Backward compatibility verified with existing test suite
  - [ ] Async overhead <100ms per call (measured in benchmarks)
  - [ ] 8-10x throughput improvement demonstrated (10 concurrent tasks)
  - [ ] Integration with RLMContext.create_child() for parallel contexts
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-003
- **Priority:** P1 (Critical)
- **Files:** src/rlm/async_engine.py (edit)

### Phase 2: Multi-Layer Caching (Week 2, 11 SP)

**Goal:** Implement L1 in-memory + L2 Redis semantic caching for 60% cost reduction
**Deliverable:** CachedAsyncEngine with 40-50% cache hit rate

#### Tasks

**[PERF-005] L1 In-Memory LRU Cache**

- **Description:** Implement in-memory LRU cache with exact-match lookups using task + agent + depth as cache key
- **Acceptance:**
  - [ ] CachedAsyncEngine class extends AsyncRecursiveEngine
  - [ ] \_compute_cache_key(task, context) -> str hash function
  - [ ] Cache key includes task, active_agent, and depth
  - [ ] SHA256 hash truncated to 16 characters for readability
  - [ ] L1 cache as dict[str, Output] with LRU eviction
  - [ ] l1_size parameter controls max cache entries (default: 1000)
  - [ ] LRU eviction removes oldest entry when size exceeded
  - [ ] Cache lookup in solve() before LLM call
  - [ ] Cache storage after successful LLM execution
  - [ ] Verbose logging for cache hits/misses
  - [ ] Cache metrics tracked: \_cache_hits, \_cache_misses counters
  - [ ] get_cache_stats() -> dict returns hit_rate, l1_size
- **Effort:** 5 story points (3.5 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-004
- **Priority:** P1 (Critical)
- **Files:** src/rlm/caching.py (create)

**[PERF-006] L2 Redis Semantic Cache Integration**

- **Description:** Add optional Redis-based semantic similarity caching using redisvl library for cross-session cache sharing
- **Acceptance:**
  - [ ] Optional redis_url parameter in **init** (None disables L2)
  - [ ] redisvl.extensions.llmcache.SemanticCache integration
  - [ ] REDIS_AVAILABLE flag with graceful fallback if import fails
  - [ ] cache_threshold parameter for similarity matching (default: 0.85)
  - [ ] L2 cache checked on L1 miss before LLM call
  - [ ] Semantic similarity matching with vector embeddings
  - [ ] L2 cache hit promotes result to L1 for faster subsequent access
  - [ ] L2 cache stores results with metadata (agent, depth, timestamp)
  - [ ] TTL parameter for cache expiration (default: 3600s = 1 hour)
  - [ ] Graceful degradation to L1-only if Redis unavailable
  - [ ] Try/except around Redis operations with warning logs
  - [ ] get_cache_stats() includes l2_enabled boolean
  - [ ] 40-50% cache hit rate demonstrated in integration tests
  - [ ] 15x speedup for cached responses measured in benchmarks
- **Effort:** 6 story points (5 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-005
- **Priority:** P1 (Critical)
- **Files:** src/rlm/caching.py (edit), pyproject.toml (add redisvl optional dependency)

### Phase 3: OpenTelemetry Observability (Week 3, 10 SP)

**Goal:** Add distributed tracing with <5% overhead for production debugging
**Deliverable:** InstrumentedAsyncEngine with 100% execution coverage

#### Tasks

**[PERF-007] OpenTelemetry Span Creation and Attributes**

- **Description:** Implement automatic span creation for all engine operations (solve, plan, execute) with rich context attributes
- **Acceptance:**
  - [ ] InstrumentedAsyncEngine class extends CachedAsyncEngine
  - [ ] Optional enable_tracing parameter in **init** (default: True)
  - [ ] service_name parameter for trace identification (default: "py-rlm")
  - [ ] OpenTelemetry TracerProvider configuration on initialization
  - [ ] OTEL_AVAILABLE flag with graceful fallback if import fails
  - [ ] tracer.start_as_current_span() for solve() execution
  - [ ] Span attributes: task (truncated to 100 chars), depth, active_agent
  - [ ] Span attributes: duration_ms, status (success/error), output_length
  - [ ] Nested spans for \_plan_async(), \_execute_leaf_async()
  - [ ] Cache hit/miss recorded in span attributes
  - [ ] Exception recording with span.record_exception(e)
  - [ ] span.set_status() for success/error states
  - [ ] Trace context propagation for distributed systems
  - [ ] <5% overhead measured in benchmarks (tracing vs no tracing)
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-006
- **Priority:** P1 (Critical)
- **Files:** src/rlm/observability.py (create)

**[PERF-008] Multi-Platform Exporter Integration**

- **Description:** Add OTLP exporter support for Datadog, Grafana, Jaeger, and New Relic with configuration guides
- **Acceptance:**
  - [ ] OTLPSpanExporter from opentelemetry.exporter.otlp.proto.grpc
  - [ ] Environment variable configuration (OTEL_EXPORTER_OTLP_ENDPOINT)
  - [ ] Support for Datadog agent (localhost:8126)
  - [ ] Support for Grafana Tempo (tempo:4317)
  - [ ] Support for Jaeger (jaeger:4317)
  - [ ] Support for New Relic (otlp.nr-data.net:4317)
  - [ ] Integration guide in docs/observability.md
  - [ ] Example docker-compose.yml for local testing
  - [ ] Example trace screenshots from Datadog/Grafana
  - [ ] Sampling configuration (sample 10% in production)
  - [ ] Batch span processor for performance
  - [ ] Resource attributes (service.name, service.version, host.name)
  - [ ] 100% of executions traced when enabled
  - [ ] Zero crashes when exporter unavailable
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-007
- **Priority:** P1 (Critical)
- **Files:** src/rlm/observability.py (edit), docs/observability.md (create)

### Phase 4: Cost Tracking and Optimization (Week 4, 10 SP)

**Goal:** Add token usage tracking and cost estimation for budget optimization
**Deliverable:** Cost analytics with 60% reduction demonstrated

#### Tasks

**[PERF-009] Token Usage Tracking**

- **Description:** Implement automatic token counting for all LLM calls with per-agent and per-task aggregation
- **Acceptance:**
  - [ ] TokenCounter class for usage tracking
  - [ ] Track prompt_tokens, completion_tokens, total_tokens per call
  - [ ] Extract token counts from LLM response metadata
  - [ ] Support for OpenAI response format (usage.prompt_tokens)
  - [ ] Support for Anthropic response format (usage.input_tokens)
  - [ ] Support for generic backends with fallback estimation
  - [ ] Per-agent token aggregation (agent_name -> total_tokens)
  - [ ] Per-task token tracking via context.task_id
  - [ ] get_token_stats() -> dict returns aggregated statistics
  - [ ] Real-time token counter updates in async callbacks
  - [ ] Thread-safe counter increments for concurrent tasks
  - [ ] Export token usage to OpenTelemetry spans
  - [ ] Log warning when task exceeds token budget
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-008
- **Priority:** P1 (High)
- **Files:** src/rlm/cost_tracking.py (create)

**[PERF-010] Cost Estimation and Reporting**

- **Description:** Add cost calculation using model pricing data and generate cost reports with optimization recommendations
- **Acceptance:**
  - [ ] CostEstimator class with model pricing database
  - [ ] Pricing data for GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Claude 3.5 Haiku
  - [ ] calculate_cost(tokens, model_name) -> float in USD
  - [ ] Separate pricing for prompt_tokens and completion_tokens
  - [ ] get_cost_report() -> dict with total_cost, cost_by_agent, cost_by_task
  - [ ] Savings calculation: baseline_cost vs cached_cost
  - [ ] 60% cost reduction demonstrated with caching enabled
  - [ ] Cost projection: estimate total cost for N tasks
  - [ ] Budget alerts when cost exceeds threshold
  - [ ] Export cost metrics to OpenTelemetry
  - [ ] Integration with CachedAsyncEngine for cache savings
  - [ ] CSV export for cost reports (task_id, agent, tokens, cost)
  - [ ] Optimization recommendations (e.g., "Use mini model for 40% savings")
- **Effort:** 5 story points (4 hours)
- **Owner:** Backend Engineer
- **Dependencies:** PERF-009
- **Priority:** P1 (High)
- **Files:** src/rlm/cost_tracking.py (edit), tests/integration/test_cost_tracking.py (create)

## Critical Path

```plaintext
PERF-002 → PERF-003 → PERF-004 → PERF-005 → PERF-006 → PERF-007 → PERF-008 → PERF-009 → PERF-010
  (2h)      (4h)       (4h)       (3.5h)     (5h)       (4h)       (4h)       (4h)       (4h)
                            [34.5 hours ≈ 4.3 days of pure coding, 4 weeks with testing/iteration]
```

**Bottlenecks:**

- PERF-003: Async refactor is foundational for all subsequent work
- PERF-006: Redis integration requires external dependency setup
- PERF-008: Multi-platform testing across Datadog/Grafana/Jaeger

**Parallel Tracks:**

- Week 1-2: Async refactor + caching (sequential, tightly coupled)
- Week 3-4: Observability + cost tracking (can overlap with testing)

## Quick Wins (Week 1)

1. **PERF-002 Async Protocols** - Unblocks async engine development (2 hours)
2. **PERF-003 AsyncRecursiveEngine** - Core async functionality complete (4 hours)
3. **PERF-004 Parallel Execution** - 8-10x speedup demonstrated (4 hours)

## Risk Mitigation

| Task     | Risk                   | Mitigation                                                    | Contingency                                  |
| -------- | ---------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| PERF-003 | Async breaking changes | Provide solve_sync() wrapper, maintain backward compatibility | Keep synchronous RecursiveEngine as fallback |
| PERF-004 | Semaphore deadlocks    | Comprehensive async tests, timeout guards                     | Add circuit breaker for stuck tasks          |
| PERF-006 | Redis unavailable      | Graceful degradation to L1-only, try/except wrappers          | Document L1-only deployment mode             |
| PERF-006 | Low cache hit rate     | Tune similarity threshold (0.85), increase L1 size            | Add cache warming strategy                   |
| PERF-007 | High tracing overhead  | Sampling (10% in prod), measure overhead in benchmarks        | Disable tracing in critical paths            |
| PERF-008 | Exporter compatibility | Test with 3+ platforms, use OTLP standard                     | Provide custom exporter interface            |
| PERF-009 | Missing token metadata | Fallback to tiktoken estimation, support generic backends     | Log warning and use approximation            |

## Testing Strategy

### Automated Testing Tasks

- **PERF-003 Async Unit Tests** (included in task) - Mock async LLM, test async patterns
- **PERF-004 Concurrency Tests** (included in task) - asyncio.gather(), semaphore limits
- **PERF-006 Cache Integration Tests** (included in task) - Real Redis, cache hit rate validation
- **PERF-007 Tracing Tests** (included in task) - Span creation, attribute validation
- **PERF-010 Cost Tracking Tests** (included in task) - Token counting, cost calculation

### Quality Gates

- ≥90% code coverage for all new modules (pytest --cov=src/rlm --cov-fail-under=90)
- mypy --strict passes with zero errors
- All unit tests pass in <10 seconds
- Integration tests pass with real Redis and OpenTelemetry collector
- Benchmarks demonstrate all performance targets:
  - 8-10x concurrent throughput (asyncio.gather vs sequential)
  - <100ms async overhead per call
  - 40-50% cache hit rate
  - 15x cached response speedup
  - <5% OpenTelemetry overhead
  - 60% cost reduction with caching

### Performance Benchmarks

**Week 1 Benchmarks** (PERF-004):

```python
# tests/benchmarks/test_async_performance.py
async def test_concurrent_throughput():
    """Verify 8-10x improvement for 10 concurrent tasks."""
    # Sequential: 10 tasks * 1s each = 10s
    # Concurrent: max(1s, 1s, ...) = 1s
    # Speedup: 10s / 1s = 10x
    assert speedup >= 8.0
```

**Week 2 Benchmarks** (PERF-006):

```python
# tests/benchmarks/test_cache_performance.py
async def test_cache_hit_rate():
    """Verify 40-50% cache hit rate with realistic workload."""
    # Run 100 tasks with 50% duplicate prompts
    # Expected: 40-50% cache hits
    assert cache_stats['hit_rate'] >= 0.40

async def test_cached_speedup():
    """Verify 15x speedup for cached responses."""
    # Measure: LLM call (1000ms) vs cache lookup (50ms)
    # Speedup: 1000ms / 50ms = 20x (exceeds target)
    assert speedup >= 15.0
```

**Week 3 Benchmarks** (PERF-007):

```python
# tests/benchmarks/test_tracing_overhead.py
async def test_tracing_overhead():
    """Verify <5% overhead from OpenTelemetry."""
    # Baseline: 1000ms without tracing
    # Traced: 1050ms with tracing
    # Overhead: (1050 - 1000) / 1000 = 5%
    assert overhead_pct < 0.05
```

**Week 4 Benchmarks** (PERF-010):

```python
# tests/benchmarks/test_cost_tracking.py
async def test_cost_reduction():
    """Verify 60% cost reduction with caching."""
    # Baseline: 100 tasks * $0.01 each = $1.00
    # Cached: 50 cached (free) + 50 LLM ($0.50) = $0.40
    # Reduction: ($1.00 - $0.40) / $1.00 = 60%
    assert cost_reduction_pct >= 0.60
```

## Team Allocation

**Backend Engineer (1 FTE)**

- Async protocols and types (PERF-002)
- Async engine core (PERF-003)
- Parallel execution (PERF-004)
- L1 caching (PERF-005)
- L2 Redis caching (PERF-006)
- OpenTelemetry tracing (PERF-007)
- Exporter integration (PERF-008)
- Token tracking (PERF-009)
- Cost estimation (PERF-010)

## Sprint Planning

**Sprint 2 (Week 1, 13 SP capacity)**

| Day     | Focus              | Story Points | Key Deliverables               |
| ------- | ------------------ | ------------ | ------------------------------ |
| Mon-Tue | Async Protocols    | 3 SP         | AsyncLLMCaller, async types    |
| Wed-Thu | Async Engine       | 5 SP         | AsyncRecursiveEngine class     |
| Fri     | Parallel Execution | 5 SP         | asyncio.gather(), solve_sync() |

**Sprint 3 (Week 2, 11 SP capacity)**

| Day     | Focus    | Story Points | Key Deliverables                      |
| ------- | -------- | ------------ | ------------------------------------- |
| Mon-Tue | L1 Cache | 5 SP         | CachedAsyncEngine, LRU eviction       |
| Wed-Fri | L2 Cache | 6 SP         | Redis semantic cache, 40-50% hit rate |

**Sprint 4 (Week 3, 10 SP capacity)**

| Day     | Focus        | Story Points | Key Deliverables                       |
| ------- | ------------ | ------------ | -------------------------------------- |
| Mon-Wed | Tracing Core | 5 SP         | InstrumentedAsyncEngine, span creation |
| Thu-Fri | Exporters    | 5 SP         | Datadog/Grafana/Jaeger integration     |

**Sprint 5 (Week 4, 10 SP capacity)**

| Day     | Focus          | Story Points | Key Deliverables                     |
| ------- | -------------- | ------------ | ------------------------------------ |
| Mon-Tue | Token Tracking | 5 SP         | TokenCounter, per-agent stats        |
| Wed-Fri | Cost Reporting | 5 SP         | Cost estimation, 60% reduction proof |

## Task Sequencing for /implement

Tasks must be executed in strict dependency order:

1. PERF-002 (depends on CORE-001 complete)
2. PERF-003 (depends on PERF-002)
3. PERF-004 (depends on PERF-003)
4. PERF-005 (depends on PERF-004)
5. PERF-006 (depends on PERF-005)
6. PERF-007 (depends on PERF-006)
7. PERF-008 (depends on PERF-007)
8. PERF-009 (depends on PERF-008)
9. PERF-010 (depends on PERF-009)

## Estimation Method

**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)

**Mapping:**

- 1 SP = ~30 min (simple, well-defined)
- 2 SP = ~1 hour (moderate complexity)
- 3 SP = ~2 hours (complex, requires design)
- 5 SP = ~4 hours (very complex, multiple components)
- 6 SP = ~5 hours (integration with external systems)

**Assumptions:**

- Developer familiar with async/await in Python 3.12+
- Developer has Redis and OpenTelemetry experience
- Development environment includes Docker for Redis testing
- Access to Datadog/Grafana/Jaeger for integration testing

## Definition of Done

For each story ticket to be marked COMPLETE:

- [ ] Code written following Python 3.12 async standards
- [ ] from **future** import annotations at top of all files
- [ ] All type hints use built-in generics (list[T], dict[K,V], T | None)
- [ ] Async functions use async def and await keywords correctly
- [ ] mypy --strict passes with zero errors
- [ ] Google-style docstrings for all public async APIs
- [ ] Unit tests written and passing (including async tests with pytest-asyncio)
- [ ] Test coverage ≥90% for new code
- [ ] Integration tests passing with real Redis/OpenTelemetry (for PERF-006, PERF-008)
- [ ] Performance benchmarks meet targets (8-10x, 40-50% hit rate, <5% overhead, 60% cost reduction)
- [ ] Code reviewed (self-review minimum)
- [ ] No FIXMEs or TODOs in committed code
- [ ] Documentation updated (docstrings, observability.md)

## Appendix

### CSV Export for Project Management

```csv
ID,Title,Description,Estimate_SP,Priority,Assignee,Dependencies,Week
PERF-002,Async Protocols,AsyncLLMCaller and async type system,3,P1,Backend,CORE-001,1
PERF-003,Async Engine Core,AsyncRecursiveEngine implementation,5,P1,Backend,PERF-002,1
PERF-004,Parallel Execution,asyncio.gather() and solve_sync() wrapper,5,P1,Backend,PERF-003,1
PERF-005,L1 Cache,In-memory LRU cache with exact matching,5,P1,Backend,PERF-004,2
PERF-006,L2 Redis Cache,Semantic similarity cache with redisvl,6,P1,Backend,PERF-005,2
PERF-007,OpenTelemetry Tracing,Span creation and context attributes,5,P1,Backend,PERF-006,3
PERF-008,Exporter Integration,Datadog/Grafana/Jaeger support,5,P1,Backend,PERF-007,3
PERF-009,Token Tracking,Token counter and per-agent aggregation,5,P1,Backend,PERF-008,4
PERF-010,Cost Reporting,Cost estimation and 60% reduction proof,5,P1,Backend,PERF-009,4
```

### Success Metrics

**Technical:**

- 8-10x concurrent task throughput (asyncio.gather vs sequential)
- <100ms async overhead per call
- 40-50% cache hit rate in production workloads
- 15x speedup for cached responses
- 100% execution coverage with tracing when enabled
- <5% overhead from OpenTelemetry instrumentation
- 60% API cost reduction with caching
- Zero crashes when Redis/OpenTelemetry unavailable
- 100% backward compatibility via solve_sync()

**Business:**

- Enables CAP-001 (tools, streaming) work to begin
- Production-ready observability for enterprise customers
- Significant cost savings for high-volume workloads
- Zero vendor lock-in (works with Datadog, Grafana, New Relic, Jaeger)

### Dependencies and Integration

**Requires CORE-001 Complete:**

- RecursiveEngine as base implementation
- RLMContext for execution state
- SharedMemory for variable offloading
- Type system (Input, Output, Item, LLMCaller)

**Requires INTEL-001 Complete:**

- Multi-agent routing patterns
- Agent registry for async agents
- Router model for planning decisions

**Enables CAP-001:**

- Async streaming will use AsyncRecursiveEngine
- Tool calling will integrate with async execution
- Checkpoints will leverage OpenTelemetry for debugging

### Performance Targets Summary

| Metric                 | Target               | Measurement                                 |
| ---------------------- | -------------------- | ------------------------------------------- |
| Concurrent Throughput  | 8-10x                | Benchmark: 10 tasks parallel vs sequential  |
| Async Overhead         | <100ms               | Benchmark: async call vs sync wrapper       |
| Cache Hit Rate         | 40-50%               | Integration test: 100 tasks, 50% duplicates |
| Cached Speedup         | 15x                  | Benchmark: LLM call vs cache lookup         |
| Tracing Coverage       | 100%                 | All solve() calls create spans              |
| Tracing Overhead       | <5%                  | Benchmark: traced vs untraced execution     |
| Cost Reduction         | 60%                  | Integration test: baseline vs cached cost   |
| Redis Availability     | Graceful degradation | Unit test: L1-only fallback                 |
| Backward Compatibility | 100%                 | All existing tests pass with solve_sync()   |

### Optional Dependencies

**Required:**

- opentelemetry-api
- opentelemetry-sdk
- opentelemetry-exporter-otlp-proto-grpc

**Optional (graceful degradation if missing):**

- redisvl (for L2 semantic cache)
- redis (for L2 cache backend)

Add to pyproject.toml:

```toml
[project.optional-dependencies]
performance = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0",
    "redisvl>=0.2.0",
    "redis>=5.0.0",
]
```

Install with:

```bash
uv add --optional performance opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc redisvl redis
```

### Migration Guide (Sync to Async)

**Before (synchronous):**

```python
from rlm import RecursiveEngine

engine = RecursiveEngine(llm=sync_llm)
result = engine.solve("Task description")
print(result["content"])
```

**After (async):**

```python
import asyncio
from rlm import AsyncRecursiveEngine

async def main():
    engine = AsyncRecursiveEngine(llm=async_llm)
    result = await engine.solve("Task description")
    print(result["content"])

asyncio.run(main())
```

**Backward compatibility (sync wrapper):**

```python
from rlm import AsyncRecursiveEngine

engine = AsyncRecursiveEngine(llm=async_llm)
result = engine.solve_sync("Task description")  # Uses asyncio.run() internally
print(result["content"])
```

### Week-by-Week Progress Tracking

**Week 1 Checklist:**

- [ ] AsyncLLMCaller protocol defined (PERF-002)
- [ ] AsyncRecursiveEngine skeleton complete (PERF-003)
- [ ] asyncio.gather() integration complete (PERF-004)
- [ ] solve_sync() backward compatibility verified (PERF-004)
- [ ] 8-10x throughput benchmark passing (PERF-004)
- [ ] <100ms async overhead benchmark passing (PERF-004)

**Week 2 Checklist:**

- [ ] CachedAsyncEngine with L1 cache complete (PERF-005)
- [ ] LRU eviction working correctly (PERF-005)
- [ ] Redis semantic cache integrated (PERF-006)
- [ ] Graceful degradation to L1-only tested (PERF-006)
- [ ] 40-50% cache hit rate benchmark passing (PERF-006)
- [ ] 15x cached speedup benchmark passing (PERF-006)

**Week 3 Checklist:**

- [ ] InstrumentedAsyncEngine with span creation (PERF-007)
- [ ] Span attributes (task, depth, agent, duration) captured (PERF-007)
- [ ] Exception recording working correctly (PERF-007)
- [ ] <5% tracing overhead benchmark passing (PERF-007)
- [ ] OTLP exporter configured (PERF-008)
- [ ] Integration guides for Datadog/Grafana/Jaeger written (PERF-008)
- [ ] 100% execution coverage verified (PERF-008)

**Week 4 Checklist:**

- [ ] TokenCounter tracking prompt/completion tokens (PERF-009)
- [ ] Per-agent and per-task aggregation working (PERF-009)
- [ ] Thread-safe counter updates verified (PERF-009)
- [ ] CostEstimator with model pricing database (PERF-010)
- [ ] Cost reports generated correctly (PERF-010)
- [ ] 60% cost reduction benchmark passing (PERF-010)
- [ ] CSV export for cost reports working (PERF-010)
