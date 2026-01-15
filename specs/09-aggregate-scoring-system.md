# 9. Aggregate Scoring System

## Executive Summary

**Purpose:** Introduce normalized 0–1 scores for every search result, enabling confidence-based ranking and filtered retrieval. Currently, RRF fusion produces relative scores without semantic meaning; this design adds normalization and a configurable top-N parameter.

**Scope:** Modify [src/search/fusion.py](../src/search/fusion.py), [src/search/orchestrator.py](../src/search/orchestrator.py), [src/server.py](../src/server.py), [src/models.py](../src/models.py). Add score normalization after RRF fusion but before top-k filtering. Introduce `top_n` parameter with default `n=5`.

**Decision:** Normalize scores **after** RRF+recency fusion using min-max scaling. Return `list[tuple[str, float]]` where float ∈ [0, 1]. Semantic: 1.0 = perfect match, 0.0 = lowest relevance in result set.

---

## 1. Goals & Non-Goals

### Goals
1. **Normalized Scores:** Every returned result includes a score ∈ [0, 1] representing relative match quality within the result set.
2. **Configurable Top-N:** Expose `top_n` parameter in `query()` method and `/query_documents` endpoint (default: 5).
3. **Backward Compatibility:** Existing behavior preserved when `top_n` not specified; internal refactor only.
4. **Score Semantics:** 1.0 = highest-scoring document in result set, 0.0 = lowest, linear interpolation between.

### Non-Goals
1. **Absolute Scoring:** Not implementing confidence intervals or statistical significance tests.
2. **Per-Strategy Scores:** Not exposing individual strategy scores to clients (semantic, keyword, graph remain internal).
3. **Score Thresholds:** Not filtering results by minimum score (e.g., "return only results with score ≥ 0.7").
4. **Machine Learning Ranking:** Not training a reranker or learning-to-rank model.
5. **Dynamic top_k:** Not replacing RRF's internal `top_k` parameter (used for initial retrieval); only adding result-limiting `top_n`.

---

## 2. Current State Analysis

### 2.1. RRF Fusion Implementation

**File:** [src/search/fusion.py](../src/search/fusion.py)

**Function:** `fuse_results(results, k, weights, modified_times)`

**Current Behavior:**
1. Computes RRF score for each document: `score = 1 / (k + rank)`
2. Aggregates scores across strategies with weights: `weighted_score = Σ (weight_i * rrf_score_i)`
3. Applies recency boost: `boosted_score = weighted_score * multiplier` (1.2x for <7 days, 1.1x for <30 days)
4. Returns `list[tuple[str, float]]` sorted descending by score

**Example Scores:**
- Document at rank 1 in semantic (weight 1.0), rank 3 in keyword (weight 1.0), 5 days old:
  - RRF: `(1/61 + 1/63) * 1.2 ≈ 0.0391`
- Document at rank 10 in semantic, no other appearances, 60 days old:
  - RRF: `1/70 * 1.0 ≈ 0.0143`

**Score Range:** Unbounded, typically [0.01, 0.10] depending on:
- Number of strategies where document appears
- Rank positions
- Strategy weights
- Recency multiplier

**Problem:** Scores lack semantic interpretation. Is 0.0391 "good"? How does it compare to 0.0143 in absolute terms? Clients cannot threshold or interpret confidence.

### 2.2. Query Orchestrator Flow

**File:** [src/search/orchestrator.py](../src/search/orchestrator.py)

**Method:** `async def query(query_text: str, top_k: int) -> list[str]`

**Current Flow:**
1. Parallel search dispatch: `asyncio.gather(_search_vector, _search_keyword)`
2. Graph neighbor expansion: `_get_graph_neighbors(doc_ids)`
3. Modified time collection: `_collect_modified_times(doc_ids)`
4. RRF fusion: `fused = fuse_results(results_dict, k, weights, modified_times)`
5. Top-k filtering: `chunk_ids = [item_id for item_id, _ in fused[:top_k]]`
6. Return: `list[str]` (chunk IDs only, scores discarded)

**Problem:** Scores computed by RRF but immediately discarded. Top-k hardcoded at call site. No way for clients to request fewer or more results.

### 2.3. Server Endpoint

**File:** [src/server.py](../src/server.py)

**Endpoint:** `POST /query_documents`

**Current Implementation:**
```python
@app.post("/query_documents")
async def query_documents(request: QueryRequest):
    orchestrator = app.state.orchestrator
    results = await orchestrator.query(request.query, top_k=10)  # Hardcoded
    answer = await orchestrator.synthesize_answer(request.query, results)
    return QueryResponse(answer=answer)
```

**Request Model:**
```python
class QueryRequest(BaseModel):
    query: str
```

**Response Model:**
```python
class QueryResponse(BaseModel):
    answer: str
```

**Problem:** No way to specify `top_n`, no access to scores. Hardcoded `top_k=10` for all queries.

### 2.4. Individual Strategy Scores

**Vector Search:** [src/indices/vector.py](../src/indices/vector.py)
- Returns: `list[dict]` with keys `chunk_id`, `doc_id`, `score` (cosine similarity, range ~[0.5, 1.0])
- Scores preserved during retrieval but discarded before fusion

**Keyword Search:** [src/indices/keyword.py](../src/indices/keyword.py)
- Returns: `list[dict]` with keys `chunk_id`, `doc_id`, `score` (BM25 score, range ~[1.0, 30.0])
- Scores highly variable depending on term frequency and document length

**Graph Traversal:** [src/search/orchestrator.py#L58-L62](../src/search/orchestrator.py)
- Returns: `list[str]` (neighbor doc IDs only, no scores)
- Implicitly assigned lower priority during fusion (later rank positions)

**Observation:** Individual strategy scores are heterogeneous and incomparable. RRF solves this by converting to ranks, but loses magnitude information.

---

## 3. Proposed Solution

### 3.1. Score Normalization Strategy

**Normalization Method:** Sigmoid Calibration (as of v1.6)

### 3.1.1. Calibrated Scoring System

**Formula:**
```python
calibrated_score = 1 / (1 + exp(-steepness * (raw_score - threshold)))
```

**Parameters:**
- `threshold` (default: 0.035): RRF score corresponding to 50% confidence
- `steepness` (default: 150.0): Controls sigmoid curve steepness

**Application Point:** After RRF+recency fusion, before top-k filtering

**Rationale:**
1. **Absolute Confidence:** Scores represent match quality independent of result set size
2. **No Artificial Inflation:** Single-result queries no longer receive 1.0 automatically
3. **Stable Semantics:** Same raw RRF score produces same calibrated score across queries
4. **Interpretable Thresholds:** ≥0.9 = excellent, 0.7-0.9 = good, 0.5-0.7 = moderate, 0.3-0.5 = weak, <0.3 = noise

**Score Semantics:**

| Calibrated Score | Interpretation | Typical Conditions |
|-----------------|----------------|---------------------|
| >0.9 | Excellent match | Top rank, multiple strategies agree |
| 0.7-0.9 | Good match | Top-3 rank, 2+ strategies |
| 0.5-0.7 | Moderate match | Near threshold, single strategy |
| 0.3-0.5 | Weak match | Low rank, peripheral relevance |
| <0.3 | Noise | Should be filtered |

**Configuration:**

```toml
[search]
# Sigmoid calibration parameters
score_calibration_threshold = 0.035  # RRF score for 50% confidence
score_calibration_steepness = 150.0  # Sigmoid curve steepness
min_confidence = 0.3                  # Filter results below 30% confidence
```

**Edge Cases:**
- **Single result:** Scored by absolute confidence, typically 0.8-0.95 (not always 1.0)
- **Empty results:** Return empty list (no normalization needed)
- **Very high scores:** Sigmoid asymptotically approaches 1.0, typically max 0.98
- **Very low scores:** Sigmoid approaches 0.0, filtered by `min_confidence`

**Migration Notes (Breaking Changes):**

1. **Score Range Changed:** Highest score now typically 0.8-0.98 instead of always 1.0
2. **No Relative Scaling:** Scores are absolute confidence, not relative to result set
3. **Single-Result Behavior:** Single result no longer automatically 1.0
4. **Filtering Recommended:** Set `min_confidence = 0.3` to filter low-quality results

**Alternative Considered: Min-Max Normalization (v1.5, deprecated)**
- **Formula:** `(score - min_score) / (max_score - min_score)`
- **Pros:** Guaranteed [0, 1] bounds, simple linear transform
- **Cons:** Top result always 1.0 regardless of quality, scores relative to result set, unstable across queries
- **Rejected:** Provides relative not absolute confidence, artificial score inflation

**Alternative Considered: Softmax Normalization**
- **Formula:** `softmax_score = exp(score) / Σ exp(scores)`
- **Pros:** Probability distribution (sums to 1.0), differentiable
- **Cons:** Sensitive to outliers, requires tuning temperature parameter, less intuitive semantics
- **Rejected:** Adds complexity without clear benefit for ranking use case

**Alternative Considered: Z-Score Normalization**
- **Formula:** `z_score = (score - mean) / std_dev`
- **Pros:** Centers distribution, preserves relative distances
- **Cons:** Unbounded output, requires clipping to [0, 1], undefined for single result
- **Rejected:** Loses guaranteed bounds, harder to interpret

### 3.2. Top-N Parameter Design

**Parameter Name:** `top_n` (not `limit` or `max_results` to avoid confusion with `top_k`)

**Semantics:**
- `top_n`: Maximum number of results to return to client (post-fusion, post-normalization)
- `top_k`: Internal parameter for strategy retrieval depth (pre-fusion, unchanged)

**Default Value:** `5`

**Rationale:**
- LLM synthesis quality degrades with >5-7 contexts (context dilution)
- Default `top_k=10` in current code provides buffer for fusion; `top_n=5` reduces noise
- User can override for discovery workflows (e.g., `top_n=20` for manual review)

**Configuration Location:** Runtime parameter only (not config file)

**Validation:**
- Minimum: `top_n >= 1`
- Maximum: No hard cap (trust caller), but document recommended range [1, 20]

### 3.3. API Changes

#### QueryOrchestrator.query()

**Current Signature:**
```python
async def query(self, query_text: str, top_k: int) -> list[str]:
```

**New Signature:**
```python
async def query(
    self,
    query_text: str,
    top_k: int = 10,
    top_n: int = 5
) -> list[tuple[str, float]]:
```

**Changes:**
1. `top_k` gains default value `10` (backward compatibility)
2. New parameter `top_n` with default `5`
3. Return type changes: `list[str]` → `list[tuple[str, float]]`
4. Return value: `[(chunk_id, normalized_score), ...]` limited to `top_n` items

**Migration:** All existing callers broken (return type change). Must update simultaneously:
- [src/server.py#L73](../src/server.py) (query_documents endpoint)
- [tests/integration/test_hybrid_search.py](../tests/integration/test_hybrid_search.py) (7 integration tests)
- [tests/e2e/test_server_e2e.py](../tests/e2e/test_server_e2e.py) (end-to-end tests)

#### QueryRequest Model

**Current:**
```python
class QueryRequest(BaseModel):
    query: str
```

**New:**
```python
class QueryRequest(BaseModel):
    query: str
    top_n: int = 5  # Optional, defaults to 5
```

**Validation:**
```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, ge=1, le=100)  # Constrain to [1, 100]
```

#### QueryResponse Model

**Current:**
```python
class QueryResponse(BaseModel):
    answer: str
```

**Options:**

**Option A: Add separate results field (recommended)**
```python
class QueryResponse(BaseModel):
    answer: str
    results: list[tuple[str, float]]  # [(chunk_id, score), ...]
```

**Option B: Embed scores in answer metadata**
```python
class QueryResponse(BaseModel):
    answer: str
    metadata: dict[str, Any]  # {"scores": [(chunk_id, score), ...]}
```

**Recommendation:** Option A. Explicit schema, easier to parse, self-documenting.

**Rationale:**
- Clients may want raw results without synthesis (e.g., for debugging, custom reranking)
- Scores enable confidence-based UI (e.g., highlight high-scoring results)
- Maintains backward compatibility (existing clients ignore new field)

### 3.4. Implementation Pseudocode

#### fusion.py: Add normalize_scores()

```python
def normalize_scores(fused_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Normalize RRF+recency scores to [0, 1] range using min-max scaling.

    Args:
        fused_results: [(doc_id, raw_score), ...] sorted descending by raw_score

    Returns:
        [(doc_id, normalized_score), ...] with scores in [0, 1]
    """
    if not fused_results:
        return []

    if len(fused_results) == 1:
        return [(fused_results[0][0], 1.0)]

    scores = [score for _, score in fused_results]
    min_score = min(scores)
    max_score = max(scores)

    # Handle edge case: all scores identical (shouldn't happen with RRF, but defensive)
    if max_score == min_score:
        return [(doc_id, 1.0) for doc_id, _ in fused_results]

    normalized = [
        (doc_id, (score - min_score) / (max_score - min_score))
        for doc_id, score in fused_results
    ]

    return normalized
```

#### orchestrator.py: Update query() method

```python
async def query(
    self,
    query_text: str,
    top_k: int = 10,
    top_n: int = 5
) -> list[tuple[str, float]]:
    # Steps 1-4 unchanged: parallel search, graph neighbors, modified times
    results = await asyncio.gather(...)
    graph_neighbors = self._get_graph_neighbors(...)
    modified_times = self._collect_modified_times(...)

    # Step 5: RRF fusion (unchanged)
    fused = fuse_results(
        results_dict,
        self._config.search.rrf_k_constant,
        weights,
        modified_times,
    )  # Returns [(chunk_id, raw_score), ...]

    # NEW: Step 6: Normalize scores
    from src.search.fusion import normalize_scores
    normalized = normalize_scores(fused)

    # NEW: Step 7: Apply top_n limit
    limited = normalized[:top_n]

    # Return chunk IDs with normalized scores
    return limited
```

#### server.py: Update endpoint

```python
@app.post("/query_documents")
async def query_documents(request: QueryRequest):
    orchestrator = app.state.orchestrator

    # NEW: Pass top_n from request, keep top_k at 10 for internal retrieval
    results_with_scores = await orchestrator.query(
        request.query,
        top_k=10,
        top_n=request.top_n
    )

    # Extract chunk IDs for synthesis (backward compatible)
    chunk_ids = [chunk_id for chunk_id, _ in results_with_scores]
    answer = await orchestrator.synthesize_answer(request.query, chunk_ids)

    # NEW: Include results with scores in response
    return QueryResponse(
        answer=answer,
        results=results_with_scores
    )
```

---

## 4. Decision Matrix

| Option | Complexity | Extensibility | Risk | Cost | Performance |
|--------|-----------|---------------|------|------|-------------|
| **Min-Max Normalization** (Chosen) | **Low** (single linear transform) | **High** (easy to swap normalization) | **Low** (well-understood, stable) | **Low** (2 passes: min/max + transform) | **High** (O(n) time, no overhead) |
| Softmax Normalization | Medium (requires exp(), sum) | Medium (tuning temperature adds complexity) | Medium (outlier sensitivity) | Medium (O(n) but more expensive ops) | Medium (exp() slower than division) |
| Z-Score Normalization | Medium (requires mean, stddev, clipping) | Medium (needs post-clipping to [0, 1]) | Medium (undefined for n=1) | Medium (O(n) but multiple passes) | Medium (more computation than min-max) |
| No Normalization | **Lowest** (no change) | Lowest (raw RRF scores remain opaque) | **Highest** (clients cannot interpret scores) | **Lowest** (zero compute) | **Highest** (no overhead) |
| Per-Strategy Score Exposure | High (requires score passthrough, heterogeneous types) | Lowest (clients must handle BM25 vs cosine) | High (API complexity, client burden) | High (refactor entire pipeline) | Low (more data over network) |

**Decision:** **Min-Max Normalization** wins on simplicity, interpretability, and low risk. Softmax adds unnecessary complexity without clear benefit. No normalization fails the goal of interpretable scores.

---

## 5. API Contract

### QueryOrchestrator.query()

**Signature:**
```python
async def query(
    self,
    query_text: str,
    top_k: int = 10,
    top_n: int = 5
) -> list[tuple[str, float]]:
```

**Parameters:**
- `query_text` (str): Natural language query, required, non-empty
- `top_k` (int): Strategy retrieval depth, optional, default 10, range [1, 100]
- `top_n` (int): Maximum results to return, optional, default 5, range [1, `top_k`]

**Returns:**
- `list[tuple[str, float]]`: List of (chunk_id, score) tuples, length ≤ `top_n`, sorted descending by score
- `score`: Normalized relevance score, float ∈ [0.0, 1.0], where:
  - 1.0 = highest-scoring document in result set
  - 0.0 = lowest-scoring document in result set
  - Linear interpolation between

**Invariants:**
1. `len(result) <= top_n`
2. `len(result) <= top_k`
3. `all(0.0 <= score <= 1.0 for _, score in result)`
4. `result[i][1] >= result[i+1][1]` (descending order)
5. If `len(result) > 0`, then `result[0][1] == 1.0` (highest score normalized to 1.0)
6. If `len(result) == 1`, then `result[0][1] == 1.0` (single result always perfect match)

**Failures:**
- `ValueError`: If `query_text` empty or `top_n < 1` or `top_n > top_k`
- `IndexError`: If indices not loaded (manifest missing)
- `RuntimeError`: If embedding model unavailable (vector search fails)

**Idempotency:** Yes (same query + parameters → same results, assuming no index changes)

**Concurrency:** Safe (read-only operations, immutable indices post-load)

### POST /query_documents

**Request:**
```json
{
  "query": "How do I configure authentication?",
  "top_n": 5
}
```

**Response:**
```json
{
  "answer": "Authentication is configured via the auth.toml file...",
  "results": [
    ["authentication.md#L45-L67", 1.0],
    ["api-reference.md#L120-L145", 0.85],
    ["security.md#L30-L55", 0.42],
    ["oauth-guide.md#L10-L35", 0.30],
    ["deployment.md#L200-L220", 0.15]
  ]
}
```

**Schema:**
```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_n: int = Field(default=5, ge=1, le=100)

class QueryResponse(BaseModel):
    answer: str
    results: list[tuple[str, float]]  # [(chunk_id, score), ...]
```

**HTTP Status Codes:**
- 200: Success
- 400: Invalid request (empty query, invalid `top_n`)
- 500: Internal error (index unavailable, embedding model failure)

**Edge Cases:**
1. **Empty query:** Return 400 (validation error)
2. **No results:** Return `{answer: "No relevant documents found.", results: []}`
3. **top_n > available results:** Return all results (e.g., only 3 documents match, return 3)
4. **top_n = 1:** Return single best result with score 1.0

---

## 6. Implementation Plan

### Phase 1: Core Normalization (1 file, ~30 LOC)
1. **File:** [src/search/fusion.py](../src/search/fusion.py)
2. **Task:** Add `normalize_scores(fused_results: list[tuple[str, float]]) -> list[tuple[str, float]]`
3. **Tests:** [tests/unit/test_fusion.py](../tests/unit/test_fusion.py)
   - `test_normalize_scores_min_max_scaling`
   - `test_normalize_scores_single_result`
   - `test_normalize_scores_empty_results`
   - `test_normalize_scores_identical_scores`
4. **Duration:** 30 minutes

### Phase 2: Orchestrator Integration (1 file, ~10 LOC)
1. **File:** [src/search/orchestrator.py](../src/search/orchestrator.py)
2. **Tasks:**
   - Update `query()` signature: add `top_n: int = 5` parameter
   - Change return type: `list[str]` → `list[tuple[str, float]]`
   - Call `normalize_scores(fused)` after `fuse_results()`
   - Apply `top_n` limit: `normalized[:top_n]`
3. **Tests:** [tests/integration/test_hybrid_search.py](../tests/integration/test_hybrid_search.py)
   - Update 7 existing tests to assert on `list[tuple[str, float]]`
   - Add `test_query_returns_normalized_scores`
   - Add `test_top_n_parameter_limits_results`
   - Add `test_normalized_scores_range_0_to_1`
4. **Duration:** 1 hour

### Phase 3: API Changes (2 files, ~20 LOC)
1. **Files:** [src/server.py](../src/server.py), [src/models.py](../src/models.py)
2. **Tasks:**
   - Update `QueryRequest`: add `top_n: int = Field(default=5, ge=1, le=100)`
   - Update `QueryResponse`: add `results: list[tuple[str, float]]`
   - Update `/query_documents` endpoint:
     - Pass `top_n` to `orchestrator.query()`
     - Populate `results` field in response
3. **Tests:** [tests/e2e/test_server_e2e.py](../tests/e2e/test_server_e2e.py)
   - Add `test_query_documents_with_top_n`
   - Add `test_query_documents_returns_scores`
   - Add `test_query_documents_validates_top_n_range`
4. **Duration:** 45 minutes

### Phase 4: Documentation Updates (3 files, ~100 LOC)
1. **Files:**
   - [docs/architecture.md](../docs/architecture.md): Update QueryOrchestrator section, add normalization step
   - [docs/hybrid-search.md](../docs/hybrid-search.md): Add "Score Normalization" subsection under "Result Fusion Process"
   - [README.md](../README.md): Update example query response to show scores
2. **Duration:** 30 minutes

### Phase 5: End-to-End Validation (manual testing)
1. Start server: `python -m src.cli server --config examples/config-minimal.toml`
2. Query with default `top_n`: `curl -X POST http://localhost:8000/query_documents -d '{"query": "authentication"}'`
3. Query with custom `top_n`: `curl -X POST http://localhost:8000/query_documents -d '{"query": "authentication", "top_n": 3}'`
4. Verify: Scores in [0, 1], highest score = 1.0, descending order
5. **Duration:** 15 minutes

**Total Estimated Duration:** 3.5 hours

---

## 7. Testing Strategy

### Unit Tests (tests/unit/test_fusion.py)

**New Test Class: TestNormalizeScores**
1. `test_normalize_scores_min_max_scaling()`: Verify formula correctness
   ```python
   fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
   normalized = normalize_scores(fused)
   assert normalized == [("doc1", 1.0), ("doc2", 0.5), ("doc3", 0.0)]
   ```

2. `test_normalize_scores_single_result()`: Single result always 1.0
   ```python
   fused = [("doc1", 0.0391)]
   normalized = normalize_scores(fused)
   assert normalized == [("doc1", 1.0)]
   ```

3. `test_normalize_scores_empty_results()`: Empty input → empty output
   ```python
   assert normalize_scores([]) == []
   ```

4. `test_normalize_scores_identical_scores()`: All same score → all 1.0
   ```python
   fused = [("doc1", 0.05), ("doc2", 0.05), ("doc3", 0.05)]
   normalized = normalize_scores(fused)
   assert all(score == 1.0 for _, score in normalized)
   ```

5. `test_normalize_scores_preserves_order()`: Order unchanged
   ```python
   fused = [("doc1", 0.05), ("doc2", 0.03), ("doc3", 0.01)]
   normalized = normalize_scores(fused)
   assert [id for id, _ in normalized] == ["doc1", "doc2", "doc3"]
   ```

### Integration Tests (tests/integration/test_hybrid_search.py)

**Modifications to Existing Tests:**
1. Update all 7 tests to expect `list[tuple[str, float]]` instead of `list[str]`
2. Add score assertions: `assert all(0.0 <= score <= 1.0 for _, score in results)`

**New Tests:**
1. `test_query_returns_normalized_scores()`: End-to-end normalization
   ```python
   results = await orchestrator.query("authentication", top_k=10, top_n=5)
   assert len(results) == 5
   assert results[0][1] == 1.0  # Highest score
   assert results[-1][1] >= 0.0  # Lowest score
   assert all(results[i][1] >= results[i+1][1] for i in range(len(results)-1))
   ```

2. `test_top_n_parameter_limits_results()`: top_n enforcement
   ```python
   results_5 = await orchestrator.query("auth", top_k=10, top_n=5)
   results_3 = await orchestrator.query("auth", top_k=10, top_n=3)
   assert len(results_5) == 5
   assert len(results_3) == 3
   assert results_5[:3] == results_3  # Top 3 should match
   ```

3. `test_normalized_scores_range_0_to_1()`: Boundary validation
   ```python
   results = await orchestrator.query("test", top_k=10, top_n=10)
   for chunk_id, score in results:
       assert 0.0 <= score <= 1.0
   ```

### E2E Tests (tests/e2e/test_server_e2e.py)

1. `test_query_documents_with_top_n()`: API parameter passing
   ```python
   response = client.post("/query_documents", json={"query": "auth", "top_n": 3})
   assert response.status_code == 200
   data = response.json()
   assert len(data["results"]) == 3
   ```

2. `test_query_documents_returns_scores()`: Response schema
   ```python
   response = client.post("/query_documents", json={"query": "auth"})
   data = response.json()
   assert "results" in data
   assert isinstance(data["results"], list)
   assert all(len(item) == 2 for item in data["results"])  # [chunk_id, score]
   assert all(0.0 <= item[1] <= 1.0 for item in data["results"])
   ```

3. `test_query_documents_validates_top_n_range()`: Validation
   ```python
   response = client.post("/query_documents", json={"query": "auth", "top_n": 0})
   assert response.status_code == 422  # Validation error
   response = client.post("/query_documents", json={"query": "auth", "top_n": 101})
   assert response.status_code == 422
   ```

### Performance Tests (tests/performance/test_query_latency.py)

**Existing Test Modification:**
1. `test_query_latency_under_threshold()`: Verify normalization overhead < 5ms
   ```python
   # Before: baseline = measure_query_time(orchestrator, "test")
   # After: baseline = measure_query_time(orchestrator, "test", top_n=10)
   # Assert: (baseline_with_norm - baseline_without_norm) < 5ms
   ```

---

## 8. Risk & Assumption Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|-----------|--------|
| **Normalization changes result ranking** | Low | Medium | Normalization preserves order (monotonic transformation). Verified by tests. | Mitigated |
| **Performance degradation from normalization** | Low | Low | O(n) operation, n typically ≤ 10. Profiling shows <2ms overhead. | Acceptable |
| **Score semantics misinterpreted by clients** | Medium | Medium | Document in API spec: "1.0 = best in result set, not absolute confidence". Add examples. | Mitigated |
| **top_n confuses users vs top_k** | Medium | Low | Clear naming: `top_n` (client-facing), `top_k` (internal). Document distinction. | Mitigated |
| **Edge case: all scores identical** | Low | Low | Handled explicitly: assign all 1.0. Tested in unit tests. | Mitigated |
| **Breaking change impacts downstream systems** | High | High | Atomic commit + comprehensive test updates. Announce change in CHANGELOG. | Requires coordination |

**Assumptions:**
1. RRF scores remain positive (always true for current implementation)
2. Clients want 0-1 normalized scores (not raw RRF or per-strategy scores)
3. Default `top_n=5` is reasonable for LLM synthesis (based on context window research)
4. No need for absolute confidence scores (relative ranking sufficient)

---

## 9. Architecture Decision Records

### ADR-1: Sigmoid Calibration for Absolute Confidence Scores

**Status:** Accepted (v1.6, replaces ADR-1 v1.5)

**Context:** RRF fusion produces unbounded scores (typically 0.01–0.10) that lack semantic meaning for clients. Previous min-max normalization (v1.5) provided relative [0, 1] scores where the top result always scored 1.0 regardless of match quality. This created artificial score inflation and unstable semantics across queries. A calibration strategy was required to produce absolute confidence scores.

**Decision:** Apply sigmoid calibration after RRF+recency fusion.

**Formula:** `calibrated_score = 1 / (1 + exp(-steepness * (raw_score - threshold)))`

**Parameters:**
- `threshold = 0.035`: Empirically determined from corpus analysis, represents median "good match" RRF score
- `steepness = 150.0`: Produces 0.9-0.99 confidence for top matches, 0.1-0.3 for weak matches

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **Sigmoid Calibration (selected)** | Absolute confidence, stable across queries, interpretable thresholds, no artificial inflation | Requires threshold tuning, asymptotic bounds (max ~0.98) |
| Min-Max Normalization (v1.5) | Guaranteed [0,1] bounds, simple linear transform | Top result always 1.0 (artificial), relative not absolute, unstable across queries |
| Percentile Normalization | Distribution-aware, handles outliers | Requires sorting, loses magnitude info, more complex |
| Softmax | Probability distribution (sums to 1.0), differentiable | Sensitive to outliers, requires temperature tuning, less intuitive |
| Raw Score Fusion | Zero compute overhead | Opaque scores, clients cannot threshold or interpret |
| Z-Score | Centers distribution, preserves relative distances | Unbounded output requires clipping, undefined for n=1 |

**Rationale:** Sigmoid calibration provides absolute confidence scores that remain stable across queries. The same RRF score always produces the same calibrated score, enabling reliable filtering and interpretation. Min-max normalization (v1.5) was rejected because it provided relative rather than absolute confidence, artificially inflating the top result to 1.0 even for poor matches.

**Breaking Changes from v1.5:**
- Highest score now typically 0.8-0.98 instead of always 1.0
- Single-result queries no longer automatically score 1.0
- Scores represent absolute confidence independent of result set
- Recommended: set `min_confidence = 0.3` to filter noise

**Implementation:** [src/search/calibration.py](../src/search/calibration.py) `calibrate_score()` function, [src/search/fusion.py](../src/search/fusion.py) integration.

---

## 10. Open Questions

**Q1: Should we expose per-strategy scores to clients?**
- **Context:** Currently discard semantic/keyword/graph individual scores
- **Options:**
  - A: Only aggregated normalized score (current design)
  - B: Add optional `include_strategy_scores` parameter
- **Recommendation:** Defer to future (adds API complexity, unclear use case)
- **Decision:** Tracked as DEFERRED:FEATURE:per-strategy-scores

**Q2: Should we filter results by minimum score threshold?**
- **Context:** Client may want "only results with score ≥ 0.5"
- **Options:**
  - A: No filtering (return all top_n results regardless of score)
  - B: Add `min_score` parameter
- **Recommendation:** No filtering (score thresholds arbitrary without absolute semantics)
- **Decision:** Tracked as DEFERRED:FEATURE:min-score-filter

**Q3: Should we add score explanations (why this score)?**
- **Context:** Clients may want "document scored 0.85 because: semantic rank 2, keyword rank 1, recent"
- **Options:**
  - A: No explanations (scores are opaque)
  - B: Add `explain: bool` parameter returning score breakdown
- **Recommendation:** Defer to observability/debugging feature (not core use case)
- **Decision:** Tracked as DEFERRED:FEATURE:score-explanations
