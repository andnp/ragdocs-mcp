# Implementation Plan: Search Infrastructure Overhaul (Spec 17)

## 1. Executive Summary

This plan details the implementation of GraphRAG, Score-Aware Fusion, and HyDE features for `mcp-markdown-ragdocs`. Phase 1 introduces typed edges and Leiden community detection to GraphStore. Phase 2 replaces static RRF weights with variance-aware dynamic fusion. Phase 3 adds a `search_with_hypothesis` MCP tool for hypothesis-driven search. The upgrade is backwards-compatible; existing indices migrate on re-index.

**Total estimated LOC:** ~650 (new) + ~200 (modifications)

---

## 2. Implementation Phases

### Phase 1: Graph Upgrade (~280 LOC)

**Goal:** Transform GraphStore from untyped edges to typed relationships with community detection.

| Task | File | LOC | Description |
|------|------|-----|-------------|
| 1.1 | `src/models.py` | 15 | Add `EdgeType` enum |
| 1.2 | `src/indices/graph.py` | 80 | Add community detection, storage, and query methods |
| 1.3 | `src/parsers/markdown.py` | 60 | Infer edge types from header context |
| 1.4 | `src/indexing/manager.py` | 25 | Pass header context to `add_edge()` |
| 1.5 | `src/search/orchestrator.py` | 40 | Add community boosting to graph neighbor scoring |
| 1.6 | `src/config.py` | 20 | Add `[search.graph]` config section |
| 1.7 | `pyproject.toml` | 5 | Add `cdlib` dependency |
| 1.8 | Tests | 35 | Unit tests for edge types and community detection |

**Dependencies:** None (standalone phase)

### Phase 2: Fusion Logic (~220 LOC)

**Goal:** Replace static RRF with variance-aware dynamic fusion.

| Task | File | LOC | Description |
|------|------|-----|-------------|
| 2.1 | `src/search/fusion.py` | 80 | Add `calculate_variance()`, `normalize_strategy_scores()`, `compute_dynamic_weights()` |
| 2.2 | `src/search/orchestrator.py` | 60 | Integrate dynamic weights into query flow |
| 2.3 | `src/search/classifier.py` | 30 | Extend query classification for variance signals |
| 2.4 | `src/config.py` | 15 | Add `[search.fusion]` config for variance thresholds |
| 2.5 | Tests | 35 | Unit tests for variance calculation and dynamic weights |

**Dependencies:** None (can run parallel to Phase 1)

### Phase 3: HyDE Support (~150 LOC)

**Goal:** Add hypothesis-driven search via new MCP tool.

| Task | File | LOC | Description |
|------|------|-----|-------------|
| 3.1 | `src/mcp_server.py` | 60 | Add `search_with_hypothesis` tool registration and handler |
| 3.2 | `src/search/orchestrator.py` | 40 | Add `query_with_hypothesis()` method |
| 3.3 | `src/config.py` | 10 | Add `[search.hyde]` config section |
| 3.4 | Tests | 40 | Integration tests for HyDE tool |

**Dependencies:** Phase 2 (uses normalized fusion scores)

---

## 3. File Manifest

### Files to Create

| File | Purpose |
|------|---------|
| `tests/unit/test_edge_types.py` | Unit tests for EdgeType inference |
| `tests/unit/test_community_detection.py` | Unit tests for Leiden algorithm integration |
| `tests/unit/test_dynamic_fusion.py` | Unit tests for variance-aware weights |
| `tests/integration/test_hyde_search.py` | Integration tests for HyDE tool |

### Files to Modify

| File | Changes |
|------|---------|
| [src/models.py](../../src/models.py) | Add `EdgeType` enum |
| [src/indices/graph.py](../../src/indices/graph.py) | Add community detection, typed edge queries |
| [src/parsers/markdown.py](../../src/parsers/markdown.py) | Infer edge types from header context |
| [src/indexing/manager.py](../../src/indexing/manager.py) | Pass header context to graph edges |
| [src/search/fusion.py](../../src/search/fusion.py) | Add variance calculation, dynamic weights |
| [src/search/orchestrator.py](../../src/search/orchestrator.py) | Integrate community boost and dynamic weights |
| [src/search/classifier.py](../../src/search/classifier.py) | Extend for variance signals |
| [src/mcp_server.py](../../src/mcp_server.py) | Add `search_with_hypothesis` tool |
| [src/config.py](../../src/config.py) | Add `GraphConfig`, `FusionConfig`, `HydeConfig` |
| [pyproject.toml](../../pyproject.toml) | Add `cdlib` dependency |

---

## 4. Dependency Graph

```
Phase 1 (Graph)          Phase 2 (Fusion)
     │                        │
     │                        │
     └────────────┬───────────┘
                  │
                  ▼
           Phase 3 (HyDE)
```

- Phases 1 and 2 are **independent** and can be developed in parallel.
- Phase 3 depends on Phase 2 (uses normalized fusion scores for HyDE results).

---

## 5. Function Signatures

### Phase 1: Graph Upgrade

```python
# src/models.py
class EdgeType(Enum):
    LINKS_TO = "links_to"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    RELATED = "related"

# src/indices/graph.py
class GraphStore:
    def detect_communities(self) -> dict[str, int]:
        ...

    def get_community(self, doc_id: str) -> int | None:
        ...

    def get_community_members(self, community_id: int) -> list[str]:
        ...

    def boost_by_community(
        self,
        doc_ids: list[str],
        seed_doc_ids: set[str],
        boost_factor: float = 1.1,
    ) -> dict[str, float]:
        ...

# src/parsers/markdown.py
def infer_edge_type(header_context: str, target: str) -> EdgeType:
    ...
```

### Phase 2: Fusion Logic

```python
# src/search/fusion.py
def normalize_strategy_scores(
    scores: list[float],
) -> list[float]:
    ...

def calculate_variance(scores: list[float]) -> float:
    ...

def compute_dynamic_weights(
    vector_scores: list[float],
    keyword_scores: list[float],
    base_vector_weight: float,
    base_keyword_weight: float,
    variance_threshold: float = 0.1,
) -> tuple[float, float]:
    ...

def fuse_results_v2(
    results: dict[str, list[tuple[str, float]]],
    k: int,
    base_weights: dict[str, float],
    modified_times: dict[str, float],
    use_dynamic_weights: bool = True,
) -> list[tuple[str, float]]:
    ...
```

### Phase 3: HyDE Support

```python
# src/search/orchestrator.py
class SearchOrchestrator:
    async def query_with_hypothesis(
        self,
        hypothesis: str,
        top_k: int = 10,
        top_n: int = 5,
    ) -> tuple[list[ChunkResult], CompressionStats]:
        ...

# src/mcp_server.py (tool schema)
Tool(
    name="search_with_hypothesis",
    description="Search documentation using a hypothesis about what the answer might look like...",
    inputSchema={
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string", "description": "Hypothesis about the expected answer"},
            "top_n": {"type": "integer", "default": 5},
        },
        "required": ["hypothesis"],
    },
)
```

---

## 6. Acceptance Criteria

### Phase 1: Graph Upgrade

| ID | Criterion | Verification |
|----|-----------|--------------|
| P1-AC1 | `EdgeType` enum exists with 4 values | Unit test |
| P1-AC2 | `GraphStore.add_edge()` accepts `edge_type` parameter | Unit test |
| P1-AC3 | Links in `# Testing` sections create `TESTS` edges | Unit test |
| P1-AC4 | Links in `# Implementation` sections create `IMPLEMENTS` edges | Unit test |
| P1-AC5 | `detect_communities()` returns community assignments | Unit test |
| P1-AC6 | Community data persists across restarts | Integration test |
| P1-AC7 | Search results boost same-community documents by configurable factor | Integration test |

### Phase 2: Fusion Logic

| ID | Criterion | Verification |
|----|-----------|--------------|
| P2-AC1 | `normalize_strategy_scores()` returns values in [0, 1] | Unit test |
| P2-AC2 | `calculate_variance()` returns 0 for identical scores | Unit test |
| P2-AC3 | Low variance reduces weight for that strategy | Unit test |
| P2-AC4 | `fuse_results_v2()` produces same order as `fuse_results()` when variance is high | Unit test |
| P2-AC5 | Dynamic weights enabled via config flag | Unit test |
| P2-AC6 | Search quality improves on benchmark queries (manual verification) | Manual test |

### Phase 3: HyDE Support

| ID | Criterion | Verification |
|----|-----------|--------------|
| P3-AC1 | `search_with_hypothesis` tool registered in MCP | E2E test |
| P3-AC2 | Tool embeds hypothesis text (not raw query) | Integration test |
| P3-AC3 | Vague query + good hypothesis finds relevant docs | Integration test |
| P3-AC4 | Tool respects `top_n` parameter | Unit test |
| P3-AC5 | Tool returns normalized scores in [0, 1] | Unit test |

---

## 7. Risk Register

| ID | Risk | Severity | Likelihood | Mitigation |
|----|------|----------|------------|------------|
| R1 | `cdlib` adds heavy dependencies (igraph, numpy) | Medium | High | Verify package size; consider vendoring Leiden only |
| R2 | Community detection slow on large graphs | Medium | Medium | Run async during `persist()`; cache results |
| R3 | Dynamic weights degrade search quality | High | Low | Feature flag; A/B testing; fallback to static |
| R4 | HyDE increases latency (extra embedding call) | Low | High | Document tradeoff; optional tool |
| R5 | Edge type inference misclassifies links | Medium | Medium | Conservative defaults (`LINKS_TO`); user override via frontmatter |
| R6 | Backwards compatibility with existing indices | Medium | Low | Graceful migration: missing edge types default to `LINKS_TO` |

---

## 8. Testing Strategy

### Unit Tests

**Phase 1:**
```python
# tests/unit/test_edge_types.py
def test_infer_edge_type_testing_section():
    """
    Links under '# Testing' header infer TESTS edge type.
    """

def test_infer_edge_type_implementation_section():
    """
    Links under '# Implementation' header infer IMPLEMENTS edge type.
    """

def test_infer_edge_type_default():
    """
    Links without context default to LINKS_TO.
    """

# tests/unit/test_community_detection.py
def test_detect_communities_empty_graph():
    """
    Empty graph returns empty community dict.
    """

def test_detect_communities_single_cluster():
    """
    Fully connected graph forms single community.
    """

def test_community_boost_same_community():
    """
    Documents in same community receive boost factor.
    """
```

**Phase 2:**
```python
# tests/unit/test_dynamic_fusion.py
def test_normalize_strategy_scores_min_max():
    """
    Scores normalize to [0, 1] range using min-max.
    """

def test_calculate_variance_flat_scores():
    """
    Identical scores yield variance of 0.
    """

def test_dynamic_weights_low_vector_variance():
    """
    Low vector variance reduces vector weight.
    """

def test_fuse_results_v2_backwards_compatible():
    """
    New fusion produces same ranking as old when variance high.
    """
```

**Phase 3:**
```python
# tests/integration/test_hyde_search.py
async def test_search_with_hypothesis_tool_registered():
    """
    Tool appears in list_tools response.
    """

async def test_search_with_hypothesis_embeds_hypothesis():
    """
    Hypothesis text is embedded, not literal query.
    """

async def test_search_with_hypothesis_finds_relevant_docs():
    """
    Vague query with good hypothesis returns relevant results.
    """
```

### Integration Tests

- **Graph persistence:** Index documents, restart server, verify communities loaded.
- **Fusion quality:** Benchmark queries against ground truth (manual curation).
- **HyDE latency:** Measure overhead of extra embedding call.

### E2E Tests

- **MCP protocol:** Verify `search_with_hypothesis` tool callable via MCP client.
- **Full flow:** Index → Search → HyDE → Verify results.

---

## 9. Configuration Schema

```toml
[search.graph]
community_detection_enabled = true
community_boost_factor = 1.1

[search.fusion]
dynamic_weights_enabled = true
variance_threshold = 0.1
min_weight_factor = 0.5

[search.hyde]
enabled = true
```

---

## 10. Migration Notes

1. **Existing graphs:** Load without community data; run `detect_communities()` on first persist.
2. **Edge types:** Missing `edge_type` attribute defaults to `LINKS_TO`.
3. **Config:** New config sections have sensible defaults; no breaking changes.
4. **Re-indexing:** Full re-index recommended to populate edge types from header context.

---

## 11. Open Questions

1. Should community detection run synchronously during persist or as background task?
2. What header keywords map to `IMPLEMENTS` vs `RELATED`?
3. Should HyDE tool allow custom embedding models?

---

## 12. References

- [Spec 17: Search Infrastructure Overhaul](../specs/17-search-overhaul.md)
- [Spec 16: Memory Management](../specs/16-memory-management.md) (edge attributes)
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [cdlib Documentation](https://cdlib.readthedocs.io/)
