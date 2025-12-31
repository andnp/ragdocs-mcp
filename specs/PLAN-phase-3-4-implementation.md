# Implementation Plan: Phase 3 & Phase 4 Search Quality Improvements

**Prepared for:** Code Agent
**Date:** 2025-01-01
**Spec Reference:** [specs/11-search-quality-improvements.md](11-search-quality-improvements.md)

---

## 1. Executive Summary

This plan covers implementation of 5 features across Phase 3 and Phase 4:

| Phase | Feature | Impact | Complexity | LOC Est. |
|-------|---------|--------|------------|----------|
| 3 | P3: Query Type Classification + Adaptive Weights | High | Medium | ~80 |
| 3 | P7: Code Block Index | Medium | Medium | ~200 |
| 4 | P11: Maximal Marginal Relevance (MMR) | High | Low | ~80 |
| 4 | P12: N-gram Overlap Deduplication | Medium | Low | ~60 |
| 4 | P13: Parent Document Retrieval | High | Medium | ~180 |

**Total Estimated LOC:** ~600

---

## 2. Dependency Graph

```
                    ┌─────────────────────────────────────┐
                    │       Phase 3 Features              │
                    └─────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    │                    ▼
┌─────────────────┐           │          ┌─────────────────┐
│ P3: Query Type  │           │          │ P7: Code Block  │
│ Classification  │           │          │ Index           │
│ (no deps)       │           │          │ (no deps)       │
└─────────────────┘           │          └─────────────────┘
                              │
                    ┌─────────────────────────────────────┐
                    │       Phase 4 Features              │
                    └─────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
┌───────────┐         ┌───────────────┐         ┌───────────────┐
│ P12: N-gram│         │ P11: MMR      │         │ P13: Parent   │
│ Pre-filter │────────▶│ Selection     │         │ Retrieval     │
│ (no deps)  │         │ (depends on   │         │ (no deps)     │
└───────────┘         │  existing     │         └───────────────┘
                      │  dedup.py)    │
                      └───────────────┘
```

### Dependency Analysis

| Feature | Depends On | Blocks |
|---------|------------|--------|
| P3: Query Classification | Nothing | Nothing |
| P7: Code Block Index | Nothing | Nothing |
| P11: MMR | `dedup.py` (existing) | Nothing |
| P12: N-gram Dedup | Nothing | P11 (pipeline order) |
| P13: Parent Retrieval | Nothing | Nothing |

**Key Insight:** P12 (N-gram) should be implemented before P11 (MMR) because the pipeline order is:
```
threshold → n-gram dedup (fast) → MMR/semantic dedup (slow) → doc-limit → rerank
```

---

## 3. Implementation Order

### Recommended Sequence

1. **P3: Query Type Classification** (Phase 3) - Foundation for adaptive search
2. **P12: N-gram Overlap Dedup** (Phase 4) - Fast pre-filter, minimal deps
3. **P11: MMR Selection** (Phase 4) - Replaces/augments semantic dedup
4. **P7: Code Block Index** (Phase 3) - New index, parallel searchable
5. **P13: Parent Document Retrieval** (Phase 4) - Most complex, requires reindex

### Rationale

- **P3 first:** Enables testing of adaptive weights throughout other feature development
- **P12 before P11:** N-gram dedup reduces candidates for MMR, improving performance
- **P7 independent:** Can be developed in parallel with P11/P12
- **P13 last:** Requires chunking changes and reindex; highest risk

---

## 4. File Manifest

### 4.1. P3: Query Type Classification + Adaptive Weights

#### Files to Create
| File | Purpose |
|------|---------|
| `src/search/classifier.py` | Query classification logic |
| `tests/unit/test_query_classifier.py` | Unit tests |

#### Files to Modify
| File | Changes |
|------|---------|
| [src/models.py](../src/models.py) | Add `QueryType` enum |
| [src/search/orchestrator.py](../src/search/orchestrator.py) | Call classifier, apply adaptive weights |

#### Functions to Add

**`src/search/classifier.py`:**
```
QueryType (Enum):
    FACTUAL
    NAVIGATIONAL
    EXPLORATORY

classify_query(query: str) -> QueryType
    - Regex-based heuristics
    - Factual: camelCase, snake_case, backticks, version numbers, quoted phrases
    - Navigational: "section", "chapter", "guide", wikilink patterns
    - Exploratory: question words (what, how, why), default

get_adaptive_weights(query_type: QueryType, base_weights: dict) -> dict
    - Returns modified weights based on query type
    - Factual: keyword_weight × 1.5
    - Navigational: graph_weight × 1.5
    - Exploratory: semantic_weight × 1.3
```

**`src/models.py`:**
```
class QueryType(Enum):
    FACTUAL = "factual"
    NAVIGATIONAL = "navigational"
    EXPLORATORY = "exploratory"
```

**`src/search/orchestrator.py`:**
- Import `classify_query`, `get_adaptive_weights` from `src.search.classifier`
- In `query()`: classify query, adjust weights before fusion

---

### 4.2. P7: Code Block Index

#### Files to Create
| File | Purpose |
|------|---------|
| `src/indices/code.py` | Dedicated Whoosh index for code blocks |
| `tests/unit/test_code_index.py` | Unit tests |

#### Files to Modify
| File | Changes |
|------|---------|
| [src/parsers/markdown.py](../src/parsers/markdown.py) | Extract code blocks during parsing |
| [src/models.py](../src/models.py) | Add `CodeBlock` dataclass |
| [src/search/orchestrator.py](../src/search/orchestrator.py) | Query code index, include in fusion |
| [src/indexing/manager.py](../src/indexing/manager.py) | Initialize and persist code index |

#### Functions to Add

**`src/indices/code.py`:**
```
CodeIndex (class):
    __init__()
        - Schema with code-aware analyzer
        - RegexTokenizer for camelCase splitting, preserve punctuation

    add_code_block(code_block: CodeBlock)
    remove_document(doc_id: str)
    search(query: str, top_k: int, language: str | None) -> list[dict]
    persist(path: Path)
    load(path: Path)
    clear()
```

**Code-aware analyzer:**
```python
from whoosh.analysis import RegexTokenizer, LowercaseFilter

# Split on camelCase boundaries, preserve punctuation in identifiers
code_tokenizer = RegexTokenizer(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+|\S+')
code_analyzer = code_tokenizer | LowercaseFilter()
```

**Schema:**
```python
Schema(
    id=ID(stored=True, unique=True),
    doc_id=ID(stored=True),
    chunk_id=ID(stored=True),  # Reference to parent text chunk
    code=TEXT(stored=True, analyzer=code_analyzer),
    language=KEYWORD(stored=True),
)
```

**`src/models.py`:**
```
@dataclass
class CodeBlock:
    id: str
    doc_id: str
    chunk_id: str
    code: str
    language: str
    start_line: int
    end_line: int
```

**`src/parsers/markdown.py`:**
```
_extract_code_blocks(content: str, doc_id: str) -> list[CodeBlock]
    - Regex: ```(\w+)?\n(.*?)\n```
    - Extract language, code content, line numbers
```

---

### 4.3. P11: Maximal Marginal Relevance (MMR)

#### Files to Create
| File | Purpose |
|------|---------|
| `src/search/diversity.py` | MMR selection algorithm |
| `tests/unit/test_mmr.py` | Unit tests |

#### Files to Modify
| File | Changes |
|------|---------|
| [src/search/pipeline.py](../src/search/pipeline.py) | Add MMR as alternative to semantic dedup |
| [src/config.py](../src/config.py) | Add `mmr_enabled`, `mmr_lambda` options |

#### Functions to Add

**`src/search/diversity.py`:**
```
select_mmr(
    query_embedding: list[float],
    candidates: list[tuple[str, float]],
    get_embedding: Callable[[str], list[float] | None],
    lambda_param: float = 0.7,
    top_n: int = 10,
) -> list[tuple[str, float]]

    Algorithm:
    1. Start with empty selected set
    2. While len(selected) < top_n and candidates remain:
       a. For each candidate, compute:
          mmr_score = λ × relevance - (1-λ) × max_sim_to_selected
       b. Select candidate with highest MMR score
       c. Move from candidates to selected
    3. Return selected with original scores
```

**`src/config.py` additions:**
```python
@dataclass
class SearchConfig:
    # ... existing fields ...
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
```

**`src/search/pipeline.py` changes:**
- Import `select_mmr` from `src.search.diversity`
- Add `mmr_enabled`, `mmr_lambda` to `SearchPipelineConfig`
- In `process()`: if `mmr_enabled`, use MMR instead of `deduplicate_by_similarity`

---

### 4.4. P12: N-gram Overlap Deduplication

#### Files to Modify
| File | Changes |
|------|---------|
| [src/search/dedup.py](../src/search/dedup.py) | Add n-gram functions |
| [src/search/pipeline.py](../src/search/pipeline.py) | Insert n-gram dedup in pipeline |
| [src/config.py](../src/config.py) | Add `ngram_dedup_enabled`, `ngram_dedup_threshold` |
| `tests/unit/test_dedup.py` | Add tests (create if not exists) |

#### Functions to Add

**`src/search/dedup.py`:**
```
get_char_ngrams(text: str, n: int = 3) -> set[str]
    - Lowercase, remove spaces
    - Return set of character n-grams

jaccard_similarity(ngrams_a: set[str], ngrams_b: set[str]) -> float
    - |intersection| / |union|

deduplicate_by_ngram(
    results: list[tuple[str, float]],
    get_content: Callable[[str], str | None],
    threshold: float = 0.7,
    n: int = 3,
) -> tuple[list[tuple[str, float]], int]
    - For each result, compare against kept results
    - If Jaccard >= threshold, skip (it's a duplicate)
    - Return (kept_results, removed_count)
```

**`src/config.py` additions:**
```python
@dataclass
class SearchConfig:
    # ... existing fields ...
    ngram_dedup_enabled: bool = True  # On by default (fast)
    ngram_dedup_threshold: float = 0.7
```

**Pipeline order in `src/search/pipeline.py`:**
```
normalize → threshold → content_hash_dedup → ngram_dedup → semantic_dedup/MMR → doc_limit → rerank
```

---

### 4.5. P13: Parent Document Retrieval

#### Files to Create
| File | Purpose |
|------|---------|
| `tests/unit/test_parent_retrieval.py` | Unit tests |

#### Files to Modify
| File | Changes |
|------|---------|
| [src/parsers/markdown.py](../src/parsers/markdown.py) | Two-level chunking |
| [src/models.py](../src/models.py) | Add parent fields to `Chunk` |
| [src/indices/vector.py](../src/indices/vector.py) | Store parent reference in metadata |
| [src/search/orchestrator.py](../src/search/orchestrator.py) | Expand to parents before returning |
| [src/config.py](../src/config.py) | Add `parent_retrieval_enabled` |
| `src/chunking/header_chunker.py` | Modify to produce two-level chunks |

#### Data Model Changes

**`src/models.py` - Chunk dataclass:**
```python
@dataclass
class Chunk:
    # ... existing fields ...
    parent_chunk_id: str | None = None  # Reference to parent section
    is_parent: bool = False  # True if this is a parent section chunk
```

**Two-Level Chunking Strategy:**

| Level | Size | Overlap | Purpose |
|-------|------|---------|---------|
| Parent (Section) | 1500-2000 chars | 200 chars | Return unit |
| Child (Sub-chunk) | 400-600 chars | 100 chars | Retrieval unit |

**`src/config.py` additions:**
```python
@dataclass
class ChunkingConfig:
    # ... existing fields ...
    parent_retrieval_enabled: bool = False
    parent_chunk_min_chars: int = 1500
    parent_chunk_max_chars: int = 2000
    child_chunk_min_chars: int = 400
    child_chunk_max_chars: int = 600
```

**`src/search/orchestrator.py` changes:**
```
_expand_to_parents(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]
    - For each child chunk, get parent_chunk_id from metadata
    - Deduplicate parents (multiple children may share parent)
    - Return parent chunks with highest child score
```

---

## 5. Testing Strategy

### 5.1. Unit Tests

| Feature | Test File | Test Cases |
|---------|-----------|------------|
| P3 | `tests/unit/test_query_classifier.py` | `test_classify_factual_camelcase`, `test_classify_factual_snake_case`, `test_classify_factual_backticks`, `test_classify_navigational_section`, `test_classify_navigational_wikilink`, `test_classify_exploratory_question`, `test_classify_exploratory_default`, `test_adaptive_weights_factual`, `test_adaptive_weights_navigational`, `test_adaptive_weights_exploratory` |
| P7 | `tests/unit/test_code_index.py` | `test_camelcase_tokenization`, `test_snake_case_tokenization`, `test_preserve_punctuation`, `test_search_function_name`, `test_search_class_name`, `test_language_filter`, `test_persist_and_load` |
| P11 | `tests/unit/test_mmr.py` | `test_mmr_selects_diverse`, `test_mmr_respects_lambda_high`, `test_mmr_respects_lambda_low`, `test_mmr_empty_candidates`, `test_mmr_single_candidate`, `test_mmr_all_identical` |
| P12 | `tests/unit/test_dedup.py` | `test_ngram_extraction`, `test_jaccard_identical`, `test_jaccard_disjoint`, `test_jaccard_partial`, `test_ngram_dedup_removes_exact`, `test_ngram_dedup_keeps_different`, `test_ngram_dedup_threshold_boundary` |
| P13 | `tests/unit/test_parent_retrieval.py` | `test_two_level_chunking`, `test_parent_reference_stored`, `test_expand_to_parents`, `test_parent_dedup_across_children`, `test_parent_retrieval_disabled` |

### 5.2. Integration Tests

| Test Case | Features Covered | Description |
|-----------|------------------|-------------|
| `test_adaptive_weights_improve_factual` | P3 | Query with function name ranks higher with adaptive weights |
| `test_code_search_finds_function` | P7 | Query `getAuthToken` finds exact function in code index |
| `test_mmr_vs_threshold_diversity` | P11 | MMR produces more diverse results than threshold dedup |
| `test_ngram_before_semantic` | P12 | N-gram dedup removes exact duplicates, semantic handles near-dupes |
| `test_parent_returns_context` | P13 | Search returns parent section containing matched child chunk |
| `test_full_pipeline_with_all_features` | All | End-to-end test with all features enabled |

### 5.3. Test Fixtures

Create `tests/fixtures/` with:
- `code_samples/` - Python, JavaScript, Rust code blocks
- `markdown_docs/` - Sample documents with headers, code, and varied content
- `expected_results/` - Ground truth for integration tests

---

## 6. Risk Assessment

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | MMR latency too high with many candidates | Medium | Medium | Limit MMR input to top-50 after n-gram dedup |
| R2 | Code tokenizer doesn't handle all languages | Medium | Low | Start with Python/JS/Rust; document limitations |
| R3 | Parent retrieval breaks existing chunk IDs | High | High | Maintain backward compat: child chunks keep original IDs |
| R4 | N-gram dedup too aggressive | Medium | Medium | Conservative default threshold (0.7); user-configurable |
| R5 | Query classifier misclassifies hybrid queries | Medium | Low | Allow manual override in API; default to exploratory |
| R6 | Reindex required for P7 and P13 | High | Low | Document clearly; provide `rebuild-index` command |
| R7 | Code index increases storage significantly | Low | Low | Monitor index size; consider optional feature flag |

### Blockers

1. **P13 (Parent Retrieval):** Requires changes to chunking strategy. Existing indices become incompatible. Must provide migration path.

2. **P7 (Code Index):** New Whoosh index. If `IndexManager` doesn't support multiple indices cleanly, may need refactoring.

---

## 7. Configuration Summary

### New Config Options

```toml
[search]
# P3: Query Classification (no config needed - always on)

# P11: MMR Diversity
mmr_enabled = false
mmr_lambda = 0.7  # 0.0=diversity only, 1.0=relevance only

# P12: N-gram Dedup
ngram_dedup_enabled = true  # On by default (fast)
ngram_dedup_threshold = 0.7

# P13: Parent Retrieval
parent_retrieval_enabled = false

[chunking]
# P13: Two-level chunking sizes
parent_chunk_min_chars = 1500
parent_chunk_max_chars = 2000
child_chunk_min_chars = 400
child_chunk_max_chars = 600
```

### SearchPipelineConfig Updates

```python
@dataclass
class SearchPipelineConfig:
    # Existing
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 0
    dedup_enabled: bool = True
    dedup_threshold: float = 0.85
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10

    # P11: MMR
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7

    # P12: N-gram
    ngram_dedup_enabled: bool = True
    ngram_dedup_threshold: float = 0.7
```

---

## 8. Implementation Checklist

### Phase 3

- [ ] **P3: Query Type Classification**
  - [ ] Create `src/search/classifier.py`
  - [ ] Add `QueryType` enum to `src/models.py`
  - [ ] Integrate into `src/search/orchestrator.py`
  - [ ] Create `tests/unit/test_query_classifier.py`
  - [ ] Run tests: `uv run pytest tests/unit/test_query_classifier.py -v`

- [ ] **P7: Code Block Index**
  - [ ] Create `src/indices/code.py`
  - [ ] Add `CodeBlock` dataclass to `src/models.py`
  - [ ] Add code extraction to `src/parsers/markdown.py`
  - [ ] Update `src/indexing/manager.py`
  - [ ] Integrate into `src/search/orchestrator.py`
  - [ ] Create `tests/unit/test_code_index.py`
  - [ ] Run tests: `uv run pytest tests/unit/test_code_index.py -v`

### Phase 4

- [ ] **P12: N-gram Overlap Deduplication**
  - [ ] Add functions to `src/search/dedup.py`
  - [ ] Add config options to `src/config.py`
  - [ ] Update `src/search/pipeline.py`
  - [ ] Add tests to `tests/unit/test_dedup.py`
  - [ ] Run tests: `uv run pytest tests/unit/test_dedup.py -v`

- [ ] **P11: MMR Selection**
  - [ ] Create `src/search/diversity.py`
  - [ ] Add config options to `src/config.py`
  - [ ] Update `src/search/pipeline.py`
  - [ ] Create `tests/unit/test_mmr.py`
  - [ ] Run tests: `uv run pytest tests/unit/test_mmr.py -v`

- [ ] **P13: Parent Document Retrieval**
  - [ ] Update `src/models.py` with parent fields
  - [ ] Modify chunking in `src/chunking/header_chunker.py`
  - [ ] Update `src/indices/vector.py` metadata
  - [ ] Add parent expansion to `src/search/orchestrator.py`
  - [ ] Add config options to `src/config.py`
  - [ ] Create `tests/unit/test_parent_retrieval.py`
  - [ ] Run tests: `uv run pytest tests/unit/test_parent_retrieval.py -v`

### Final Validation

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Run type checker: `uv run pyright src/`
- [ ] Run linter: `uv run ruff check src/`
- [ ] Test reindex: `uv run mcp-markdown-ragdocs rebuild-index`

---

## 9. Code Style Constraints

Per project instructions:

1. **No docstrings** - Code should be self-documenting
2. **No return type annotations** - Rely on type inference
3. **Absolute imports only** - Use `from src.x import y`
4. **Real implementation tests** - Avoid mocks where possible
5. **Zero type/lint errors** - Must pass `ruff` and `pyright`

---

## 10. Handoff Notes

### For Code Agent

1. **Start with P3** - It's self-contained and enables adaptive weight testing
2. **P12 before P11** - The pipeline order matters
3. **P7 can be parallel** - No dependencies on other Phase 4 features
4. **P13 is risky** - Test thoroughly before merging; consider feature flag

### Key Files to Keep Open

- [src/search/pipeline.py](../src/search/pipeline.py) - Central pipeline logic
- [src/search/orchestrator.py](../src/search/orchestrator.py) - Query coordination
- [src/config.py](../src/config.py) - All config options
- [src/models.py](../src/models.py) - Data models

### Questions to Resolve During Implementation

1. Should MMR replace semantic dedup or be an alternative? (Recommend: alternative via config)
2. Should code index be queried in parallel with text indices? (Recommend: yes)
3. Should parent retrieval return child content as well? (Recommend: no, just parent)
