# Feature Spec 17: Search Infrastructure Overhaul

**Status:** ✅ Complete (January 2026)

## Implementation Notes

| Feature | Planned | Implemented | Deviation |
|---------|---------|-------------|-----------|
| Edge Types | 4 types (LINKS_TO, IMPLEMENTS, TESTS, RELATED) | 4 types via `edge_type` attribute | None |
| Community Detection | Leiden Algorithm | Louvain Algorithm (NetworkX) | Louvain used (no additional dependencies) |
| Score-Aware Fusion | Weighted CombSUM + RRF | RRF + variance-based weight adjustment | Equivalent approach |
| HyDE | `search_with_hypothesis` tool | Implemented | None |
| Configuration | `[search.advanced]` section | Implemented | None |

**Algorithm Deviation:** The spec proposed Leiden algorithm via `cdlib`. Implementation uses Louvain algorithm from NetworkX (`nx.community.louvain_communities`), avoiding additional dependencies. Both algorithms produce comparable community structures for document graphs.

---

## 1. Overview
This specification details the roadmap for upgrading the core search infrastructure of `mcp-markdown-ragdocs` to align with 2025 state-of-the-art techniques (GraphRAG, Score-Aware Fusion, HyDE). These changes apply to **both** the Main Corpus and the Memory Corpus, though some features (Community Detection) are particularly impactful for large documentation bases.

## 2. Goals
1.  **Semantic Graph**: Move from "dumb" file links to typed relationships with community detection.
2.  **Advanced Fusion**: Replace static RRF with Score-Aware Fusion to better handle variance in vector/keyword confidence.
3.  **Hypothesis-Driven Search**: Enable HyDE (Hypothetical Document Embeddings) for handling vague queries.

## 3. Architecture Upgrades

### A. Graph Index: Typed Edges & Community Detection
**Current**: `GraphStore` stores `A -> B` (unweighted, untyped).
**New**:
1.  **Edge Schema**:
    ```python
    class EdgeType(Enum):
        LINKS_TO = "links_to"      # Default
        IMPLEMENTS = "implements"  # [[src/foo.py]] in "Implementation" section
        TESTS = "tests"            # [[src/foo.py]] in tests/test_foo.py
        RELATED = "related"        # "See also"
    ```
2.  **Community Detection (Leiden Algorithm)**:
    - Periodically (on index save), run the **Leiden Algorithm** (via `cdlib` or similar) to detect clusters.
    - **Storage**: Store `community_id` on each Node.
    - **Search**: When a node is retrieved, boost other nodes in the same `community_id` by a factor (e.g., 1.1x).
    - *Why*: If a user searches for "Auth", and we find `auth.md`, we should also boost `login.md` and `user.md` even if they don't share exact keywords, because they form a tight graph cluster.

### B. Fusion Layer: Score-Aware RRF
**Current**: Standard RRF ($1 / (k + rank)$). Ignores the fact that Vector score might be 0.99 (high confidence) or 0.60 (low confidence).
**New**: **Weighted CombSUM + RRF**
- **Logic**:
    - Normalize Vector scores ($0..1$).
    - Normalize Keyword scores ($0..1$).
    - Calculate **Variance**: If Vector scores are flat (low variance), implies "muddy" semantic match. Lower the Vector weight.
    - Fuse: $FinalScore = (W_v \cdot S_v) + (W_k \cdot S_k) + RRF_{score}$

### C. Search Pipeline: HyDE Support
**Concept**: For vague queries like "How do I add a tool?", semantic search might fail against dry documentation.
**Implementation**:
- **Tool**: `search_with_hypothesis(hypothesis: str)`
- **Workflow**:
    1.  AI generates hypothesis: *"To add a tool, I likely need to modify src/mcp_server.py and register it in the list_tools method..."*
    2.  Server embeds this hypothesis.
    3.  Server searches Vector Index with this embedding.
    4.  *Result*: Finds `src/mcp_server.py` even if the query was vague.

## 4. Implementation Plan

### Phase 1: Graph Upgrade
1.  **Parser Update**: Modify `MarkdownParser` to infer edge types based on context (Header section name).
    - E.g., Links under `# Testing` -> `TESTS`.
2.  **GraphStore Update**: Update `nx.DiGraph` to store edge attributes.
3.  **Community Algo**: Integrate a community detection library. Run during `IndexManager.persist()`.

### Phase 2: Fusion Logic
4.  **Math**: Implement `normalize_scores` and `calculate_variance` in `src/search/fusion.py`.
5.  **Dynamic Weights**: Update `SearchOrchestrator` to compute weights dynamically per-query instead of just using config defaults.

### Phase 3: HyDE
6.  **Tool**: Expose `search_with_hypothesis` in `MCPServer`.
    - *Note*: The underlying `VectorIndex.search` already takes a string; we just need to pass the hypothesis instead of the raw query.

## 5. Migration Strategy
- **Graph**: Existing graphs are compatible (edges just lack types). A full re-index will populate types.
- **Config**: Add `[search.advanced]` section to toggle Community Detection and HyDE features.
