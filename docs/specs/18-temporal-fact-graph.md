# Specification: Temporal Fact Graph (Project #3)

## 1. Overview
The **Temporal Fact Graph** transforms the memory system from a flat key-value store into a lineage-aware knowledge base. It introduces the concept of *time* and *supersession* to the memory graph, allowing the system to understand that Fact A (e.g., "Use SQLite") has been replaced by Fact B (e.g., "Use PostgreSQL"), without losing the historical context that Fact A was once true.

## 2. Problem Statement
Currently, if a user creates two conflicting memories:
1. `journal-01.md`: "We decided to use SQLite."
2. `journal-02.md`: "We are migrating to PostgreSQL."

The search engine treats these as equally valid "facts" about the database. A query for "database choice" might return both with high confidence, confusing the user. There is no structural way to indicate that the second memory *supersedes* the first.

## 3. Goals
- **Explicit Lineage:** Allow memories to explicitly supersede or deprecate others.
- **Conflict Detection:** Proactively identify contradicting facts in the memory bank.
- **Time-Aware Ranking:** Downrank superseded information in search results, but keep it available for "history" queries.
- **Audit Trail:** Maintain a complete history of decisions (ADRs) and their evolution.

## 4. Data Model Changes

### 4.1 New Edge Types
We will extend the Graph Index with semantic edge types specifically for temporal relationships:

| Edge Type | Direction | Description |
|-----------|-----------|-------------|
| `SUPERSEDES` | New → Old | The source memory replaces the target memory as the current source of truth. |
| `DEPRECATES` | New → Old | The source memory marks the target as obsolete but not necessarily replaced (e.g., removing a feature). |
| `AMENDS` | New → Old | The source memory updates or corrects part of the target without fully replacing it. |
| `CONTRADICTS` | (Undirected) | System-detected potential conflict between two memories (inferred, not user-authored). |

### 4.2 Graph Node Attributes
Nodes in the `GraphStore` will gain new attributes:
- `is_superseded` (bool): True if any incoming `SUPERSEDES` edge exists.
- `valid_from` (datetime): Creation time.
- `valid_until` (datetime): Time when the first `SUPERSEDES` edge was created pointing to this node.

## 5. Logic & Algorithms

### 5.1 Conflict Detection (`detect_contradictions`)
A background or on-demand process to find potential conflicts.

**Algorithm:**
1. **Cluster:** Group memories by similarity (using `suggest_memory_merges` logic).
2. **Type Check:** Focus on `fact` and `plan` types (ignore `journal` as it's episodic).
3. **LLM Verification:** For tight clusters, use an LLM (via MCP delegation or internal call) to compare statements:
   - "Does Memory A contradict Memory B?"
   - "Does Memory B imply Memory A is obsolete?"
4. **Flagging:** If conflict found, create a `CONTRADICTS` edge and notify the user.

### 5.2 Search Impact
The `MemorySearchOrchestrator` will be updated to respect lineage:

- **Standard Query:**
  - Exclude or heavily penalize nodes where `is_superseded = True`.
  - *Boost* the "head" of a supersession chain.
- **History Query:** (e.g., "history of database decisions")
  - Include superseded nodes.
  - Sort by `valid_from`.

### 5.3 User Workflow

**Resolving a Conflict:**
1. User sees search result with "Potential Conflict" warning (or runs `detect_contradictions`).
2. User decides `memory-02.md` is the new truth.
3. User adds link to `memory-02.md`:
   ```markdown
   ...
   Supersedes: [[memory:journal-01]]
   ...
   ```
4. **Link Parser Update:** The system parses `Supersedes: [[...]]` (case-insensitive) and creates the `SUPERSEDES` edge.
5. **Graph Update:** The system marks `journal-01` node as `is_superseded=True`.

## 6. Implementation Plan

### Phase 1: Graph Schema & Link Parsing
- [ ] Update `ExtractedLink` model to support `edge_type` extraction from context (e.g., "Supersedes: [[...]]").
- [ ] Update `MemoryIndexManager._add_ghost_nodes_and_edges` to handle these specific types.

### Phase 2: Search Logic
- [ ] Modify `MemorySearchOrchestrator` to filter superseded items by default.
- [ ] Add `include_obsolete` flag to `search_memories` tool.

### Phase 3: Detection Tooling
- [ ] Implement `detect_contradictions` tool using vector similarity + lightweight LLM check (if available) or pure heuristic (high similarity + different timestamps).

## 7. Future Work
- Visualizing the decision tree (Graphviz export).
- Auto-suggesting `AMENDS` links for small diffs.
