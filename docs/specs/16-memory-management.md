# Feature Spec 16: Memory Management System

**Status:** ✅ Implemented

**Implementation Date:** 2026-01-11

**Implementation Notes:**
- Core infrastructure in `src/memory/` module
- CRUD tools in `src/memory/tools.py`
- Search orchestration in `src/memory/search.py`
- Index management in `src/memory/manager.py`
- Ghost node and typed edge support in GraphStore
- Memory-specific recency boost algorithm
- All 9 tools exposed via MCP server

## 1. Overview
This feature introduces a **Memory Management System** to the MCP Server. It allows AI assistants to maintain a persistent, project-specific "Memory Bank" of markdown files. These memories are stored separately from the main document corpus but use the same powerful hybrid search (Vector + Keyword + Graph) infrastructure.

**Key Goals:**
- **Separation of Concerns**: Memories are a distinct corpus; they do not pollute the main document search.
- **Cross-Corpus Linking**: Memories can link to main documents (e.g., `[[src/server.py]]`), enabling "What memories link to this file?" queries.
- **CRUD Tools**: Full suite of tools for AI to manage its own long-term state.
- **Flexible Storage**: Configurable storage location (Global `~/.local` vs. Project-local `.memories/`).

## 2. Architecture

### A. The "Dual-Lane" Pattern
We will replicate the core indexing/search stack to create a parallel "Memory Lane".

| Component | Main Corpus (Existing) | Memory Corpus (New) |
| :--- | :--- | :--- |
| **Source** | `docs/**/*.md` | `memories/**/*.md` (or `~/.local/...`) |
| **IndexStorage** | `indices/vector`, `indices/keyword` | `memories/indices/vector`, `memories/indices/keyword` |
| **Orchestrator** | `MainSearchOrchestrator` | `MemorySearchOrchestrator` |
| **Graph** | `MainGraph` (Nodes = Docs) | `MemoryGraph` (Nodes = Memories + Ghost Docs) |

### B. The Federated Graph (Ghost Nodes & Typed Edges)
To support querying "Memories linked to Document X" with high fidelity:
1.  **Memory Graph**: Contains nodes for Memory files (e.g., `memory-01.md`).
2.  **Ghost Nodes**: When a memory contains `[[src/server.py]]`, we create a node `ghost:src/server.py` in the **Memory Graph**.
3.  **Typed Edges & Context**:
    - Edges should carry a `type` (e.g., `mentions`, `refactors`, `plans`) derived from context if possible, or default to `related_to`.
    - Edges store **Anchor Context**: The surrounding ~100 characters of text around the link.
4.  **Query**: `search_linked_memories("bug fix", "src/server.py")` performs a graph traversal on the Memory Graph starting from `ghost:src/server.py`, filtering edges by context relevance.

### C. Chunking & Storage
- **Format**: Standard Markdown with YAML Frontmatter.
- **Chunking**: Use `HeaderChunker`.
    - *Why*: Allows large memory files (journals) to be retrievable by specific sections (headers).
    - *Constraint*: AI instructed to keep headers atomic.
- **Indices**:
    - **Vector/Keyword**: Standard.
    - **Memory-Specific**: Implement **Chronological Boosting** in the fusion layer. Memories from the last 7 days get a 1.2x score boost (configurable).

## 3. Configuration

New section in `config.toml`:

```toml
[memory]
enabled = true
storage_strategy = "user" # "project" or "user"
recency_boost_days = 7
recency_boost_factor = 1.2
```

## 4. Data Schema

### Frontmatter
```yaml
---
type: "plan"       # plan | journal | fact | observation | reflection
status: "active"   # active | archived
tags: ["refactor", "auth"]
created_at: "2023-10-27T10:00:00Z"
---
```

## 5. Tool Definitions

The following tools will be exposed via MCP:

### A. CRUD
1.  **`create_memory(filename: str, content: str, tags: List[str], type: str = "journal")`**
    - Creates a new file. Fails if exists.
2.  **`append_memory(filename: str, content: str)`**
    - Appends text to the end of the file.
3.  **`read_memory(filename: str)`**
    - Returns full content.
4.  **`update_memory(filename: str, content: str)`**
    - **Full replacement** (simplest for V1).
5.  **`delete_memory(filename: str)`**
    - Moves to `.trash/` (safety).

### B. Search
6.  **`search_memories(query: str, limit: int = 5, filter_tags: List[str] = [], filter_type: str | None = None)`**
    - Standard hybrid search on Memory Lane.
    - **Recency Boost**: Applied automatically.
7.  **`search_linked_memories(query: str, target_document: str)`**
    - Finds memories that explicitly link to `target_document`.
    - Uses Graph Index (Ghost Nodes).
    - Returns the **Anchor Context** from the link edge to explain *why* it's linked.

### C. Maintenance
8.  **`get_memory_stats()`**
    - Returns: `{ count: 12, total_size: "45KB", tags: {"bug": 5}, types: {"plan": 2} }`
9.  **`merge_memories(source_files: List[str], target_file: str, summary_content: str)`**
    - Reads sources, writes `summary_content` to `target_file`, deletes sources.

## 6. Implementation Plan

### Phase 1: Core Infrastructure
1.  **Config**: Update `Config` model to include `MemoryConfig`.
2.  **Manager**: Refactor `IndexManager` to be instantiable with a `base_path`.
3.  **Context**: Update `ApplicationContext` to hold `main_manager` and `memory_manager`.

### Phase 2: Graph & Indexing
4.  **Ghost Nodes**: Update `GraphIndex` to handle "external" links and store **Edge Attributes** (type, context).
    - *Refactor*: `GraphStore.add_edge` needs to support arbitrary kwargs for attributes.
5.  **Boosting**: Modify `fuse_results` (or create `memory_fuse_results`) to accept a `recency_config`.

### Phase 3: Tools & MCP
6.  **Tool Impl**: Implement the CRUD functions in `src/memory/tools.py`.
7.  **Registration**: Register new tools in `src/mcp_server.py`.

### Phase 4: Testing
8.  **Integration**: Test `memory_manager` interacting with `main_manager`.
9.  **E2E**: Verify MCP calls persist data and return correct search results.
