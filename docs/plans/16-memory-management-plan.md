# Implementation Plan: Memory Management System (Spec 16)

## Executive Summary

This plan implements a dual-lane Memory Management System enabling AI assistants to maintain project-specific Memory Banks separate from the main documentation corpus. The architecture replicates the existing Vector/Keyword/Graph indexing stack with memory-specific extensions: Ghost Nodes for cross-corpus linking, Typed Edges with anchor context, and configurable Recency Boost. Nine MCP tools provide CRUD, search, and maintenance operations. Implementation spans 5 phases (~1,800 LOC) with each phase delivering testable increments.

---

## 1. Implementation Phases

### Phase 1: Configuration & Storage Foundation
**Goal**: Extend configuration and establish memory storage paths.
**LOC Estimate**: ~200

| Step | Description |
|------|-------------|
| 1.1 | Add `MemoryConfig` dataclass to [src/config.py](../../src/config.py) |
| 1.2 | Add `[memory]` section parsing in `load_config()` |
| 1.3 | Create `resolve_memory_path()` function for project vs user storage |
| 1.4 | Add `memory` field to `Config` dataclass |

**Acceptance Criteria**:
- [ ] `config.memory.enabled` returns bool
- [ ] `config.memory.storage_strategy` is `"project"` or `"user"`
- [ ] `resolve_memory_path()` returns correct path for both strategies
- [ ] Config loads from TOML without errors

---

### Phase 2: Memory Index Manager
**Goal**: Create a dedicated `MemoryIndexManager` that mirrors `IndexManager` patterns.
**LOC Estimate**: ~350

| Step | Description |
|------|-------------|
| 2.1 | Create `src/memory/` package directory |
| 2.2 | Create `src/memory/manager.py` with `MemoryIndexManager` class |
| 2.3 | Implement `_compute_memory_id()` with `memory:` prefix |
| 2.4 | Implement `index_memory()`, `remove_memory()`, `persist()`, `load()` |
| 2.5 | Create `src/memory/models.py` with `MemoryDocument` model |
| 2.6 | Implement YAML frontmatter parsing (`type`, `status`, `tags`, `created_at`) |

**Acceptance Criteria**:
- [ ] `MemoryIndexManager` indexes memory files with correct ID prefix
- [ ] Frontmatter metadata extracted into chunk metadata
- [ ] Indices persist/load to `memories/indices/` subdirectory
- [ ] Memory documents use `HeaderChunker`

---

### Phase 3: Ghost Nodes & Typed Edges
**Goal**: Extend `GraphStore` for cross-corpus linking with context.
**LOC Estimate**: ~400

| Step | Description |
|------|-------------|
| 3.1 | Add `edge_context: str` parameter to `GraphStore.add_edge()` |
| 3.2 | Implement ghost node creation in `MemoryIndexManager` for `[[target]]` links |
| 3.3 | Extract ~100 chars anchor context around each link |
| 3.4 | Infer edge type from context keywords (`mentions`, `refactors`, `plans`, `related_to`) |
| 3.5 | Add `get_edges_to()` method for reverse lookup (ghost → memory) |
| 3.6 | Update `GraphStore.persist()/load()` to handle edge attributes |

**Acceptance Criteria**:
- [ ] Links like `[[src/server.py]]` create `ghost:src/server.py` node
- [ ] Edges store `edge_type` and `edge_context` attributes
- [ ] `get_edges_to("ghost:X")` returns all memories linking to X
- [ ] Edge attributes survive persist/load cycle

---

### Phase 4: Memory Search Orchestrator with Recency Boost
**Goal**: Implement memory-specific hybrid search with chronological boosting.
**LOC Estimate**: ~400

| Step | Description |
|------|-------------|
| 4.1 | Create `src/memory/search.py` with `MemorySearchOrchestrator` |
| 4.2 | Implement `search_memories()` with hybrid search (reuse fusion logic) |
| 4.3 | Add memory-specific recency boost tiers from config |
| 4.4 | Implement `search_linked_memories()` using ghost node traversal |
| 4.5 | Add tag/type filtering to search pipeline |
| 4.6 | Return anchor context in linked memory results |

**Acceptance Criteria**:
- [ ] `search_memories()` returns results with recency boost applied
- [ ] `search_linked_memories("query", "src/server.py")` returns memories linking to target
- [ ] Results include `anchor_context` field explaining link
- [ ] Tag/type filters narrow results correctly

---

### Phase 5: MCP Tools & Context Integration
**Goal**: Register all 9 tools and integrate with `ApplicationContext`.
**LOC Estimate**: ~450

| Step | Description |
|------|-------------|
| 5.1 | Update `ApplicationContext` to hold `memory_manager: MemoryIndexManager | None` |
| 5.2 | Initialize memory manager in `ApplicationContext.create()` when enabled |
| 5.3 | Create `src/memory/tools.py` with CRUD tool implementations |
| 5.4 | Implement `create_memory()`, `append_memory()`, `read_memory()`, `update_memory()` |
| 5.5 | Implement `delete_memory()` (move to `.trash/`) |
| 5.6 | Implement `search_memories()`, `search_linked_memories()` tool handlers |
| 5.7 | Implement `get_memory_stats()`, `merge_memories()` |
| 5.8 | Register all tools in `MCPServer.list_tools()` and `MCPServer.call_tool()` |

**Acceptance Criteria**:
- [ ] All 9 tools appear in MCP tool list
- [ ] CRUD operations modify files correctly
- [ ] Delete moves to `.trash/` instead of hard delete
- [ ] `merge_memories()` consolidates files and deletes sources
- [ ] `get_memory_stats()` returns accurate counts

---

## 2. File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/memory/__init__.py` | Package marker (empty) |
| `src/memory/manager.py` | `MemoryIndexManager` class |
| `src/memory/models.py` | `MemoryDocument`, `MemoryFrontmatter` models |
| `src/memory/search.py` | `MemorySearchOrchestrator` with recency boost |
| `src/memory/tools.py` | CRUD and search tool implementations |
| `src/memory/link_parser.py` | Extract `[[links]]` and anchor context |
| `tests/unit/test_memory_manager.py` | Unit tests for manager |
| `tests/unit/test_memory_link_parser.py` | Unit tests for link extraction |
| `tests/unit/test_memory_search.py` | Unit tests for search orchestrator |
| `tests/integration/test_memory_tools.py` | Integration tests for MCP tools |
| `tests/integration/test_memory_ghost_nodes.py` | Integration tests for graph linking |

### Modified Files

| File | Changes |
|------|---------|
| [src/config.py](../../src/config.py) | Add `MemoryConfig` dataclass, `[memory]` loading |
| [src/context.py](../../src/context.py) | Add `memory_manager` field, initialization |
| [src/mcp_server.py](../../src/mcp_server.py) | Register 9 memory tools |
| [src/indices/graph.py](../../src/indices/graph.py) | Add `edge_context` to `add_edge()`, `get_edges_to()` |
| [src/search/fusion.py](../../src/search/fusion.py) | Extract recency boost to reusable function |

---

## 3. Dependency Graph

```
Phase 1 (Config)
    │
    ▼
Phase 2 (Manager) ◄──────┐
    │                    │
    ▼                    │
Phase 3 (Graph) ─────────┘
    │
    ▼
Phase 4 (Search)
    │
    ▼
Phase 5 (Tools)
```

**Notes**:
- Phase 2 depends on Phase 1 for config access
- Phase 3 modifies shared `GraphStore` (used by Phase 2)
- Phase 4 uses components from Phase 2 and Phase 3
- Phase 5 integrates everything

---

## 4. Function Signatures

### Config (Phase 1)

```python
# src/config.py
@dataclass
class MemoryConfig:
    enabled: bool
    storage_strategy: str  # "project" | "user"
    recency_boost_days: int
    recency_boost_factor: float

def resolve_memory_path(config: Config, project_name: str | None) -> Path: ...
```

### Manager (Phase 2)

```python
# src/memory/manager.py
class MemoryIndexManager:
    def __init__(
        self,
        config: Config,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
    ): ...

    def index_memory(self, file_path: str): ...
    def remove_memory(self, memory_id: str): ...
    def persist(self): ...
    def load(self): ...
    def get_memory_count(self) -> int: ...
```

### Models (Phase 2)

```python
# src/memory/models.py
@dataclass
class MemoryFrontmatter:
    type: str  # "plan" | "journal" | "fact" | "observation" | "reflection"
    status: str  # "active" | "archived"
    tags: list[str]
    created_at: datetime

@dataclass
class MemoryDocument:
    id: str
    content: str
    frontmatter: MemoryFrontmatter
    links: list[str]
    file_path: str
```

### Graph Extensions (Phase 3)

```python
# src/indices/graph.py (existing, modified)
class GraphStore:
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        edge_context: str = "",
    ) -> None: ...

    def get_edges_to(self, target: str) -> list[dict[str, str]]: ...
```

### Link Parser (Phase 3)

```python
# src/memory/link_parser.py
@dataclass
class ExtractedLink:
    target: str
    edge_type: str
    anchor_context: str
    position: int

def extract_links(content: str, context_chars: int = 100) -> list[ExtractedLink]: ...
def infer_edge_type(context: str) -> str: ...
```

### Search (Phase 4)

```python
# src/memory/search.py
class MemorySearchOrchestrator:
    def __init__(
        self,
        vector: VectorIndex,
        keyword: KeywordIndex,
        graph: GraphStore,
        config: Config,
        manager: MemoryIndexManager,
    ): ...

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        filter_tags: list[str] | None = None,
        filter_type: str | None = None,
    ) -> list[MemorySearchResult]: ...

    async def search_linked_memories(
        self,
        query: str,
        target_document: str,
        limit: int = 5,
    ) -> list[LinkedMemoryResult]: ...
```

### Tools (Phase 5)

```python
# src/memory/tools.py
async def create_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
    tags: list[str],
    memory_type: str = "journal",
) -> dict[str, str]: ...

async def append_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
) -> dict[str, str]: ...

async def read_memory(
    ctx: ApplicationContext,
    filename: str,
) -> dict[str, str]: ...

async def update_memory(
    ctx: ApplicationContext,
    filename: str,
    content: str,
) -> dict[str, str]: ...

async def delete_memory(
    ctx: ApplicationContext,
    filename: str,
) -> dict[str, str]: ...

async def search_memories(
    ctx: ApplicationContext,
    query: str,
    limit: int = 5,
    filter_tags: list[str] | None = None,
    filter_type: str | None = None,
) -> list[dict]: ...

async def search_linked_memories(
    ctx: ApplicationContext,
    query: str,
    target_document: str,
) -> list[dict]: ...

async def get_memory_stats(ctx: ApplicationContext) -> dict: ...

async def merge_memories(
    ctx: ApplicationContext,
    source_files: list[str],
    target_file: str,
    summary_content: str,
) -> dict[str, str]: ...
```

---

## 5. Risk Register

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Ghost node pollution in main graph | High | Medium | Use separate `MemoryGraph` instance, not shared `GraphStore` |
| Edge attribute migration breaks existing graphs | High | Low | Version graph JSON schema; add migration in `load()` |
| Recency boost config conflicts with main search | Medium | Low | Use `MemoryConfig.recency_*` fields, not `SearchConfig` |
| Memory path resolution incorrect for user strategy | Medium | Medium | Unit test both strategies with mocked `$XDG_DATA_HOME` |
| YAML frontmatter parsing edge cases | Medium | Medium | Use `pyyaml` safe_load, strict schema validation |
| Large memory files slow indexing | Low | Low | Warn if >50KB; suggest splitting |

---

## 6. Testing Strategy

### Phase 1 Tests

```
tests/unit/test_config.py
├── test_memory_config_defaults
├── test_memory_config_from_toml
├── test_resolve_memory_path_project_strategy
└── test_resolve_memory_path_user_strategy
```

### Phase 2 Tests

```
tests/unit/test_memory_manager.py
├── test_index_memory_creates_chunks
├── test_memory_id_prefix
├── test_frontmatter_extraction
├── test_persist_load_cycle
└── test_remove_memory

tests/unit/test_memory_models.py
├── test_memory_frontmatter_validation
└── test_memory_document_links_parsed
```

### Phase 3 Tests

```
tests/unit/test_memory_link_parser.py
├── test_extract_links_basic
├── test_extract_links_multiple
├── test_anchor_context_boundaries
├── test_infer_edge_type_refactor
├── test_infer_edge_type_default
└── test_no_links_returns_empty

tests/integration/test_memory_ghost_nodes.py
├── test_ghost_node_created_for_link
├── test_edge_has_context_attribute
├── test_get_edges_to_returns_sources
└── test_edge_attributes_persist
```

### Phase 4 Tests

```
tests/unit/test_memory_search.py
├── test_search_memories_basic
├── test_search_memories_with_recency_boost
├── test_search_memories_filter_tags
├── test_search_memories_filter_type
└── test_recency_boost_configurable

tests/integration/test_memory_search_linked.py
├── test_search_linked_memories_finds_linkers
├── test_search_linked_memories_includes_anchor
└── test_search_linked_memories_empty_when_no_links
```

### Phase 5 Tests

```
tests/integration/test_memory_tools.py
├── test_create_memory_success
├── test_create_memory_exists_fails
├── test_append_memory_adds_content
├── test_read_memory_returns_content
├── test_update_memory_replaces_content
├── test_delete_memory_moves_to_trash
├── test_get_memory_stats_counts
└── test_merge_memories_consolidates

tests/e2e/test_memory_mcp.py
├── test_mcp_list_tools_includes_memory
├── test_mcp_create_and_search_memory
└── test_mcp_linked_memory_search
```

---

## 7. Implementation Notes

### Reuse Patterns from Existing Code

1. **IndexManager pattern**: `MemoryIndexManager` mirrors [src/indexing/manager.py](../../src/indexing/manager.py) structure
2. **SearchOrchestrator pattern**: `MemorySearchOrchestrator` reuses fusion from [src/search/fusion.py](../../src/search/fusion.py)
3. **Config loading**: Follow `load_config()` pattern in [src/config.py](../../src/config.py)
4. **MCP registration**: Follow `list_tools()`/`call_tool()` pattern in [src/mcp_server.py](../../src/mcp_server.py)
5. **Test fixtures**: Use `tmp_path`, `shared_embedding_model` from [tests/conftest.py](../../tests/conftest.py)

### Storage Strategy Details

**Project Strategy (`storage_strategy = "project"`)**:
- Path: `{project_root}/.memories/`
- Indices: `{project_root}/.memories/indices/`
- Trash: `{project_root}/.memories/.trash/`

**User Strategy (`storage_strategy = "user"`)**:
- Path: `$XDG_DATA_HOME/mcp-markdown-ragdocs/{project_name}/memories/`
- Indices: `$XDG_DATA_HOME/mcp-markdown-ragdocs/{project_name}/memories/indices/`
- Trash: `$XDG_DATA_HOME/mcp-markdown-ragdocs/{project_name}/memories/.trash/`

### Edge Type Inference Rules

| Context Contains | Edge Type |
|-----------------|-----------|
| "refactor", "rewrite", "restructure" | `refactors` |
| "plan", "todo", "will", "should" | `plans` |
| "bug", "fix", "issue", "error" | `debugs` |
| "note", "remember", "mention" | `mentions` |
| (default) | `related_to` |

---

## 8. Open Questions

1. **Q**: Should memory search results appear in main `query_documents` when relevant?
   **Recommendation**: No. Keep corpora strictly separate. Add explicit `include_memories: bool` param in future version.

2. **Q**: How to handle orphaned ghost nodes when linked document is deleted from main corpus?
   **Recommendation**: Ghost nodes are lightweight references. Keep them; they indicate historical context.

3. **Q**: Should `merge_memories` preserve frontmatter from sources?
   **Recommendation**: No. The `summary_content` param should include new frontmatter. Document this in tool description.
