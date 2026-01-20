# GitHub Copilot Instructions for mcp-markdown-ragdocs

## Project Overview

**mcp-markdown-ragdocs** is a local-first RAG server providing semantic search over Markdown documentation via the Model Context Protocol (MCP). It combines vector search (FAISS), keyword search (Whoosh BM25), and graph traversal (NetworkX) using Reciprocal Rank Fusion.

**Core Features**: Hybrid search, git history search, AI memory bank with time range filtering, zero-config auto-indexing, multi-project support

## Technology Stack

**Language**: Python 3.13+ (modern typing with `list[T]`, `dict[K,V]`, `T | None`)
**Search**: FAISS (vectors), Whoosh (BM25), NetworkX (graph)
**Parsing**: tree-sitter (Markdown AST)
**MCP**: stdio protocol via mcp SDK
**Web**: FastAPI (optional HTTP interface)
**Tools**: uv (package manager), pytest (testing), ruff (linting), pyright + ty (type checking)

## Architecture

**Four Layers**:
1. **Interface**: `src/mcp_server.py` (MCP tools), `src/server.py` (FastAPI)
2. **Core**: `src/context.py` (singleton state), `src/config.py` (TOML config), `src/models.py` (dataclasses)
3. **Indexing**: `src/indexing/manager.py` (coordinator), `src/parsers/` (pluggable), `src/chunking/` (hierarchy-aware)
4. **Search**: `src/search/orchestrator.py` (parallel + RRF fusion), `src/search/pipeline.py` (dedup, MMR)

**Key Patterns**: Protocol-based abstractions, async-first with `asyncio.gather()`, dataclass configs, singleton coordination, lazy initialization

## Project Layout

```
src/
├── cli.py                    # Click commands (run, mcp, query, rebuild-index)
├── mcp_server.py            # MCP tool registration and handlers
├── server.py                # FastAPI HTTP server
├── context.py               # ApplicationContext singleton
├── config.py                # TOML config loading, project detection
├── models.py                # Document, Chunk, ChunkResult dataclasses
├── chunking/                # HeaderChunker (preserves hierarchy)
├── git/                     # Commit indexing, search, watching
├── indexing/                # IndexManager, FileWatcher, reconciliation
├── indices/                 # VectorIndex (FAISS), KeywordIndex (Whoosh), GraphStore (NetworkX)
├── memory/                  # AI memory bank (CRUD, search, graph linking)
├── parsers/                 # MarkdownParser (tree-sitter), PlainTextParser
└── search/                  # Orchestrator (RRF fusion), Pipeline (dedup, MMR)
```

## Coding Philosophy

**Type Discipline**: Modern Python 3.13+ typing (`list[T]`, `T | None`, no `typing.Optional`/`typing.List`) with strict pyright enforcement
**Async Everything**: All I/O is `async def`, use `asyncio.gather()` for parallelism, `asyncio.to_thread()` for blocking calls
**No Silent Failures**: Log with `exc_info=True`, raise specific exceptions (`ValueError`, `RuntimeError`), graceful degradation for optional features
**Real Tests**: Use actual indices with realistic data, no mocks, ephemeral fixtures with `tmp_path`
**Protocol Over ABC**: Structural typing with `Protocol`, not abstract base classes
**Dataclasses**: Immutable configs with `field(default_factory=...)` for mutables

### Type Safety and Validation

**Static typing** with pyright provides compile-time safety. For runtime validation at API boundaries:

- **MCP tools**: MCP SDK validates JSON schemas automatically
- **REST endpoints**: Pydantic models provide runtime validation
- **Config loading**: Manual validation in `__post_init__` methods
- **Internal code**: Trust static type checker (pyright)

```python
from pydantic import BaseModel, Field

# REST API validation (using Pydantic)
class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, ge=1, le=100)

# Config validation (manual in __post_init__)
@dataclass
class MemoryConfig:
    recency_boost_days: int = 7

    def __post_init__(self):
        if self.recency_boost_days < 0:
            raise ValueError("recency_boost_days must be non-negative")
```

**Validation strategy**:
- ✅ Static types everywhere (enforced by pyright)
- ✅ Pydantic for REST API endpoints
- ✅ Manual validation for dataclass configs
- ✅ MCP schema validation handled by SDK
- ❌ No additional runtime type decorators needed

## Naming Conventions

- **Modules**: `snake_case` (`commit_indexer.py`)
- **Classes**: `PascalCase` (`SearchOrchestrator`)
- **Functions**: `snake_case` (`query_documents()`)
- **Private**: Leading underscore (`_search_vector()`)
- **Constants**: `UPPER_SNAKE_CASE` (`DEFAULT_RRF_K`)
- **Protocols**: Suffix `Protocol` (`DocumentParser`)

## Key Anti-Patterns

❌ **Don't block event loop**: No `open()` in async, use `asyncio.to_thread()`
❌ **Don't swallow exceptions**: Always log with `exc_info=True`
❌ **Don't use mocks in tests**: Use real indices with `tmp_path`
❌ **Don't use mutable defaults**: Use `None` with `field(default_factory=...)`
❌ **Don't use `typing.Optional/List/Dict`**: Use `T | None`, `list[T]`, `dict[K,V]`

## Self-Healing Index Pattern

All index types implement corruption detection and automatic recovery:

- **Detection**: Catch `json.JSONDecodeError`, `FileNotFoundError`, `OSError`, `DatabaseError` at operation boundaries
- **Recovery**: Call `_reinitialize_after_corruption()` to reset to clean state
- **Behavior**: Return empty results (graceful degradation), log warning with `exc_info=True`
- **Rebuild**: Reconciliation will repopulate indices from source documents

```python
# Pattern for corruption-safe operations
def search(self, query: str, top_k: int = 10) -> list[dict]:
    try:
        searcher = self._index.searcher()
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Index corruption detected: {e}. Reinitializing.", exc_info=True)
        self._reinitialize_after_corruption()
        return []  # Graceful degradation
    # ... rest of search logic
```

See [docs/specs/19-self-healing-indices.md](../docs/specs/19-self-healing-indices.md) for full specification.

## Critical Files

- **`src/mcp_server.py`**: Add MCP tools (list_tools → call_tool → handler)
- **`src/search/orchestrator.py`**: Modify search/RRF fusion logic
- **`src/indexing/manager.py`**: Change indexing behavior (atomic updates)
- **`src/config.py`**: Config loading, project detection
- **`src/context.py`**: Lifecycle, background tasks, signal handling

## Development

```bash
uv sync                                          # Install dependencies
uv run mcp-markdown-ragdocs mcp                  # Run MCP server
uv run pytest --cov=src --cov-report=html        # Test with coverage
uv run ruff check --fix . && ruff format .       # Lint and format
uv run pyright                                    # Type check (pyright)
uv tool run ty check .                           # Type check (ty alternative)
```

---

## Memory Search Features

**Time Range Filtering** (`search_memories` tool):
- **Absolute timestamps**: `after_timestamp`, `before_timestamp` (Unix timestamps)
- **Relative filtering**: `relative_days` (last N days, overrides absolute)
- **Validation**: `after < before`, `relative_days ≥ 0`
- **Time source**: `created_at` frontmatter field with fallback to file `mtime`
- **Timezone handling**: UTC normalization

**Usage examples**:
```python
# Last 7 days
await search_memories(ctx, query="bug fixes", relative_days=7)

# Absolute range (Jan 2024)
await search_memories(ctx, query="features",
                     after_timestamp=1704067200,
                     before_timestamp=1706745600)

# Combined with tag filtering
await search_memories(ctx, query="auth",
                     relative_days=30,
                     filter_tags=["security"])
```

---

**Additional Context**: See `AGENTS.md` for AI behavioral guidelines, `docs/architecture.md` for system design, `docs/specs/` for ADRs.
