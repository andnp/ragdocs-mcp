# Memory Management

This document describes the Memory Management System for persistent AI memory storage.

## Overview

The Memory Management System provides a separate "Memory Lane" corpus for AI assistants to store and retrieve persistent knowledge across sessions. Memories are stored as Markdown files with YAML frontmatter, indexed using the same hybrid search infrastructure as main documents (Vector + Keyword + Graph).

**Key capabilities:**
- CRUD operations on memory files
- Hybrid search with memory-specific recency boost
- Cross-corpus linking via ghost nodes
- Tag and type-based filtering
- Memory consolidation (merge)

## Configuration

Enable memory management in `config.toml`:

```toml
[memory]
enabled = true
storage_strategy = "project"  # "project" or "user"
recency_boost_days = 7
recency_boost_factor = 1.2
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the Memory Management System |
| `storage_strategy` | string | `"project"` | `"project"`: `.memories/` in project root; `"user"`: `~/.local/share/mcp-markdown-ragdocs/memories/` |
| `recency_boost_days` | int | `7` | Days within which memories receive recency boost |
| `recency_boost_factor` | float | `1.2` | Score multiplier for recent memories (1.2 = 20% boost) |

### Storage Strategies

**Project storage (`"project"`):**
- Memories stored in `.memories/` within project directory
- Isolated per project
- Committed to version control (optional)
- Indices stored in `.memories/indices/`

**User storage (`"user"`):**
- Memories stored in `~/.local/share/mcp-markdown-ragdocs/memories/`
- Shared across all projects
- Persistent across project switches
- Single memory bank for global knowledge

## Memory Format

Memories are Markdown files with YAML frontmatter:

```yaml
---
type: "plan"
status: "active"
tags: ["refactor", "auth"]
created_at: "2025-01-10T10:00:00Z"
---

Memory content in Markdown.

Use [[wikilinks]] to reference documents from the main corpus.
```

### Frontmatter Fields

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `type` | string | No | `"journal"` (default), `"plan"`, `"fact"`, `"observation"`, `"reflection"` |
| `status` | string | No | `"active"` (default), `"archived"` |
| `tags` | list[string] | No | Arbitrary tags for filtering |
| `created_at` | ISO 8601 | No | Auto-generated on creation |

## Tool Reference

### CRUD Operations

#### `create_memory`

Create a new memory file.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Filename including `.md` extension |
| `content` | string | Yes | Memory content (Markdown) |
| `tags` | list[string] | Yes | Tags for categorization |
| `memory_type` | string | No | Memory type (default: `"journal"`) |

**Example:**

```json
{
  "filename": "auth-refactor-plan.md",
  "content": "Plan to refactor [[src/auth.py]] to use JWT tokens.\n\n## Goals\n- Remove session-based auth",
  "tags": ["refactor", "auth"],
  "memory_type": "plan"
}
```

**Response:**

```json
{
  "status": "created",
  "filename": "auth-refactor-plan.md",
  "path": "/project/.memories/auth-refactor-plan.md"
}
```

#### `read_memory`

Read full content of a memory file.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Filename to read |

**Response:**

```json
{
  "filename": "auth-refactor-plan.md",
  "content": "---\ntype: \"plan\"\n...",
  "path": "/project/.memories/auth-refactor-plan.md"
}
```

#### `update_memory`

Replace memory content entirely.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Filename to update |
| `content` | string | Yes | New content (replaces existing) |

**Note:** Full replacement. Include frontmatter in content if preserving metadata.

#### `append_memory`

Append content to existing memory.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Filename to append to |
| `content` | string | Yes | Content to append |

**Behavior:** Adds two newlines before appending content.

#### `delete_memory`

Soft-delete a memory (moves to `.trash/`).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Filename to delete |

**Response:**

```json
{
  "status": "deleted",
  "filename": "old-notes.md",
  "moved_to": "/project/.memories/.trash/old-notes_20250110_143022.md"
}
```

**Note:** Files moved to `.trash/` with timestamp suffix. Not permanently deleted.

### Search Operations

#### `search_memories`

Hybrid search across memory corpus with recency boost.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language query |
| `limit` | int | No | 5 | Maximum results |
| `filter_tags` | list[string] | No | None | Filter by tags (OR logic) |
| `filter_type` | string | No | None | Filter by memory type |

**Example:**

```json
{
  "query": "authentication improvements",
  "filter_tags": ["auth", "security"],
  "filter_type": "plan",
  "limit": 10
}
```

**Response:**

```json
[
  {
    "memory_id": "memory:auth-refactor-plan",
    "score": 0.92,
    "content": "Plan to refactor src/auth.py...",
    "type": "plan",
    "status": "active",
    "tags": ["refactor", "auth"],
    "file_path": "/project/.memories/auth-refactor-plan.md",
    "header_path": "Goals"
  }
]
```

**Recency Boost:**

Memories created within `recency_boost_days` receive score × `recency_boost_factor`:

```python
if (now - created_at).days <= 7:
    score *= 1.2  # 20% boost
```

#### `search_linked_memories`

Find memories that link to a specific document via ghost nodes.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Query to rank results |
| `target_document` | string | Yes | - | Document path (e.g., `"src/auth.py"`) |
| `limit` | int | No | 5 | Maximum results |

**Example:**

```json
{
  "query": "refactor plans",
  "target_document": "src/auth.py",
  "limit": 5
}
```

**Response:**

```json
[
  {
    "memory_id": "memory:auth-refactor-plan",
    "score": 0.88,
    "content": "Plan to refactor [[src/auth.py]] to use JWT...",
    "anchor_context": "...refactor [[src/auth.py]] to use JWT tokens...",
    "edge_type": "mentions",
    "file_path": "/project/.memories/auth-refactor-plan.md"
  }
]
```

**Ghost Node Mechanism:**

When a memory contains `[[src/auth.py]]`:
1. A ghost node `ghost:src/auth.py` is created in the Memory Graph
2. An edge connects the memory to the ghost node
3. `search_linked_memories` traverses edges from `ghost:{target}`
4. `anchor_context` shows ~100 characters surrounding the link

### Maintenance Operations

#### `get_memory_stats`

Get memory bank statistics.

**Parameters:** None

**Response:**

```json
{
  "count": 12,
  "total_size": "45.2KB",
  "tags": {"auth": 5, "refactor": 3, "bug": 2},
  "types": {"plan": 4, "journal": 6, "fact": 2},
  "memory_path": "/project/.memories"
}
```

#### `merge_memories`

Consolidate multiple memories into one.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `source_files` | list[string] | Yes | Filenames to merge |
| `target_file` | string | Yes | New filename for merged memory |
| `summary_content` | string | Yes | Content for merged memory |

**Example:**

```json
{
  "source_files": ["auth-notes-1.md", "auth-notes-2.md", "auth-notes-3.md"],
  "target_file": "auth-consolidated.md",
  "summary_content": "---\ntype: \"journal\"\ntags: [\"auth\", \"consolidated\"]\n---\n\n# Auth Notes Consolidated\n\nKey insights from auth work..."
}
```

**Behavior:**
1. Creates `target_file` with `summary_content`
2. Moves source files to `.trash/` (timestamped)
3. Re-indexes target, removes sources from index

#### `suggest_memory_merges`

Suggest groups of memories that could be merged based on content similarity.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `threshold` | float | No | 0.85 | Similarity threshold (0.0 to 1.0) |
| `limit` | int | No | 5 | Max clusters to return |
| `filter_type` | string | No | `"journal"` | Only cluster memories of this type |

**Example:**

```json
{
  "threshold": 0.85,
  "filter_type": "journal"
}
```

**Response:**

```json
[
  {
    "cluster_id": 0,
    "score": 0.92,
    "reason": "High vector similarity (> 0.85)",
    "memory_count": 3,
    "memories": [
      { "id": "chunk_id_1", "file_path": "/memories/journal-1.md", "preview": "..." },
      { "id": "chunk_id_2", "file_path": "/memories/journal-2.md", "preview": "..." }
    ]
  }
]
```

## Architecture

### Dual-Lane Pattern

Memory Management implements a parallel indexing pipeline:

| Component | Main Corpus | Memory Corpus |
|:--|:--|:--|
| Source | `docs/**/*.md` | `.memories/*.md` |
| Vector Index | `indices/vector/` | `.memories/indices/vector/` |
| Keyword Index | `indices/keyword/` | `.memories/indices/keyword/` |
| Graph | Document nodes | Memory + Ghost nodes |
| Orchestrator | `SearchOrchestrator` | `MemorySearchOrchestrator` |

### Ghost Nodes

Ghost nodes enable cross-corpus linking without indexing the main corpus in the memory graph:

```
memory:auth-plan  ──[mentions]──▶  ghost:src/auth.py
       │
       └──[plans]──▶  ghost:docs/roadmap.md
```

**Edge attributes:**
- `edge_type`: Relationship type (default: `"related_to"`)
- `edge_context`: Surrounding text (~100 chars)

### Index Structure

```
.memories/
├── auth-plan.md
├── project-journal.md
├── .trash/
│   └── old-note_20250110_143022.md
└── indices/
    ├── vector/
    │   ├── docstore.json
    │   └── faiss_index.bin
    ├── keyword/
    │   └── (Whoosh index files)
    └── graph/
        └── graph.json
```

## Usage Patterns

### Session Journal

Maintain a running journal of work sessions:

```json
{
  "filename": "session-2025-01-10.md",
  "content": "## Session Notes\n\n- Investigated auth bug in [[src/auth.py]]\n- Root cause: token expiry not checked\n- Fixed in commit abc123",
  "tags": ["session", "auth", "bugfix"],
  "memory_type": "journal"
}
```

### Project Plans

Store architectural decisions and plans:

```json
{
  "filename": "database-migration-plan.md",
  "content": "## Database Migration Plan\n\nMigrate from SQLite to PostgreSQL.\n\n### Affected Files\n- [[src/db/connection.py]]\n- [[src/db/models.py]]",
  "tags": ["migration", "database", "plan"],
  "memory_type": "plan"
}
```

### Knowledge Facts

Store reusable facts about the codebase:

```json
{
  "filename": "api-rate-limits.md",
  "content": "## API Rate Limits\n\n- Free tier: 100 req/min\n- Pro tier: 1000 req/min\n- Enterprise: unlimited\n\nImplemented in [[src/middleware/rate_limit.py]]",
  "tags": ["api", "limits", "reference"],
  "memory_type": "fact"
}
```

### Retrieving Context

Search for relevant memories before starting work:

```json
{
  "query": "authentication token handling",
  "filter_tags": ["auth"],
  "limit": 5
}
```

Find all memories related to a file:

```json
{
  "query": "recent changes and plans",
  "target_document": "src/auth.py"
}
```

## Troubleshooting

### Memories Not Appearing in Search

1. **Check memory system enabled:**
   ```zsh
   uv run mcp-markdown-ragdocs check-config
   ```
   Verify `[memory] enabled = true`.

2. **Rebuild memory index:**
   Memory index rebuilds automatically on changes. For manual rebuild, delete `.memories/indices/` and restart server.

### Ghost Node Links Not Working

Ensure wikilinks use correct format:
- Correct: `[[src/auth.py]]`
- Incorrect: `[src/auth.py]`, `[[./src/auth.py]]`

### Storage Strategy Change

Changing `storage_strategy` does not migrate existing memories. To migrate:
1. Copy memory files to new location
2. Delete old `.memories/indices/` directory
3. Restart server (triggers reindex)

### Performance Considerations

- Memory search uses same hybrid pipeline as main search (~100-150ms)
- Ghost node traversal adds ~10ms for linked memory search
- Large memory banks (1000+ files) may benefit from tag/type filtering
