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

## Search Scoring & Calibration

Memory search scores undergo a multi-stage pipeline to produce interpretable, well-separated confidence values.

### Scoring Pipeline

Memory search applies transformations in this order:

1. **RRF Fusion**: Reciprocal Rank Fusion combines vector and keyword search rankings → raw scores (typically 0.01-0.05 range)
2. **Calibration**: Sigmoid expansion maps compressed RRF scores to interpretable 0-1 range
3. **Decay**: Exponential decay adjusts scores based on memory age and type
4. **Threshold**: Filter results below `score_threshold` (default: 0.2)

**Calibration vs Normalization:** Memory search uses sigmoid calibration to produce absolute confidence scores. Legacy min-max normalization is deprecated and not used for memory scoring.

### Score Ranges

After calibration, scores represent absolute match quality:

| Score Range | Interpretation | Typical Use Case |
|-------------|----------------|------------------|
| **0.6 - 0.9** | High relevance | Strong semantic match, directly answers query |
| **0.3 - 0.6** | Medium relevance | Related content, partial match |
| **0.1 - 0.3** | Low relevance | Tangentially related, may contain keywords |
| **< 0.2** | Filtered | Below threshold (configurable via `score_threshold`) |

### Calibration Formula

The system uses sigmoid calibration to expand compressed RRF scores:

```python
def calibrate_score(rrf_score: float, threshold: float = 0.035, steepness: float = 150.0) -> float:
    """Sigmoid calibration: maps [0.01, 0.05] → [0, 1] with interpretable separation."""
    return 1.0 / (1.0 + exp(-steepness * (rrf_score - threshold)))
```

**Parameters:**
- `threshold` (default: 0.035): RRF score corresponding to 50% confidence
- `steepness` (default: 150.0): Controls curve sharpness (higher = more separation)

**Before/After Example:**

```python
# Without calibration (compressed)
raw_rrf_score = 0.039
score_separation = 0.039 / 0.020  # 1.95x between best and good match

# With calibration (expanded)
calibrated_score = 0.646  # sigmoid(0.039)
separation = 0.646 / 0.095  # 6.77x separation (3.5x improvement)
```

**Impact:**
- Score compression: **Fixed** (1.95x → 6.77x separation)
- Range: 0.01-0.03 → **0-1** (interpretable)
- Filtering: Effective threshold-based filtering at `score_threshold`

### Configuration

Tune calibration and filtering in `config.toml`:

```toml
[memory]
score_threshold = 0.2  # Minimum score after calibration (default: 0.2)

[search]  # Affects memory search calibration
score_calibration_threshold = 0.035  # RRF score → 50% confidence
score_calibration_steepness = 150.0  # Curve sharpness
```

**Tuning Guidelines:**

| Scenario | Adjust | Value | Effect |
|----------|--------|-------|--------|
| Too few results | Lower `score_threshold` | 0.15 or 0.1 | Allow more low-confidence matches |
| Too many irrelevant | Raise `score_threshold` | 0.25 or 0.3 | Stricter filtering |
| Scores too strict | Lower `score_calibration_threshold` | 0.03 | Shift curve left (higher scores) |
| Scores too generous | Raise `score_calibration_threshold` | 0.04 | Shift curve right (lower scores) |
| Need more separation | Raise `score_calibration_steepness` | 200.0 | Sharper distinctions near threshold |
| Smoother gradation | Lower `score_calibration_steepness` | 100.0 | Gentler transitions |

## Configuration

Enable memory management in `config.toml`:

```toml
[memory]
enabled = true
storage_strategy = "project"  # "project" or "user"
score_threshold = 0.1

# Per-type recency boost configurations
[memory.recency_journal]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95

[memory.recency_plan]
boost_window_days = 21
max_boost_amount = 0.15
boost_decay_rate = 0.93

[memory.recency_fact]
boost_window_days = 7
max_boost_amount = 0.1
boost_decay_rate = 0.98

[memory.recency_observation]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.92

[memory.recency_reflection]
boost_window_days = 30
max_boost_amount = 0.15
boost_decay_rate = 0.98
```

### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the Memory Management System |
| `storage_strategy` | string | `"project"` | `"project"`: `.memories/` in project root; `"user"`: `~/.local/share/mcp-markdown-ragdocs/memories/` |

### Recency Boost System

Memory search uses an **exponential additive recency boost** that rewards recent memories without penalizing older ones. Recent memories receive an exponentially decaying bonus *added* to their base score.

**Boost Formula:**

```
if age_days ≤ boost_window_days:
    boost_factor = boost_decay_rate^age_days
    bonus = boost_factor × max_boost_amount
    final_score = min(1.0, base_score + bonus)
else:
    final_score = base_score  # No penalty for old memories
```

- `base_score`: Hybrid search score (semantic + keyword fusion)
- `boost_decay_rate`: Exponential decay rate for boost amount (0.0-1.0)
- `age_days`: Days since memory creation
- `max_boost_amount`: Maximum bonus at age 0 (range: 0.0-0.5, typically 0.1-0.2)
- `boost_window_days`: Days within which to apply boost (7-30 days)

**Key Insight:** Old memories beyond the boost window retain their full base score—they are not penalized for age. This prevents the double-filtering effect that occurred with multiplicative decay.

**Threshold Filtering:**

Memories scoring below `score_threshold` (default: 0.1) after boost are filtered. The threshold can be adjusted based on precision/recall requirements.

### Per-Type Boost Configurations

| Memory Type | Window | Max Boost | Decay Rate | Rationale |
|------------|--------|-----------|------------|----------|
| `journal` | 14 days | 0.2 | 0.95 | Recent journals highly relevant |
| `plan` | 21 days | 0.15 | 0.93 | Plans stay relevant longer |
| `fact` | 7 days | 0.1 | 0.98 | Facts are mostly timeless |
| `observation` | 14 days | 0.2 | 0.92 | Observations moderately temporal |
| `reflection` | 30 days | 0.15 | 0.98 | Reflections age well |

**Example:** 7-day-old journal memory with base score 0.4:
```
boost_factor = 0.95^7 ≈ 0.698
bonus = 0.698 × 0.2 = 0.140
final_score = 0.4 + 0.140 = 0.540
```

### Configuration Examples

**Default (Balanced):**

```toml
[memory]
enabled = true
storage_strategy = "project"
score_threshold = 0.1

[memory.recency_journal]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95
```

**Aggressive Recency Boost (Short-Term Focus):**

```toml
[memory.recency_journal]
boost_window_days = 7  # Shorter window
max_boost_amount = 0.3  # Higher boost
boost_decay_rate = 0.90  # Faster decay

[memory.recency_plan]
boost_window_days = 10
max_boost_amount = 0.25
boost_decay_rate = 0.88
```

**Conservative Boost (Long-Term Focus):**

```toml
[memory.recency_fact]
boost_window_days = 30  # Longer window
max_boost_amount = 0.05  # Minimal boost
boost_decay_rate = 0.99  # Very slow decay

[memory.recency_reflection]
boost_window_days = 60
max_boost_amount = 0.1
boost_decay_rate = 0.99
```

### Boost Curves Over Time

**Journal (decay_rate=0.90, floor=0.1):**

| Days | Multiplier | Example Score (0.020) |
|------|------------|----------------------|
| 0 | 1.00 | 0.020 |
| 7 | 0.48 | 0.0096 |
| 14 | 0.23 | 0.0046 |
| 30 | 0.10 (floor) | 0.002 |
| 60+ | 0.10 (floor) | 0.002 |

**Fact (decay_rate=0.98, floor=0.2):**

| Days | Multiplier | Example Score (0.020) |
|------|------------|----------------------|
| 0 | 1.00 | 0.020 |
| 35 | 0.50 | 0.010 |
| 70 | 0.25 | 0.005 |
| 150 | 0.20 (floor) | 0.004 |
| 300+ | 0.20 (floor) | 0.004 |

### When to Tune Decay Rates

**Increase decay rate (slower decay):**
- Knowledge base with evergreen content
- Research notes requiring long-term recall
- Reference documentation

**Decrease decay rate (faster decay):**
- Session journals tracking daily work
- Short-term task planning
- Volatile project state

**Adjust floor multiplier:**
- **Lower floor (0.05-0.1)**: Aggressive filtering, surface only relevant recent memories
- **Higher floor (0.2-0.3)**: Preserve older memories, ensure discoverability

### Migration from Deprecated Config

The old `recency_boost_days` and `recency_boost_factor` fields are **deprecated** and will be ignored if present.

**Old config (deprecated):**

```toml
[memory]
recency_boost_days = 7
recency_boost_factor = 1.2
```

**New config (equivalent decay):**

```toml
[memory.decay_journal]
decay_rate = 0.90  # Approximates 7-day window
floor_multiplier = 0.1
```

**Behavioral differences:**

| Aspect | Old Boost | New Decay |
|--------|-----------|----------|
| Scoring | Binary (1.0x or 1.2x) | Continuous (exponential) |
| Threshold | Fixed time window | Age-aware gradual reduction |
| Type-awareness | None | Per-type rates |
| Long-term | Cliff at window edge | Smooth floor approach |

**Backward compatibility:** The system automatically detects and uses decay configs. If deprecated fields are present, warnings are logged but decay system still applies.

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

### Metadata Fields

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
| `filter_type` | string | No | None | Filter by memory type |
| `after_timestamp` | int | No | None | Unix timestamp: only return memories created/modified after this time |
| `before_timestamp` | int | No | None | Unix timestamp: only return memories created/modified before this time |
| `relative_days` | int | No | None | Only return memories from last N days (overrides absolute timestamps) |

**Time Filtering Behavior:**

- **Precedence:** `relative_days` overrides `after_timestamp` and `before_timestamp`
- **Validation:** `after_timestamp` must be less than `before_timestamp`; `relative_days` must be ≥ 0
- **Time Source:** Uses `created_at` from frontmatter, falls back to file modification time if unavailable
- **Timezone:** All timestamps normalized to UTC

**Examples:**

```json
// Basic search
{
  "query": "authentication improvements",
  "filter_type": "plan",
  "limit": 10
}

// Last 7 days
{
  "query": "bug fixes",
  "relative_days": 7
}

// Absolute time range (Jan 1-31, 2024)
{
  "query": "new features",
  "after_timestamp": 1704067200,
  "before_timestamp": 1706745600
}

// Combined filters
{
  "query": "auth improvements",
  "relative_days": 30,
  "filter_type": "plan"
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

**Score Interpretation:**

Memory scores after calibration and recency boost represent absolute match quality:

- **0.7-1.0**: Highly relevant (recent memories may be boosted to 1.0)
- **0.5-0.7**: Moderately relevant, related content
- **0.2-0.5**: Low relevance, tangential matches
- **< 0.1**: Filtered by default threshold

**Boost Scoring:**

Recency boost applies *after* calibration, adding bonus to recent memories:

```python
def apply_boost(base_score: float, created_at: datetime, config: MemoryRecencyConfig) -> float:
    """Apply exponential additive boost to base score."""
    age_days = (datetime.now() - created_at).days
    if age_days <= config.boost_window_days:
        boost_factor = config.boost_decay_rate ** age_days
        bonus = boost_factor * config.max_boost_amount
        return min(1.0, base_score + bonus)
    else:
        return base_score  # Old memories: no penalty
```

**Example:** 7-day-old journal (base_score=0.4, max_boost=0.2, boost_rate=0.95):

```
boost_factor = 0.95^7 ≈ 0.698
bonus = 0.698 × 0.2 = 0.140
final_score = 0.4 + 0.140 = 0.540  # Above threshold (0.1)
```

Memories scoring below `score_threshold` (default: 0.1) after boost are filtered.

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

### Graph Relationship Operations

Memory-to-memory relationships are created via wikilink syntax with context keywords. The system detects relationship types based on surrounding text.

#### `get_memory_relationships`

**Primary tool for querying memory relationships.** Get version history (SUPERSEDES), dependencies (DEPENDS_ON), or contradictions (CONTRADICTS) for a memory.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | string | Yes | Memory filename to query |
| `relationship_type` | string | No | `"supersedes"`, `"depends_on"`, `"contradicts"`, or omit for all |

**Relationship Types:**

| Type | Edge Type | Meaning | Context Keywords |
|------|-----------|---------|------------------|
| `supersedes` | `SUPERSEDES` | Version/revision chain | "supersedes", "replaces", "updates" |
| `depends_on` | `DEPENDS_ON` | Prerequisites/dependencies | "depends on", "requires", "needs" |
| `contradicts` | `CONTRADICTS` | Conflicting information | "contradicts", "disagrees with", "challenges" |

**Creating Relationships:**

Use `[[memory:filename]]` wikilinks with context keywords:

```markdown
This plan [[memory:auth-v1]] supersedes the previous approach.

The implementation [[memory:jwt-impl]] depends on [[memory:auth-refactor-plan]].

Our new strategy [[memory:stateless-auth]] contradicts [[memory:session-based-auth]].
```

**Examples:**

```json
// Get all relationships
{
  "filename": "auth-refactor-v2.md"
}

// Get only version history
{
  "filename": "auth-refactor-v2.md",
  "relationship_type": "supersedes"
}

// Get only dependencies
{
  "filename": "jwt-implementation.md",
  "relationship_type": "depends_on"
}

// Get only contradictions
{
  "filename": "new-approach.md",
  "relationship_type": "contradicts"
}
```

**Response (all relationships):**

```json
{
  "supersedes": {
    "version_chain": [
      {
        "memory_id": "memory:auth-refactor-v2",
        "file_path": "/project/.memories/auth-refactor-v2.md"
      },
      {
        "memory_id": "memory:auth-refactor-v1",
        "file_path": "/project/.memories/auth-refactor-v1.md"
      }
    ],
    "count": 2
  },
  "depends_on": [
    {
      "memory_id": "memory:jwt-library-research",
      "file_path": "/project/.memories/jwt-library-research.md",
      "context": "...implementation depends on [[memory:jwt-library-research]] for library selection..."
    }
  ],
  "contradicts": [
    {
      "memory_id": "memory:session-based-auth",
      "file_path": "/project/.memories/session-based-auth.md",
      "context": "...this approach contradicts [[memory:session-based-auth]] which relied on cookies..."
    }
  ]
}
```

**Response (single relationship type):**

```json
{
  "supersedes": {
    "version_chain": [
      {
        "memory_id": "memory:auth-refactor-v2",
        "file_path": "/project/.memories/auth-refactor-v2.md"
      },
      {
        "memory_id": "memory:auth-refactor-v1",
        "file_path": "/project/.memories/auth-refactor-v1.md"
      }
    ],
    "count": 2
  }
}
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

### Score Interpretation

**Q: What do memory search scores mean?**

Scores represent calibrated confidence after RRF fusion, exponential decay, and threshold filtering:

- **0.6-0.9**: High confidence, strong semantic match
- **0.3-0.6**: Medium confidence, related content
- **0.1-0.3**: Low confidence, tangentially related
- **< 0.2**: Filtered (below default `score_threshold`)

Scores above 0.6 indicate the memory directly addresses the query. Scores 0.3-0.6 suggest partial relevance. Adjust `score_threshold` to tune precision/recall.

**Q: Why are my memory scores different from document search?**

Memory search applies exponential decay based on age, while document search uses recency tiers:

- **Memory decay**: Continuous score reduction (`score × decay_rate^days_old`)
- **Document recency**: Fixed multipliers (1.2x for 7 days, 1.1x for 30 days)

Recent memories decay less, older memories decay more. Decay curves are type-specific (facts decay slower than journals).

**Q: How do I tune the score threshold for my use case?**

Adjust `score_threshold` in `config.toml` based on result quality:

```toml
[memory]
score_threshold = 0.1  # Default: balanced precision/recall
```

**Tuning decision tree:**

1. **Getting zero or very few results?**
   - Lower threshold: `score_threshold = 0.05` or `0.08`
   - Trade-off: More results but lower average relevance

2. **Getting too many irrelevant results?**
   - Raise threshold: `score_threshold = 0.15` or `0.2`
   - Trade-off: Fewer results but higher precision

3. **Scores seem wrong (too low or too high)?**
   - Check boost configuration for memory type
   - Verify memory age (recent memories get boost)
   - Adjust `score_calibration_threshold` in `[search]` section (affects calibration curve)

**Q: When should I adjust score_threshold up or down?**

**Lower threshold (0.05 or 0.08) when:**
- Building a comprehensive knowledge base (prioritize recall)
- Exploring broad topics with diverse memories
- Most memories are older (beyond boost window)

**Raise threshold (0.15 or 0.2) when:**
- Need high precision for critical queries
- Memory bank has lots of tangentially-related content
- Want only the most relevant results (prioritize precision)

**Keep default (0.1) when:**
- Balanced use case (general knowledge retrieval)
- Mix of recent and older memories
- Standard documentation/journal workflow

**Q: How does recency boost affect scores?**

**Recent memories (within boost window):**
- Get exponential bonus added to base score
- Maximum boost at age 0 (e.g., +0.2 for journals)
- Boost decays exponentially (e.g., 0.95^days)

**Old memories (beyond boost window):**
- Retain full base score (no penalty)
- Not affected by boost system
- Scored purely on content relevance

**Common Scenarios:**

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| "Too few results" | Threshold too high | Lower `score_threshold` to 0.08; check if most memories are old (no boost) |
| "Too many irrelevant results" | Threshold too low | Raise `score_threshold` to 0.15 or 0.2 |
| "Recent memories dominate" | Boost too aggressive | Lower `max_boost_amount` (e.g., 0.15 → 0.1) |
| "Old memories never appear" | Threshold calibrated for boosted scores | Lower `score_threshold` to account for unboosted scores |
| "Good matches scored low" | Calibration threshold too high | Lower `score_calibration_threshold` to 0.03 in `[search]` |
| "All scores bunched together" | Steepness too low | Raise `score_calibration_steepness` to 200.0 in `[search]` |

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
