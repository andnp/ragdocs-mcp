# 10. Multi-Project Support

## Executive Summary

**Purpose:** Enable users to manage isolated document indices for multiple projects on a single machine, while maintaining the simplicity of single-project workflows.

**Scope:** Add global configuration registry (`~/.config/mcp-markdown-ragdocs/config.toml`) with project definitions, automatic project detection via directory matching, and isolated database storage per project. Project-local configs retain precedence. Backward compatible with existing single-project deployments.

**Decision:** Use project "keyname" as database identifier stored in `~/.local/share/mcp-markdown-ragdocs/{keyname}/`, with CWD-based project detection at startup. No runtime project switching; each server instance serves one project.

---

## 1. Goals & Non-Goals

### Goals

1. **Multi-Project Isolation:** Each project maintains separate vector, keyword, and graph indices with zero cross-contamination.
2. **Global Registry:** Centralized project mapping in `~/.config/mcp-markdown-ragdocs/config.toml` with `[[projects]]` array.
3. **Automatic Detection:** Server detects active project by matching CWD against registered project paths at startup.
4. **Config Precedence:** Project-local `.mcp-markdown-ragdocs/config.toml` overrides global config settings (already implemented).
5. **Backward Compatibility:** Existing single-project setups continue working without global config.
6. **XDG Compliance:** Follow XDG Base Directory specification for data storage (`$XDG_DATA_HOME` or `~/.local/share`).
7. **Manual Override:** Support `--project` flag to override automatic detection for edge cases.

### Non-Goals

1. **Runtime Project Switching:** Server instance serves single project for its lifetime; no hot-swapping between projects.
2. **Project Creation CLI:** No `init-project` command; users manually edit global config (but `--project` flag supports manual override).
3. **Shared Indices:** No cross-project search or index merging.
4. **Project Templates:** No project scaffolding or boilerplate generation.
5. **Migration Tooling:** No automatic migration from old `.index_data/` to new structure (users rebuild indices).

---

## 2. Current State Analysis

### 2.1. Configuration Loading

**File:** [src/config.py](../src/config.py) (lines 60-98)

**Current Behavior:**
1. `_find_project_config()`: Walks up directory tree from CWD looking for `.mcp-markdown-ragdocs/config.toml`
2. Falls back to `~/.config/mcp-markdown-ragdocs/config.toml`
3. Returns first found config or defaults if none exists

**Limitations:**
- Global config treated identically to project-local config (no special handling)
- No concept of "projects" or keynames
- Single fallback path prevents multi-project registry

### 2.2. Index Storage

**Files:**
- [src/indexing/manager.py](../src/indexing/manager.py) (lines 84-99)
- [src/indices/vector.py](../src/indices/vector.py) (lines 191-211)
- [src/indices/keyword.py](../src/indices/keyword.py) (lines 115-141)
- [src/indices/graph.py](../src/indices/graph.py) (lines 45-61)

**Current Behavior:**
- Index path resolved from `config.indexing.index_path` (default: `.index_data/`)
- Each index type persists to subdirectory:
  - Vector: `{index_path}/vector/`
  - Keyword: `{index_path}/keyword/`
  - Graph: `{index_path}/graph/`
- Manifest stored at `{index_path}/index.manifest.json`

**Storage Layout:**
```
{index_path}/
├── index.manifest.json
├── vector/
│   ├── docstore.json
│   ├── index_store.json
│   ├── faiss_index.bin
│   ├── doc_id_mapping.json
│   └── chunk_id_mapping.json
├── keyword/
│   └── MAIN_*.{toc,seg,pos}
└── graph/
    └── graph.json
```

**Limitations:**
- Path controlled by config, typically project-local (`.index_data/`)
- No global data directory convention
- Manual config changes required to isolate projects

### 2.3. Server Initialization

**Files:**
- [src/mcp_server.py](../src/mcp_server.py) (lines 102-159)
- [src/server.py](../src/server.py) (lines 47-95)

**Current Flow:**
1. Load config via `load_config()`
2. Initialize indices (in-memory, empty)
3. Create `IndexManager` with config's `index_path`
4. Check manifest for rebuild
5. Load or build indices
6. Start file watcher on `documents_path`

**Limitations:**
- No project detection logic
- Index path directly from config (no transformation)
- Single `documents_path` per server instance

### 2.4. Existing Config Structure

**File:** [specs/05-configuration.md](../specs/05-configuration.md)

**Current Schema:**
```toml
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "."
index_path = ".index_data/"
recursive = true

[parsers]
"**/*.md" = "MarkdownParser"

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
rrf_k_constant = 60

[llm]
embedding_model = "local"

[chunking]
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 1500
overlap_chars = 100
```

---

## 3. Proposed Solution

### 3.1. Architecture Overview

```mermaid
graph TB
    subgraph "Config Resolution"
        CWD[Current Working Directory]
        LOCAL[.mcp-markdown-ragdocs/config.toml]
        GLOBAL[~/.config/mcp-markdown-ragdocs/config.toml]
        CWD -->|walk up tree| LOCAL
        LOCAL -.->|not found| GLOBAL
    end

    subgraph "Project Detection"
        GLOBAL -->|load projects array| DETECT[Project Detector]
        CWD -->|current path| DETECT
        DETECT -->|match subdirectory| KEYNAME[Project Keyname]
        DETECT -.->|no match| FALLBACK[Default Project]
    end

    subgraph "Index Storage"
        KEYNAME --> DATADIR[~/.local/share/mcp-markdown-ragdocs/{keyname}/]
        DATADIR --> VECTOR[vector/]
        DATADIR --> KEYWORD[keyword/]
        DATADIR --> GRAPH[graph/]
        DATADIR --> MANIFEST[index.manifest.json]
    end

    subgraph "Config Precedence"
        LOCAL -->|has index_path| OVERRIDE[Use project-local path]
        GLOBAL -->|no local config| DATADIR
    end
```

### 3.2. Global Config Format

**Location:** `~/.config/mcp-markdown-ragdocs/config.toml`

**Schema:**
```toml
# Global defaults (optional)
[indexing]
# These apply to all projects unless overridden locally
recursive = true

[search]
semantic_weight = 1.0
keyword_weight = 1.0

# Project registry (required for multi-project)
[[projects]]
name = "monorepo"
path = "/home/andy/Projects/rlcore/monorepo"

[[projects]]
name = "mcp-markdown-ragdocs"
path = "/home/andy/Projects/personal/mcp-markdown-ragdocs"

[[projects]]
name = "notes"
path = "/home/andy/Documents/notes"
```

**Constraints:**
- `name`: Alphanumeric + hyphens/underscores only (used as directory name)
- `path`: Absolute path required (expanded with `expanduser()`, checked for existence)
- `name` must be unique across projects
- `path` must be unique (no overlapping directories)

### 3.3. Project Detection Algorithm

**Pseudocode:**
```python
def detect_project(
    cwd: Path,
    projects: list[ProjectConfig],
    project_override: str | None = None
) -> str | None:
    """
    Match CWD against registered projects, with optional manual override.

    Args:
        cwd: Current working directory
        projects: Registered projects from global config
        project_override: Optional project name or path from --project flag

    Returns:
        Project keyname if match found, None if no match.
    """
    # Priority 1: Manual override from --project flag
    if project_override:
        for project in projects:
            # Match by name
            if project.name == project_override:
                return project.name
            # Match by path (if override is absolute path)
            if Path(project_override).is_absolute():
                if Path(project.path).resolve() == Path(project_override).resolve():
                    return project.name
        # Override not found - log warning but continue with CWD detection
        logger.warning(f"Project override '{project_override}' not found")

- **Manual override not found:** Logs warning and falls back to CWD detection
- **Manual override with VS Code:** Use `--project` flag in MCP settings to ensure correct project
    # Priority 2: Automatic CWD-based detection
    cwd_resolved = cwd.resolve()

    # Sort by path depth (deepest first) for specificity
    projects_sorted = sorted(projects, key=lambda p: len(Path(p.path).parts), reverse=True)

    for project in projects_sorted:
        project_path = Path(project.path).resolve()

        # Check if CWD is subdirectory of project path
        try:
            cwd_resolved.relative_to(project_path)
            return project.name
        except ValueError:
            continue

    return None  # No match found
```

**Edge Cases:**
- **Nested projects:** Deepest match wins (e.g., `/home/user/monorepo/subproject/` matches `subproject` before `monorepo`)
- **Symlinks:** Resolved before comparison (both CWD and project paths)
- **No match:** Server uses "default" keyname or project-local `.index_data/`

### 3.4. Index Path Resolution

**Decision Logic:**
```python
def resolve_index_path(config: Config, detected_project: str | None) -> Path:
    """
    Determine index storage location with precedence rules.

    Precedence:
        1. Project-local config's index_path (if specified)
        2. Global data directory for detected project
        3. Project-local .index_data/ (backward compat)
    """
    # Priority 1: Explicit local override
    if config.indexing.index_path != DEFAULT_INDEX_PATH:
        return Path(config.indexing.index_path)

    # Priority 2: Detected project → global storage
    if detected_project:
        data_home = os.getenv("XDG_DATA_HOME", Path.home() / ".local/share")
        return Path(data_home) / "mcp-markdown-ragdocs" / detected_project

    # Priority 3: Fallback to local directory
    return Path(".index_data/")
```

**Storage Structure:**
```
~/.local/share/mcp-markdown-ragdocs/
├── monorepo/
│   ├── index.manifest.json
│   ├── vector/
│   ├── keyword/
│   └── graph/
├── mcp-markdown-ragdocs/
│   ├── index.manifest.json
│   ├── vector/
│   ├── keyword/
│   └── graph/
└── notes/
    ├── index.manifest.json
    ├── vector/
    ├── keyword/
    └── graph/
```

### 3.5. Configuration Precedence

**Behavior:**
- **Project Detection:** Uses global config's `[[projects]]` array only
- **Settings:** Project-local config overrides global config for all other fields
- **Index Path:** Special handling (see 3.4 above)

**Example Scenario:**

**Global Config** (`~/.config/mcp-markdown-ragdocs/config.toml`):
```toml
[[projects]]
name = "monorepo"
path = "/home/andy/monorepo"

[indexing]
recursive = true

[search]
semantic_weight = 2.0
```

**Project-Local Config** (`/home/andy/monorepo/.mcp-markdown-ragdocs/config.toml`):
```toml
[indexing]
documents_path = "./docs"
# recursive inherited from global (true)

[search]
semantic_weight = 1.5  # overrides global
```

**Resulting Config** (when CWD = `/home/andy/monorepo/packages/app`):
```toml
# Detected project: "monorepo"
# Index path: ~/.local/share/mcp-markdown-ragdocs/monorepo/

[indexing]
documents_path = "/home/andy/monorepo/docs"  # from local
index_path = "/home/andy/.local/share/mcp-markdown-ragdocs/monorepo"  # computed
recursive = true  # from global

[search]
semantic_weight = 1.5  # from local
keyword_weight = 1.0   # default
```

---

## 4. Decision Matrix

### 4.1. Index Storage Location

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **A. XDG Data Home** (`~/.local/share/...`) | Standard Unix convention, cleanly separates data from config, works across projects | Requires path translation, unfamiliar to some users | ✅ **SELECTED** |
| B. Alongside global config (`~/.config/...`) | Single directory for all app files | Mixes config and data (anti-pattern) | ❌ |
| C. Project-local only (`.index_data/`) | Simple, visible, no global state | Pollutes project directories, fails for read-only projects | ❌ |
| D. Central database (`~/.mcp-indices.db`) | Single file, easy backup | Complicates architecture, requires DB dependency | ❌ |

**Rationale:** XDG compliance is industry standard. Separation of config (`~/.config`) from data (`~/.local/share`) aligns with Unix philosophy and user expectations.

### 4.2. Project Detection Strategy

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **A. CWD substring matching** | Simple, automatic, no user action | Ambiguous for nested projects | ✅ **SELECTED** (with depth sorting) |
| B. Explicit `--project` flag | Unambiguous, explicit control | Requires manual specification every time | ❌ |
| C. Environment variable (`MCP_PROJECT`) | Works across CLI and server | Non-discoverable, easy to forget | ❌ |
| D. Per-directory `.mcp-project` file | Git-trackable, explicit | Pollutes repos, requires setup | ❌ |

**Rationale:** Automatic detection minimizes friction. Depth-first matching (deepest path wins) resolves nested projects correctly.

### 4.3. Backward Compatibility

| Scenario | Current Behavior | New Behavior | Compatible? |
|----------|------------------|--------------|-------------|
| **No global config** | Uses `.index_data/` | Uses `.index_data/` | ✅ Yes |
| **Local config with `index_path`** | Uses specified path | Uses specified path (overrides global) | ✅ Yes |
| **Global config, no project match** | Uses global settings | Uses global settings + `.index_data/` | ✅ Yes |
| **Global config, project matched** | N/A (new feature) | Uses global data dir | ⚠️ Rebuilds index (acceptable) |

**Migration Path:**
- Existing users without global config: **Zero impact**
- Users adding global config: Must rebuild indices (one-time, documented)
- Projects with explicit `index_path`: **Unchanged**

---

## 5. API Contract

### 5.1. Configuration Schema

#### Global Config (`~/.config/mcp-markdown-ragdocs/config.toml`)

```toml
# Project Registry
[[projects]]
name = "project_name"     # Required: alphanumeric, hyphens, underscores
path = "/absolute/path"   # Required: absolute path to project root

# Optional: Global defaults (inherited by projects)
[indexing]
recursive = true

[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
rrf_k_constant = 60

[llm]
embedding_model = "local"

[chunking]
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 1500
overlap_chars = 100
```

**Validation Rules:**
- `[[projects]]`: Optional (empty array = no multi-project)
- `projects.name`: Must match `^[a-zA-Z0-9_-]+$`
- `projects.path`: Must exist on filesystem (warning if not)
- Duplicate `name` or `path`: Configuration error (server refuses to start)

#### Project-Local Config (`.mcp-markdown-ragdocs/config.toml`)

**No schema changes.** All existing fields supported. Takes precedence over global config for non-project-registry fields.

### 5.2. Updated Config Dataclasses

**New Dataclass:**
```python
@dataclass
class ProjectConfig:
    name: str
    path: str
```

**Modified Config Dataclass:**
```python
@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    parsers: dict[str, str] = field(default_factory=lambda: {...})
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # New field (only present if loaded from global config)
    projects: list[ProjectConfig] = field(default_factory=list)
```

### 5.3. Behavioral Invariants

1. **Single Active Project:** Each server instance serves exactly one project (determined at startup).
2. **Immutable Project Selection:** Project keyname fixed for server lifetime (no runtime switching).
3. **Index Isolation:** No shared data between projects; full rebuild required when switching projec
6. **Manual Override Priority:** `--project` flag takes precedence over CWD-based detection.ts.
4. **Config Precedence:** Local `index_path` overrides global data directory (backward compat).
5. **Graceful Degradation:** Missing global config or no project match → fallback to `.index_data/`.

---

## 6. Architecture Decision Records

### ADR-1: CWD-Based Project Detection

**Status:** Accepted

**Context:** Multi-project support requires determining which project the user intends to work with. The detection mechanism must work automatically without requiring user action on every invocation.

**Decision:** Detect active project by matching current working directory (CWD) against registered project paths at server startup. Deepest match wins for nested projects.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **CWD Matching (selected)** | Automatic, no user action, works with nested projects | Ambiguous if CWD outside all projects |
| Workspace File Detection | Explicit, IDE-native (`.code-workspace`) | IDE-specific, requires file creation, breaks CLI usage |
| Explicit Config Flag | Unambiguous, full user control | Requires `--project` on every invocation |
| Environment Variable | Works across CLI and server | Non-discoverable, easy to forget, stale values |
| Per-Directory `.mcp-project` File | Git-trackable, explicit per-directory | Pollutes repos, requires setup per directory |

**Rationale:** CWD detection minimizes friction for the common case (user runs commands from within a project). The depth-first matching algorithm handles monorepos with nested subprojects. The `--project` flag provides escape hatch for edge cases.

**Implementation:** [src/config.py](../src/config.py) `detect_project()` function.

### ADR-2: XDG Data Home for Index Storage

**Status:** Accepted

**Context:** Multi-project indices require isolated storage. Location must be predictable and follow platform conventions.

**Decision:** Store indices in `$XDG_DATA_HOME/mcp-markdown-ragdocs/{project_name}/` (defaults to `~/.local/share/`).

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| **XDG Data Home (selected)** | Unix standard, separates data from config | Requires path translation |
| Project-local `.index_data/` | Simple, visible | Pollutes project directories |
| Alongside global config | Single directory | Mixes config and data (anti-pattern) |
| Central SQLite database | Single file backup | Requires DB dependency, complicates architecture |

**Rationale:** XDG compliance is industry standard. Separation of config (`~/.config`) from data (`~/.local/share`) aligns with Unix philosophy.

**Implementation:** [src/config.py](../src/config.py) `resolve_index_path()` function.

---

## 7. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Users forget to rebuild indices** | High | Medium | Log prominent warning on first startup with global config |
| **Path matching ambiguity** (nested projects) | Medium | High | Implement depth-first sorting (deepest match wins) + validation |
| **XDG_DATA_HOME not set** (non-standard systems) | Low | Low | Fallback to `~/.local/share` (de facto standard) |
| **Index storage fills disk** | Low | Medium | Document cleanup procedures, consider future LRU eviction |
| **Symlink resolution issues** | Low | Medium | Use `Path.resolve()` consistently; document limitations |
| **Global config syntax errors** | Medium | High | Add `check-config` CLI command with validation |
| **Race condition** (multiple servers, same project) | Low | High | Document as unsupported (user responsibility); consider lockfile future |

---

## 8. Open Questions

### 7.1. Default Project Naming

**Question:** When no project matches CWD and no global config exists, what keyname should be used?

**Options:**
- A. `"default"` (hardcoded fallback)
- B. Hash of CWD path (unique but opaque)
- C. Continue using `.index_data/` (no global storage)

**Recommendation:** **Option C** for backward compatibility. Global storage only used when project explicitly registered.

**Status:** ✅ **RESOLVED** (use `.index_data/` fallback)

### 7.2. Config Validation Strictness

**Question:** Should duplicate project names/paths be an error or warning?

**Options:**
- A. Hard error (refuse to start server)
- B. Warning (use first match, log warning)

**Recommendation:** **Option A** (fail fast). Configuration errors should be explicit and unambiguous.

**Status:** ✅ **RESOLVED** (fail fast)

### 7.3. Migration Assistance

**Question:** Should we provide a migration command to move existing `.index_data/` to global storage?

**Options:**
- A. Yes, add `migrate-project` CLI command
- B. No, users rebuild indices (simpler)

**Recommendation:** **Option B** initially. Rebuilding is quick (<1min for typical projects). Can add migration later if users request.

**Status:** ⚠️ **DEFERRED** (post-v1.0)

---

## 9. Implementation Phases

See [docs/working/multi-project-implementation-plan.md](../docs/working/multi-project-implementation-plan.md) for detailed handoff.

**Phase Summary:**
1. **Config Schema** (1 file): Add `ProjectConfig` dataclass, parse `[[projects]]` array
2. **Project Detection** (1 file): Implement CWD matching algorithm
3. **Index Path Resolution** (1 file): Add XDG data directory logic
4. **Integration** (3 files): Wire detection into server/CLI initialization
5. **Validation** (1 file): Add config validation with error messages
6. **Testing** (4 files): Unit, integration, E2E tests
7. **Documentation** (3 files): Update specs, user docs, examples

**Estimated Total:** ~450 LOC across 13 files

---

## 10. Testing Strategy

### 9.1. Unit Tests

**File:** `tests/unit/test_config.py`

- `test_parse_projects_array()`: Parse `[[projects]]` from TOML
- `test_detect_project_exact_match()`: CWD equals project path
- `test_detect_project_subdirectory()`: CWD is child of project path
- `test_detect_project_nested_priority()`: Deepest project wins
- `test_detect_project_no_match()`: Returns None for unregistered path
- `test_resolve_index_path_local_override()`: Local `index_path` takes precedence
- `test_resolve_index_path_global_detected()`: Uses XDG data home for detected project
- `test_resolve_index_path_fallback()`: Uses `.index_data/` when no project detected
- `test_validate_duplicate_project_names()`: Error on duplicate names
- `test_validate_duplicate_project_paths()`: Error on duplicate paths

### 9.2. Integration Tests

**File:** `tests/integration/test_multi_project.py`

- `test_project_isolation()`: Index two projects, verify no cross-contamination
- `test_config_precedence()`: Local config overrides global settings
- `test_index_path_resolution_workflow()`: End-to-end path resolution with CWD changes
- `test_backward_compatibility()`: Existing project without global config still works

### 9.3. End-to-End Tests

**File:** `tests/e2e/test_multi_project_cli.py`

- `test_cli_project_detection()`: Run CLI from different CWDs, verify correct project selected
- `test_mcp_server_project_detection()`: Start MCP server in project subdirectory, query works
- `test_rebuild_index_per_project()`: Rebuild command respects detected project

---

## 11. Documentation Updates

### 10.1. Files to Create

- [docs/working/multi-project-implementation-plan.md](../docs/working/multi-project-implementation-plan.md): Handoff document
- [docs/guides/multi-project-setup.md](../docs/guides/multi-project-setup.md): User tutorial
- [examples/config-multi-project.toml](../examples/config-multi-project.toml): Reference configuration

### 10.2. Files to Update

- [README.md](../README.md): Add multi-project feature bullet point
- [docs/configuration.md](../docs/configuration.md): Document `[[projects]]` section
- [specs/05-configuration.md](../specs/05-configuration.md): Update config specification
