# Multi-Project Setup Guide

This guide explains how to configure mcp-markdown-ragdocs to manage documentation for multiple projects on your machine.

## Overview

Multi-project support allows you to:
- **Keep one global index** while preserving per-result `project_id` metadata
- **Centralize configuration** in a global config file
- **Automatically detect** which project you're working in based on current directory
- **Optionally uplift or explicitly filter** results by project when you ask for it

## Quick Start

### 1. Create Global Config

Create `~/.config/mcp-markdown-ragdocs/config.toml`:

```toml
# Register your projects
[[projects]]
name = "monorepo"
path = "/home/user/work/monorepo"

[[projects]]
name = "personal-notes"
path = "/home/user/Documents/notes"

# Optional: Global defaults for all projects
[indexing]
recursive = true

[search]
semantic_weight = 1.0
keyword_weight = 1.0
```

### 2. Verify Configuration

Run the check command from any registered project:

```bash
cd /home/user/work/monorepo/packages/api
mcp-markdown-ragdocs check-config
```

You should see:
- List of registered projects
- Active project detected: `monorepo`
- Index path: `~/.local/share/mcp-markdown-ragdocs`

### 3. Build Indices

Indices are now stored in one global location by default, and when `[[projects]]` are configured the default rebuild path will index across all registered project roots in one pass:

```bash
mcp-markdown-ragdocs rebuild-index
```

Project membership is preserved as metadata (`project_id`) inside the shared index, which enables ranking uplift and explicit filtering without silently partitioning storage.

If you want to narrow indexing/querying intentionally, use an explicit project/path override.

## Configuration Reference

### Project Definition

```toml
[[projects]]
name = "my-project"           # Required: alphanumeric, hyphens, underscores
path = "/absolute/path/root"  # Required: must be absolute
```

**Constraints:**
- `name`: Used as the stable `project_id`; must be unique
- `path`: Project root directory; must be unique and exist

### Project Detection

The server automatically detects your project by matching your current working directory (CWD) against registered project paths:

- **Exact match:** CWD = project path
- **Subdirectory:** CWD is child of project path
- **Nested projects:** Deepest match wins

**Example:**

```toml
[[projects]]
name = "parent"
path = "/home/user/parent"

[[projects]]
name = "child"
path = "/home/user/parent/child"
```

When you run from `/home/user/parent/child/src/`, the detected project is `child` (not `parent`).

### Config Precedence

Settings are merged with this precedence (highest first):

1. **Project-local config** (`.mcp-markdown-ragdocs/config.toml`)
2. **Global config** (`~/.config/mcp-markdown-ragdocs/config.toml`)
3. **Defaults**

**Special Case: Index Path**
- If local config sets `index_path`, it overrides the default global storage path
- Otherwise, the default authoritative path is `~/.local/share/mcp-markdown-ragdocs`
- Project detection influences ranking context and optional explicit path overrides, not the default index directory

## Advanced Usage

### Per-Project Overrides

Create `.mcp-markdown-ragdocs/config.toml` in a project to override global settings:

```toml
[indexing]
documents_path = "./documentation"  # Override default

[search]
semantic_weight = 2.0  # Emphasize semantic search for this project
```

### Indexing Performance Controls

You can tune indexing pressure globally:

```toml
[indexing]
torch_num_threads = 4
debounce_window_seconds = 0.5
task_backpressure_limit = 100
```

- `torch_num_threads` limits PyTorch/HuggingFace embedding thread fan-out
- `debounce_window_seconds` collapses rapid repeated file events so the last request wins
- `task_backpressure_limit` prevents the task queue from growing without bound during large change bursts

### Custom Index Location

Force a specific index path (bypasses global storage):

```toml
# In project-local config
[indexing]
index_path = "/mnt/fast-ssd/my-project-index"
```

### No Global Config (Single Project)

If no global config exists, behavior is unchanged:
- Uses `.mcp-markdown-ragdocs/config.toml` if present
- Falls back to `~/.config/mcp-markdown-ragdocs/config.toml`
- Defaults to `.index_data/` for storage

## Troubleshooting

### No Project Detected

**Symptom:** `check-config` shows "None detected"

**Causes:**
1. CWD not under any registered project path
2. No global config or empty `[[projects]]` array
3. Typo in project path

**Solution:**
```bash
# Verify project paths
cat ~/.config/mcp-markdown-ragdocs/config.toml

# Check current directory
pwd

# Ensure path is absolute and exists
ls -d /path/from/config
```

### Duplicate Names/Paths

**Symptom:** Server fails to start with `ValueError`

**Solution:** Edit global config to ensure unique names and paths:
```toml
# ❌ Invalid
[[projects]]
name = "project"
path = "/home/user/project"

[[projects]]
name = "project"  # Duplicate name!
path = "/home/user/other"

# ✅ Valid
[[projects]]
name = "project-a"
path = "/home/user/project"

[[projects]]
name = "project-b"
path = "/home/user/other"
```

### Index Not Found

**Symptom:** Queries return no results despite indexed documents

**Cause:** Index stored in a different location than expected, or documents were indexed from a different root than you intended

**Solution:**
```bash
# Check where index is stored
mcp-markdown-ragdocs check-config | grep "Index Path"

# Rebuild index
mcp-markdown-ragdocs rebuild-index
```

## Migration from Single Project

If you have an existing project using `.index_data/`:

1. **Option A: Keep local storage** (no action needed)
   - Server continues using `.index_data/`
   - No global config required

2. **Option B: Migrate to global storage**
   - Add project to global config
   - Run `rebuild-index` (indices regenerated automatically)
   - Optional: Remove old `.index_data/` directory

**No automatic migration is performed.** Indices are lightweight and rebuild quickly.

## Filtering vs. Context

- **Active project context** can apply a bounded `1.2x` uplift to matching results.
- **Explicit project filters** restrict eligibility only when you request them.
- The default query behavior remains global-first.
