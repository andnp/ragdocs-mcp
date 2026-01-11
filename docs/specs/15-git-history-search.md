# 15. Git History Search

## Executive Summary

**Purpose:** Enable semantic search over git commit history for mcp-markdown-ragdocs, surfacing relevant code changes and commit context through natural language queries.

**Scope:** Add `git_commits` table to SQLite DB with embeddings for commit metadata and truncated deltas. Implement recursive repository discovery with configurable exclusions. Support glob-based filtering of commits by files changed. Reuse existing embedding model, search pipeline, and incremental update patterns from document indexing infrastructure.

**Decision:** Separate commit index stored in new `git_commits` table with schema: `hash, timestamp, author, committer, title, message, files_changed, delta_truncated, embedding`. New MCP tool `search_git_history` returns commits only (no document overlap). Repository discovery via recursive `.git` directory search with exclusions matching document exclusion patterns. Delta truncation at 200 lines favoring early diff sections. Incremental updates via `.git` directory file watching using existing FileWatcher pattern.

---

## 1. Goals & Non-Goals

### Goals

1. **Semantic Commit Search:** Surface relevant commits via natural language queries (e.g., "authentication fixes in the last quarter").
2. **File Filtering:** Support recursive glob patterns (e.g., `src/**/*.py`) to filter commits by changed files.
3. **Infrastructure Reuse:** Leverage existing embedding model (BAAI/bge-small-en-v1.5), SQLite storage, search pipeline, and file watching patterns.
4. **Separate Indices:** Maintain isolated `git_commits` table; no mixing with document indices.
5. **Comprehensive Coverage:** Index all commits on all branches including merge commits, no time limits.
6. **Delta Context:** Include truncated delta (max 200 lines) for commit content analysis.
7. **Incremental Updates:** Watch `.git` directories for changes and incrementally update commit index.

### Non-Goals

1. **Unified Search:** No combined document + commit search in single tool call.
2. **Blame Integration:** No line-level authorship attribution or blame annotations.
3. **Branch Analysis:** No branch-specific queries or graph traversal (all commits treated equally).
4. **Delta Parsing:** No syntax-aware diff parsing or code context extraction beyond truncation.
5. **Commit Re-ranking:** No specialized scoring for commit metadata (use standard RRF).
6. **Git Hooks:** No real-time update via git hooks (file watching only).

---

## 2. Current State Analysis

### 2.1. Document Indexing Architecture

**Vector Index:** [src/indices/vector.py](../../src/indices/vector.py)

| Component | Implementation |
|-----------|----------------|
| Storage | FAISS IndexFlatL2 (in-memory, persisted binary) |
| Embedding Model | BAAI/bge-small-en-v1.5 (384 dims, HuggingFace) |
| Document Format | Chunk content + header_path prepended |
| Mapping Tables | `doc_id → [node_ids]`, `chunk_id → node_id` |
| Persistence | `{index_path}/vector/` directory with JSON mappings |

**Keyword Index:** [src/indices/keyword.py](../../src/indices/keyword.py)

| Component | Implementation |
|-----------|----------------|
| Engine | Whoosh with BM25F scoring |
| Schema | `id`, `doc_id`, `chunk_id`, `content`, `headers`, `title`, `aliases`, `tags`, `keywords` |
| Storage | On-disk Whoosh index files |
| Persistence | `{index_path}/keyword/` directory |

**Index Manager:** [src/indexing/manager.py](../../src/indexing/manager.py)

- Coordinates vector, keyword, graph, and code indices
- Computes doc_id from relative file path (no extension)
- Handles chunk-level indexing via `_chunker.chunk_document()`
- Persists all indices atomically
- Tracks failed files with error messages

**File Watcher:** [src/indexing/watcher.py](../../src/indexing/watcher.py)

| Component | Implementation |
|-----------|----------------|
| Library | watchdog.observers.Observer |
| Event Handler | Queues created/modified/deleted events |
| Debouncing | 0.5s cooldown via asyncio batch processing |
| Reconciliation | Periodic check for stale/new files (1 hour default) |
| Lifecycle | Start/stop async tasks with graceful shutdown |

**Search Orchestrator:** [src/search/orchestrator.py](../../src/search/orchestrator.py)

- Executes parallel queries across vector, keyword, (optional) code indices
- Applies RRF fusion with configurable weights
- Processes results through SearchPipeline (thresholding, dedup, doc limits, re-ranking)
- Returns `(list[ChunkResult], CompressionStats)` tuple
- Supports excluded_files filtering at retrieval time

### 2.2. Configuration Patterns

**Config Dataclasses:** [src/config.py](../../src/config.py)

```python
@dataclass
class IndexingConfig:
    documents_path: str = "."
    index_path: str = ".index_data/"
    recursive: bool = True
    include: list[str] = ["**/*"]
    exclude: list[str] = [
        "**/.venv/**", "**/venv/**", "**/build/**",
        "**/dist/**", "**/.git/**", "**/node_modules/**",
        "**/__pycache__/**", "**/.pytest_cache/**"
    ]
    exclude_hidden_dirs: bool = True
    reconciliation_interval_seconds: int = 3600
```

**Relevant Exclusions for Git Discovery:**
- `.stversions` (Syncthing versioning)
- `build/`, `dist/` (build artifacts)
- `.venv/`, `venv/` (virtual environments)
- `node_modules/` (JavaScript dependencies)

### 2.3. Search Pipeline Patterns

**Pipeline Configuration:** [src/search/pipeline.py](../../src/search/pipeline.py)

```python
@dataclass
class SearchPipelineConfig:
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 2
    dedup_enabled: bool = False
    dedup_threshold: float = 0.85
    ngram_dedup_enabled: bool = True
    ngram_dedup_threshold: float = 0.7
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10
```

**Processing Steps:**
1. Score normalization (min-max scaling)
2. Confidence thresholding
3. N-gram deduplication
4. Content-based deduplication (cosine similarity)
5. Per-document chunk limiting
6. MMR-based diversity (optional)
7. Cross-encoder re-ranking (optional)

**Reuse Opportunity:** Commit search can use same pipeline with `max_chunks_per_doc=1` (one commit = one chunk).

### 2.4. Storage Layout

**Current Index Structure:**

```
{index_path}/
├── index.manifest.json
├── vector/
│   ├── docstore.json
│   ├── index_store.json
│   ├── faiss_index.bin
│   ├── doc_id_mapping.json
│   ├── chunk_id_mapping.json
│   └── concept_vocabulary.json
├── keyword/
│   └── MAIN_*.{toc,seg,pos}
├── graph/
│   └── graph.json
└── code/
    └── MAIN_*.{toc,seg,pos}
```

**Proposed Addition:**

```
{index_path}/
├── git_commits.db          # NEW: SQLite database
└── (existing indices)
```

---

## 3. Proposed Solution

### 3.1. Schema Design

**SQLite Table: `git_commits`**

```sql
CREATE TABLE git_commits (
    hash TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,      -- Unix timestamp (seconds since epoch)
    author TEXT NOT NULL,             -- Author name <email>
    committer TEXT NOT NULL,          -- Committer name <email>
    title TEXT NOT NULL,              -- First line of commit message
    message TEXT NOT NULL,            -- Full commit message (body)
    files_changed TEXT NOT NULL,      -- JSON array of file paths
    delta_truncated TEXT NOT NULL,    -- Truncated diff (max 200 lines)
    embedding BLOB NOT NULL,          -- 384-dim float32 array (1536 bytes)
    indexed_at INTEGER NOT NULL       -- Unix timestamp of indexing
);

CREATE INDEX idx_timestamp ON git_commits(timestamp);
CREATE INDEX idx_indexed_at ON git_commits(indexed_at);
```

**Column Rationale:**

| Column | Type | Rationale |
|--------|------|-----------|
| `hash` | TEXT | SHA-1/SHA-256 commit hash (40/64 chars), primary key |
| `timestamp` | INTEGER | Unix timestamp for temporal queries, indexed for range scans |
| `author` | TEXT | Full attribution `Name <email>` format |
| `committer` | TEXT | Separate from author (rebases, merges) |
| `title` | TEXT | First line for display in results |
| `message` | TEXT | Full message for semantic search content |
| `files_changed` | TEXT | JSON array for glob filtering: `["src/a.py", "tests/b.py"]` |
| `delta_truncated` | TEXT | First 200 lines of `git show --format="" {hash}` output |
| `embedding` | BLOB | 384 floats × 4 bytes = 1536 bytes per commit |
| `indexed_at` | INTEGER | Track incremental updates and stale detection |

**Embedding Storage Format:**

```python
# Serialize 384-dim numpy array to bytes
embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

# Deserialize from bytes
embedding_array = np.frombuffer(blob, dtype=np.float32)
```

### 3.2. Commit Document Construction

**Searchable Text Format:**

```
{title}

{message}

Author: {author}
Committer: {committer}

Files changed:
{file_1}
{file_2}
...

{delta_truncated}
```

**Example:**

```
Fix authentication token expiration handling

Adds automatic refresh logic when token is within 5 minutes
of expiration. Prevents 401 errors during long-running operations.

Author: Jane Doe <jane@example.com>
Committer: CI Bot <ci@example.com>

Files changed:
src/auth/token_manager.py
tests/auth/test_token_refresh.py

diff --git a/src/auth/token_manager.py b/src/auth/token_manager.py
index abcdef1..1234567 100644
--- a/src/auth/token_manager.py
+++ b/src/auth/token_manager.py
@@ -45,6 +45,12 @@ class TokenManager:
     def get_token(self):
+        if self._is_near_expiration():
+            self._refresh_token()
         return self._current_token
(... remaining 195 lines of delta)
```

**Embedding Input:** Entire formatted text above (title, message, author, committer, files, delta).

### 3.3. Repository Discovery Algorithm

**Objective:** Find all `.git` directories recursively, respecting exclusions.

**Pseudocode:**

```python
def discover_git_repositories(
    documents_path: Path,
    exclude: list[str],
    exclude_hidden_dirs: bool
) -> list[Path]:
    """
    Recursively find .git directories.

    Returns:
        List of absolute paths to .git directories
    """
    git_dirs = []

    for root, dirs, files in os.walk(documents_path):
        # Filter directories in-place to prevent descending
        if exclude_hidden_dirs:
            dirs[:] = [d for d in dirs if not d.startswith('.') or d == '.git']

        # Apply glob exclusions
        root_path = Path(root)
        rel_path = root_path.relative_to(documents_path)

        # Check if current directory matches any exclude pattern
        excluded = False
        for pattern in exclude:
            if rel_path.match(pattern):
                excluded = True
                break

        if excluded:
            dirs.clear()  # Don't descend into excluded directories
            continue

        # Check if .git exists in current directory
        git_path = root_path / '.git'
        if git_path.is_dir():
            git_dirs.append(git_path)
            # Don't descend into .git contents
            dirs[:] = [d for d in dirs if d != '.git']

    return git_dirs
```

**Exclusion Patterns (from config):**

```toml
[indexing]
exclude = [
    "**/.stversions/**",   # Syncthing
    "**/build/**",         # Build artifacts
    "**/dist/**",
    "**/.venv/**",         # Virtual environments
    "**/venv/**",
    "**/node_modules/**",  # JavaScript deps
    "**/__pycache__/**",   # Python cache
    "**/.pytest_cache/**"
]
```

### 3.4. Delta Truncation Algorithm

**Objective:** Capture first 200 lines of diff output, preserving header and early hunks.

**Implementation:**

```python
def truncate_delta(commit_hash: str, max_lines: int = 200) -> str:
    """
    Get truncated diff for commit.

    Uses `git show --format="" {hash}` to get raw diff only.

    Args:
        commit_hash: Full commit SHA
        max_lines: Maximum lines to keep (default: 200)

    Returns:
        Truncated diff string with indicator if truncated
    """
    result = subprocess.run(
        ['git', 'show', '--format=', commit_hash],
        capture_output=True,
        text=True,
        cwd=repo_path
    )

    if result.returncode != 0:
        return f"Error retrieving delta: {result.stderr}"

    lines = result.stdout.splitlines()

    if len(lines) <= max_lines:
        return result.stdout

    # Keep first max_lines and add truncation marker
    truncated = '\n'.join(lines[:max_lines])
    remaining = len(lines) - max_lines
    truncated += f"\n\n... ({remaining} lines omitted)"

    return truncated
```

**Rationale for Early Truncation:**

1. **Header Preservation:** File names and change summaries appear first
2. **Hunk Context:** Early hunks often contain most significant changes
3. **Performance:** Embedding 200-line diffs is tractable (est. 2000-3000 tokens)
4. **Consistency:** Fixed truncation point prevents embedding size explosion

### 3.5. Files Changed Indexing Strategy

**Storage Format:** JSON array in TEXT column

```json
["src/auth/token_manager.py", "tests/auth/test_token_refresh.py", "docs/api.md"]
```

**Glob Filtering Algorithm:**

```python
def filter_by_glob(
    commits: list[dict],
    glob_pattern: str
) -> list[dict]:
    """
    Filter commits by glob pattern matching any changed file.

    Args:
        commits: List of commit dicts with 'files_changed' key
        glob_pattern: Glob pattern (e.g., 'src/**/*.py')

    Returns:
        Filtered list of commits
    """
    from pathlib import Path

    filtered = []
    for commit in commits:
        files_changed = json.loads(commit['files_changed'])

        for file_path in files_changed:
            if Path(file_path).match(glob_pattern):
                filtered.append(commit)
                break  # Match found, include commit

    return filtered
```

**Indexing for Performance:**

SQLite JSON functions could be used for direct query filtering:

```sql
SELECT * FROM git_commits
WHERE EXISTS (
    SELECT 1 FROM json_each(files_changed)
    WHERE json_each.value GLOB 'src/**/*.py'
);
```

However, Python-side filtering is simpler and sufficient for initial implementation (glob matching not directly supported in SQLite GLOB operator for JSON arrays).

### 3.6. Incremental Update Strategy

**Approach:** Detect new commits since last index update via timestamp comparison.

**Algorithm:**

```python
def get_new_commits(
    repo_path: Path,
    last_indexed_timestamp: int | None
) -> list[str]:
    """
    Get commit hashes added since last indexing.

    Args:
        repo_path: Path to .git directory
        last_indexed_timestamp: Unix timestamp of last index update (None = all commits)

    Returns:
        List of commit hashes (newest first)
    """
    if last_indexed_timestamp is None:
        # Full index: get all commits on all branches
        result = subprocess.run(
            ['git', 'log', '--all', '--format=%H'],
            capture_output=True,
            text=True,
            cwd=repo_path.parent
        )
    else:
        # Incremental: get commits after timestamp
        after_date = datetime.fromtimestamp(last_indexed_timestamp, timezone.utc)
        after_str = after_date.strftime('%Y-%m-%d %H:%M:%S')

        result = subprocess.run(
            ['git', 'log', '--all', f'--after={after_str}', '--format=%H'],
            capture_output=True,
            text=True,
            cwd=repo_path.parent
        )

    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr}")

    return result.stdout.strip().split('\n') if result.stdout.strip() else []
```

**Stale Commit Detection:**

Commits that disappear (e.g., force-pushed branches) are not actively detected. Acceptable trade-off: stale commits remain indexed but become irrelevant over time.

Optional: Periodic full reconciliation (similar to document reconciliation) could prune commits not found in `git log --all`.

### 3.7. File Watch Integration

**Objective:** Trigger incremental indexing when `.git` directory changes.

**Approach:**

1. **Extend FileWatcher to monitor `.git` directories:**

```python
class GitWatcher:
    def __init__(
        self,
        git_repos: list[Path],
        commit_indexer: CommitIndexer,
        cooldown: float = 5.0  # Longer cooldown for git operations
    ):
        self._git_repos = git_repos
        self._commit_indexer = commit_indexer
        self._cooldown = cooldown
        self._observers: list[BaseObserver] = []
        self._event_queue = queue.Queue[Path]()
        self._running = False
        self._task: asyncio.Task | None = None
```

2. **Watch specific files/directories in `.git`:**

```python
# Monitor HEAD, refs/, and objects/ for changes
watch_paths = [
    git_dir / 'HEAD',
    git_dir / 'refs',
    git_dir / 'objects'
]
```

3. **Event handler filters git-relevant changes:**

```python
class GitEventHandler(FileSystemEventHandler):
    def on_modified(self, event: FileSystemEvent):
        # Detect commits (refs/ or HEAD changes)
        path = Path(event.src_path)
        if 'refs' in path.parts or path.name == 'HEAD':
            self._queue.put_nowait(path.parent)  # Queue .git directory
```

4. **Batch processing with longer cooldown (5s):**

```python
async def _batch_process(self, git_repos: set[Path]):
    for git_dir in git_repos:
        try:
            await asyncio.to_thread(
                self._commit_indexer.update_incremental,
                git_dir
            )
            logger.info(f"Updated commit index for {git_dir.parent.name}")
        except Exception as e:
            logger.error(f"Failed to update commits for {git_dir}: {e}")
```

**Rationale for 5s Cooldown:**

- Git operations (push, fetch, rebase) may touch multiple files in rapid succession
- Longer cooldown ensures batch completes before indexing
- Reduces redundant indexing during interactive rebases

---

## 4. Decision Matrix

### 4.1. Storage Backend

| Option | Pros | Cons | Complexity | Extensibility | Risk | Cost | Performance |
|--------|------|------|------------|---------------|------|------|-------------|
| **SQLite DB ✅** | Self-contained, zero dependencies, ACID transactions, indexes | Requires table management | Low | High | Low | Low | High |
| FAISS Binary | Consistent with docs | No metadata storage, requires separate JSON | Medium | Low | Medium | Low | High |
| JSON Files | Simple | No indexing, slow filtering | Low | Low | High | Low | Low |

**Decision:** SQLite provides best balance of query flexibility (indexed timestamp, glob filtering) and storage efficiency (BLOB for embeddings).

### 4.2. Commit Scope

| Option | Pros | Cons | Complexity | Extensibility | Risk | Cost | Performance |
|--------|------|------|------------|---------------|------|------|-------------|
| **All branches ✅** | Comprehensive coverage | Large index for repos with many branches | Low | High | Low | High | Medium |
| Default branch only | Smaller index | Misses feature branches and history | Low | Low | High | Low | High |
| Configurable branches | Flexible | Complex configuration | High | Medium | Medium | Medium | Medium |

**Decision:** All branches (`git log --all`) provides comprehensive coverage without requiring branch configuration.

### 4.3. Delta Truncation Point

| Option | Pros | Cons | Complexity | Extensibility | Risk | Cost | Performance |
|--------|------|------|------------|---------------|------|------|-------------|
| **200 lines ✅** | Balances context and embedding size | May truncate important changes | Low | High | Low | Medium | High |
| No truncation | Complete context | Embedding explosion for large commits | Low | Low | High | High | Low |
| Adaptive (by file count) | Optimizes per commit | Complex heuristics | High | Medium | Medium | Medium | Medium |

**Decision:** Fixed 200-line truncation with early-line bias provides consistent embedding size and preserves critical context (file names, early hunks).

### 4.4. Tool Separation

| Option | Pros | Cons | Complexity | Extensibility | Risk | Cost | Performance |
|--------|------|------|------------|---------------|------|------|-------------|
| **Separate tool ✅** | Clear separation, simple API | Requires two calls for combined queries | Low | High | Low | Low | High |
| Unified tool | Single call | Complex filtering, result mixing | High | Medium | High | Medium | Medium |
| Search type parameter | Flexible | API complexity, conditional logic | Medium | Medium | Medium | Low | Medium |

**Decision:** Separate `search_git_history` tool maintains clean separation between document and commit indices, simplifying implementation and usage.

---

## 5. API Contract

### 5.1. MCP Tool: `search_git_history`

**Input Schema:**

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language query describing commits to find"
    },
    "top_n": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 100,
      "description": "Maximum number of commits to return"
    },
    "files_glob": {
      "type": "string",
      "description": "Optional glob pattern to filter by changed files (e.g., 'src/**/*.py')"
    },
    "after_timestamp": {
      "type": "integer",
      "description": "Optional Unix timestamp to filter commits after this date"
    },
    "before_timestamp": {
      "type": "integer",
      "description": "Optional Unix timestamp to filter commits before this date"
    }
  },
  "required": ["query"]
}
```

**Output Schema:**

```python
@dataclass
class CommitResult:
    hash: str                # Full commit SHA
    title: str               # First line of commit message
    author: str              # Author name <email>
    committer: str           # Committer name <email>
    timestamp: int           # Unix timestamp
    message: str             # Full commit message (body only, no title)
    files_changed: list[str] # List of changed file paths
    delta_truncated: str     # Truncated diff (max 200 lines)
    score: float             # Normalized relevance score [0.0, 1.0]
    repo_path: str           # Path to repository root

@dataclass
class GitSearchResponse:
    results: list[CommitResult]
    query: str
    total_commits_indexed: int  # Total commits in index
```

**Example Call:**

```python
# Find authentication-related commits in Python files
response = await mcp_tool.search_git_history(
    query="authentication token refresh logic",
    top_n=10,
    files_glob="src/**/*.py"
)

# Find commits from last quarter
three_months_ago = int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp())
response = await mcp_tool.search_git_history(
    query="fix memory leak",
    top_n=5,
    after_timestamp=three_months_ago
)
```

**Example Response:**

```json
{
  "results": [
    {
      "hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
      "title": "Fix authentication token expiration handling",
      "author": "Jane Doe <jane@example.com>",
      "committer": "CI Bot <ci@example.com>",
      "timestamp": 1704067200,
      "message": "Adds automatic refresh logic when token is within 5 minutes\nof expiration. Prevents 401 errors during long-running operations.",
      "files_changed": [
        "src/auth/token_manager.py",
        "tests/auth/test_token_refresh.py"
      ],
      "delta_truncated": "diff --git a/src/auth/token_manager.py ...\n(200 lines shown)",
      "score": 0.95,
      "repo_path": "/home/user/project"
    }
  ],
  "query": "authentication token refresh logic",
  "total_commits_indexed": 1234
}
```

### 5.2. Invariants

1. **No Overlap:** `search_git_history` returns commits only; `query_documents` returns documents/chunks only.
2. **Idempotency:** Multiple calls with same query and parameters return same results (modulo index updates).
3. **Ordering:** Results ordered by descending relevance score (highest first).
4. **Glob Filtering:** Applied pre-embedding-search (SQLite filtering) for efficiency.
5. **Timestamp Filtering:** Applied post-embedding-search (Python filtering) to avoid index fragmentation.

### 5.3. Failure Modes

| Failure | Cause | Behavior |
|---------|-------|----------|
| No git repos found | No `.git` directories in documents_path | Return empty results with `total_commits_indexed=0` |
| Git command failure | Invalid repo, missing git binary | Log error, skip repository, continue with others |
| Encoding errors in delta | Non-UTF-8 commit messages or diffs | Replace invalid bytes with `�`, log warning |
| Empty query | Query string is empty or whitespace | Return empty results, log warning |
| Invalid glob pattern | Malformed glob (e.g., unmatched brackets) | Raise ValueError with validation message |
| Database corruption | SQLite file corrupted | Raise RuntimeError, recommend index rebuild |

### 5.4. Concurrency

- **Read Operations:** SQLite connection in WAL mode supports concurrent reads
- **Write Operations:** Incremental updates acquire write lock, block concurrent writes
- **File Watching:** Queue-based event handling prevents race conditions
- **Embedding Generation:** Thread-safe (model loaded once, inference serialized)

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (3-4 hours, ~250 LOC)

**Tasks:**

1. **Create CommitIndexer class** (`src/indexing/commit_indexer.py`)
   - SQLite table creation with schema
   - Connection management (context managers)
   - Basic CRUD operations (add, query, remove)
   - Embedding serialization/deserialization

2. **Repository discovery** (`src/indexing/repo_discovery.py`)
   - Implement `discover_git_repositories()`
   - Respect exclude patterns from IndexingConfig
   - Unit tests for exclusion logic

3. **Commit parsing** (`src/indexing/commit_parser.py`)
   - Extract commit metadata via `git show`
   - Parse author, committer, timestamp, message
   - Build commit document text format
   - Delta truncation to 200 lines

**Acceptance Criteria:**

- SQLite table created with correct schema
- Repository discovery finds all `.git` directories excluding configured patterns
- Commit metadata extracted correctly from sample commits
- Delta truncation preserves first 200 lines

**Files:**

- `src/indexing/commit_indexer.py` (new, ~120 LOC)
- `src/indexing/repo_discovery.py` (new, ~60 LOC)
- `src/indexing/commit_parser.py` (new, ~70 LOC)

### Phase 2: Embedding & Search Integration (2-3 hours, ~180 LOC)

**Tasks:**

1. **Embed commits** (integrate with VectorIndex embedding model)
   - Reuse `VectorIndex._embedding_model` for commit text
   - Batch embedding for performance (process 100 commits at a time)
   - Store embeddings in `git_commits.embedding` column

2. **Semantic search implementation**
   - Query embedding generation
   - Cosine similarity scoring (numpy vectorized)
   - Top-k retrieval with score normalization

3. **Glob filtering**
   - Implement `filter_by_glob()` for files_changed
   - Support recursive patterns (`**`)
   - Unit tests for pattern matching

**Acceptance Criteria:**

- Commits embedded using existing model
- Semantic search returns relevant commits ranked by similarity
- Glob filtering correctly matches changed files
- Batch embedding completes in <5s per 100 commits

**Files:**

- `src/indexing/commit_indexer.py` (modify, +80 LOC)
- `src/search/commit_search.py` (new, ~100 LOC)

### Phase 3: Incremental Updates & File Watching (2-3 hours, ~150 LOC)

**Tasks:**

1. **Incremental commit indexing**
   - Implement `get_new_commits()` with timestamp filtering
   - Track `indexed_at` timestamp per repository
   - Handle edge cases (empty repos, initial index)

2. **GitWatcher implementation**
   - Extend FileWatcher pattern for `.git` directories
   - Monitor `HEAD`, `refs/`, `objects/` for changes
   - 5-second cooldown for git operations
   - Batch processing of repository updates

3. **Startup reconciliation**
   - Check for new commits on server startup
   - Update index before starting watcher

**Acceptance Criteria:**

- New commits detected and indexed within 10s of push
- No redundant indexing of existing commits
- Watcher handles rapid successive git operations gracefully
- Startup reconciliation completes in <30s for 1000-commit repos

**Files:**

- `src/indexing/commit_indexer.py` (modify, +50 LOC)
- `src/indexing/git_watcher.py` (new, ~100 LOC)

### Phase 4: MCP Tool & API (2 hours, ~120 LOC)

**Tasks:**

1. **Tool registration** (`src/mcp_server.py`)
   - Define `search_git_history` tool schema
   - Parameter validation (top_n, glob patterns, timestamps)
   - Error handling and user-friendly messages

2. **Response formatting**
   - Serialize CommitResult to JSON
   - Format delta for display (syntax highlighting optional)
   - Include repository context in results

3. **Integration with ApplicationContext**
   - Initialize CommitIndexer during startup
   - Add commit index to lifecycle management (persist, load)

**Acceptance Criteria:**

- Tool callable via MCP protocol
- Results formatted according to schema
- Query latency <500ms for 10k commit index
- Graceful degradation if no commits found

**Files:**

- `src/mcp_server.py` (modify, +60 LOC)
- `src/models.py` (modify, +30 LOC for CommitResult dataclass)
- `src/context.py` (modify, +30 LOC)

### Phase 5: Testing & Documentation (3-4 hours, ~400 LOC tests)

**Tasks:**

1. **Unit tests**
   - Repository discovery edge cases
   - Commit parsing error handling
   - Glob filtering patterns
   - Delta truncation boundaries

2. **Integration tests**
   - End-to-end commit indexing
   - Search relevance spot checks
   - Incremental update correctness

3. **E2E tests**
   - Full workflow: discover repos → index commits → search → results
   - Multi-repository scenarios
   - File watching integration

4. **Documentation**
   - Update [docs/architecture.md](../../docs/architecture.md) with commit indexing
   - Add [docs/git-search.md](../../docs/git-search.md) user guide
   - Update [README.md](../../README.md) with feature description

**Acceptance Criteria:**

- 90%+ test coverage for new modules
- All integration tests pass
- Documentation includes usage examples and configuration
- CHANGELOG updated with feature description

**Files:**

- `tests/unit/test_commit_indexer.py` (new, ~150 LOC)
- `tests/unit/test_repo_discovery.py` (new, ~80 LOC)
- `tests/integration/test_git_search.py` (new, ~120 LOC)
- `tests/e2e/test_git_workflow.py` (new, ~50 LOC)
- `docs/architecture.md` (modify, +50 LOC)
- `docs/git-search.md` (new, ~100 LOC)
- `README.md` (modify, +20 LOC)

### Phase 6: Polish & Optimization (2-3 hours, ~50 LOC)

**Tasks:**

1. **Performance optimization**
   - SQLite query optimization (EXPLAIN QUERY PLAN)
   - Index tuning (add missing indices if needed)
   - Batch size tuning for embedding generation

2. **Error recovery**
   - Handle corrupted repositories gracefully
   - Log actionable error messages
   - Provide rebuild mechanism for commit index

3. **Configuration options**
   - Add `[git_indexing]` section to config
   - Configurable delta truncation length
   - Configurable commit query batch size

**Acceptance Criteria:**

- Search latency <300ms for 10k commits (p95)
- No crashes on malformed repositories
- Configuration options documented and tested
- Index rebuild command functional

**CLI Integration:**

The `rebuild-index` CLI command triggers git commit indexing when `git_indexing.enabled = true`. This provides a convenient way to rebuild both document and commit indices in a single operation. The command displays progress bars for both phases and handles failures gracefully.

**Files:**

- `src/config.py` (modify, +20 LOC for GitIndexingConfig)
- `src/cli.py` (modify, +30 LOC for rebuild-commit-index command)

---

## 7. Testing Strategy

### 7.1. Unit Tests

**Coverage Targets:**

| Module | Tests | Coverage Goal |
|--------|-------|---------------|
| `commit_indexer.py` | Table creation, CRUD ops, serialization | 95% |
| `repo_discovery.py` | Exclusion patterns, nested repos | 90% |
| `commit_parser.py` | Metadata extraction, delta truncation | 90% |
| `commit_search.py` | Embedding search, glob filtering | 85% |
| `git_watcher.py` | Event handling, debouncing | 80% |

**Key Test Cases:**

1. **Repository Discovery:**
   - Nested repositories (monorepo with submodules)
   - Hidden directory exclusion
   - Glob pattern matching (positive and negative)

2. **Commit Parsing:**
   - Standard commits (single author, committer)
   - Merge commits (multiple parents)
   - Commits with non-UTF-8 messages
   - Commits with large diffs (>200 lines)

3. **Glob Filtering:**
   - Exact match (`src/file.py`)
   - Wildcard (`src/*.py`)
   - Recursive (`src/**/*.py`)
   - Negation (not supported, test error handling)

### 7.2. Integration Tests

**Test Scenarios:**

1. **Full Index Build:**
   - Discover 2 repositories with 50 commits each
   - Verify all commits indexed with correct metadata
   - Check embedding storage and retrieval

2. **Incremental Update:**
   - Index initial commits
   - Add new commits via `git commit`
   - Verify only new commits indexed

3. **Search Relevance:**
   - Index commits with known content
   - Execute queries with expected results
   - Verify ranking order

4. **Glob Filtering:**
   - Query with `files_glob="src/**/*.py"`
   - Verify only Python file commits returned

### 7.3. E2E Tests

**Workflow Tests:**

1. **Cold Start:**
   - Server startup with no commit index
   - Full repository discovery and indexing
   - First query returns results

2. **Incremental Update:**
   - Server running with indexed commits
   - Make new commit in watched repository
   - Query returns new commit within 10s

3. **Multi-Repository:**
   - Index 3 repositories in different subdirectories
   - Query matches commits from all repositories
   - Results include correct `repo_path`

### 7.4. Performance Benchmarks

**Target Metrics:**

| Scenario | Metric | Target |
|----------|--------|--------|
| Full index (1000 commits) | Time | <60s |
| Incremental update (10 commits) | Time | <10s |
| Query (10k commit index) | Latency (p50) | <200ms |
| Query (10k commit index) | Latency (p95) | <500ms |
| Embedding generation (100 commits) | Time | <5s |

---

## 8. Risk Register

### 8.1. High Risks

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| **Large repository performance** | Query latency >1s, poor UX | High | Implement pagination, add SQLite query optimization, limit initial index to last N commits | Architect |
| **Embedding size explosion** | OOM for large commit history | Medium | Fixed 200-line delta truncation, batch processing with memory monitoring | Architect |
| **Git command failures** | Partial index, missing commits | Medium | Robust error handling, skip failed repos, log errors with actionable messages | Code Agent |

### 8.2. Medium Risks

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| **Encoding errors in deltas** | Indexing failures, corrupted text | Medium | Try UTF-8 → Latin-1 → CP1252 fallback chain, replace invalid bytes | Code Agent |
| **Watch file descriptor limits** | FileWatcher crashes on many repos | Low | Document file descriptor limit increase (`ulimit -n 4096`), warn if >10 repos | Architect |
| **SQLite locking contention** | Slow writes during concurrent queries | Low | Use WAL mode, read-heavy workload unlikely to cause issues | Code Agent |

### 8.3. Low Risks

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| **Glob pattern parsing edge cases** | Incorrect filtering | Low | Comprehensive unit tests, validate patterns at tool invocation | Code Agent |
| **Commit timestamp timezone issues** | Incorrect temporal filtering | Low | Store Unix timestamps (UTC), document timezone handling | Architect |

---

## 9. Assumptions

1. **Git Availability:** `git` binary is available in system PATH.
2. **Repository Health:** Repositories are not corrupted (`.git` directory intact).
3. **Commit Volume:** Most repositories have <10k commits (performance tested to this scale).
4. **File Descriptor Limits:** System supports at least 1024 open file descriptors (standard Linux default).
5. **Embedding Model Reuse:** BAAI/bge-small-en-v1.5 model suitable for commit text (same as documents).
6. **Disk Space:** Sufficient disk space for SQLite DB (approx. 2KB per commit: 1.5KB embedding + 0.5KB metadata).
7. **UTF-8 Commits:** Majority of commits use UTF-8 encoding (fallback to Latin-1 for legacy).

---

## 10. Appendix: Example Queries

### 10.1. Semantic Queries

```python
# Find bug fixes
search_git_history(query="fix null pointer exception")

# Find feature additions
search_git_history(query="add support for websockets")

# Find refactoring work
search_git_history(query="refactor database connection pooling")

# Find security fixes
search_git_history(query="prevent SQL injection")
```

### 10.2. File-Filtered Queries

```python
# Python files only
search_git_history(
    query="authentication improvements",
    files_glob="src/**/*.py"
)

# Configuration changes
search_git_history(
    query="update production config",
    files_glob="config/*.toml"
)

# Test changes
search_git_history(
    query="add integration tests",
    files_glob="tests/**/*.py"
)
```

### 10.3. Temporal Queries

```python
# Last month
one_month_ago = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
search_git_history(
    query="performance optimizations",
    after_timestamp=one_month_ago
)

# Specific date range
q3_start = int(datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp())
q3_end = int(datetime(2024, 9, 30, tzinfo=timezone.utc).timestamp())
search_git_history(
    query="API changes",
    after_timestamp=q3_start,
    before_timestamp=q3_end
)
```

### 10.4. Combined Filters

```python
# Recent authentication changes in Python
last_week = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())
search_git_history(
    query="authentication token handling",
    files_glob="src/auth/**/*.py",
    after_timestamp=last_week,
    top_n=10
)
```

---

## 11. Open Questions

None (all design decisions made based on user requirements).

---

## 12. References

1. **Existing Specs:**
   - [specs/11-search-quality-improvements.md](11-search-quality-improvements.md) - Search pipeline patterns
   - [specs/10-multi-project-support.md](10-multi-project-support.md) - Project detection and isolation
   - [specs/08-document-chunking.md](08-document-chunking.md) - Chunking strategies

2. **Implementation Files:**
   - [src/indices/vector.py](../../src/indices/vector.py) - Embedding model and FAISS storage
   - [src/indexing/watcher.py](../../src/indexing/watcher.py) - File watching patterns
   - [src/search/orchestrator.py](../../src/search/orchestrator.py) - Search pipeline integration

3. **External:**
   - FAISS documentation: https://github.com/facebookresearch/faiss
   - SQLite BLOB storage: https://www.sqlite.org/datatype3.html
   - Git log formats: https://git-scm.com/docs/git-log
