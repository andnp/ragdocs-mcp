# Plan: Extract `searchkernel/` into a Standalone PyPI Library — Implementation Tranches

**Status:** Planning (2026-07-30)

**Governing design:** Plan #10 (Promote ragdocs In-Place into the Search Kernel)

**Repo scope:** This plan spans TWO repositories:
- **`mcp-markdown-ragdocs`** (application layer): owns daemon lifecycle, file-watching, content-source adapters (git, markdown, plaintext), CLI, MCP server, bootstrap coordination
- **`andnp-searchkernel`** (new library): extracted generic search/indexing kernel (import name: `searchkernel`, dist name: `andnp-searchkernel`)

---

## Executive summary

Plan #10 successfully refactored ragdocs into a layered `searchkernel/` package with clean ports and generic query/ingestion pipeline stages. However, `searchkernel/` currently lives inside the ragdocs repo and is tightly coupled to ragdocs-specific bootstrap, file-watching, and task-queue logic. This plan extracts `searchkernel/` into its own standalone PyPI library (`andnp-searchkernel`) so that:

1. Any Python project can `pip install andnp-searchkernel` and build a search kernel over their own content sources (not just markdown files and git commits).
2. The library owns only source-agnostic, domain-generic ingestion/search/indexing primitives.
3. `mcp-markdown-ragdocs` becomes the application-layer owner: it wires content sources (git, markdown, plaintext, future: Google Drive), daemon lifecycle, file-watching, CLI, and MCP server on top of the library.
4. Library and app ship independently: ragdocs can evolve its UI/daemon/integrations without bumping the library version; new content sources prove the library's extensibility without library changes.

---

## Library vs app boundary (finalized)

### Top-level directory classification

| Directory/File | Classification | Rationale |
|---|---|---|
| `searchkernel/domain/` | **LIBRARY** | `Record`, `Chunk`, `SearchResult`, domain primitives; source-agnostic |
| `searchkernel/ports/` | **LIBRARY** | `ContentSource`, `EmbeddingProvider`, `Reranker`, `VectorStore`, `KeywordStore`, `GraphStore`, `LLMProvider` protocols; extensibility contracts |
| `searchkernel/adapters/embedding/` | **LIBRARY** | `HuggingFaceEmbeddingProvider`; pluggable encoder implementations |
| `searchkernel/adapters/llm/` | **LIBRARY** | `CopilotLLMProvider`, tiered LLM facade; pluggable completion |
| `searchkernel/adapters/rerank/` | **LIBRARY** | `HuggingFaceReranker`; pluggable reranking |
| `searchkernel/adapters/stores/` | **LIBRARY** | `PGVectorIndex`, `VectorIndex` (FAISS), `KeywordIndex` (SQLite), `GraphStore` (SQLite); port implementations |
| `searchkernel/adapters/cache/` | **LIBRARY** | In-memory/SQLite cache implementations behind `CacheStore` port |
| `searchkernel/adapters/sources/git.py` | **APP** | Git-specific `ContentSource`; stays in app (proves the port, but git is a ragdocs choice, not universal) |
| `searchkernel/adapters/sources/{local.py,markdown.py,plaintext.py}` | **APP** | Markdown/plaintext parsing; content-source adapters for ragdocs (not part of library) |
| `searchkernel/pipeline/` | **LIBRARY** | `SearchStage`, `PipelineSpec`, `PipelineExecutor`; query-execution contracts and default pipeline |
| `searchkernel/search/` | **LIBRARY** | Query orchestration, scoring, dedup, graph expansion, routing; generic search logic |
| `searchkernel/eval/` | **LIBRARY** | Metrics, golden-set schema, eval runner; quality measurement infrastructure |
| `searchkernel/chunking/` | **LIBRARY** | Chunking strategies; generic text partitioning |
| `searchkernel/indices/` | **LIBRARY** | `VectorIndex`, `KeywordIndex`, `GraphStore`; in-memory index APIs |
| `searchkernel/storage/` | **LIBRARY** | Postgres schema, SQLite schema; storage layer abstractions |
| `searchkernel/runtime/` | **LIBRARY** | `@cached` decorator, fan-out helper, `QueryTrace`; generic runtime utilities |
| `searchkernel/utils/` | **LIBRARY** | Atomic I/O, path utilities; generic helpers |
| `searchkernel/config.py` | **APP** | Config schema with ragdocs-specific fields (documents_path, projects, etc.) |
| `searchkernel/models.py` | **APP** | `Document` construction from files; app-specific record shapes (stays in app, library uses domain-agnostic `Record`) |
| `searchkernel/compression/` | **LIBRARY** | `CompressionStats`, compression-aware result processing |
| `searchkernel/parsers/dispatcher.py` | **APP** | Parser selection logic; app-specific |
| `searchkernel/parsers/{markdown.py,plaintext.py}` | **APP** | Content parsing; markdown/plaintext specific to ragdocs |

### `searchkernel/indexing/` detailed breakdown

| File | Classification | Rationale |
|---|---|---|
| `bootstrap_checkpoint.py` | **LIBRARY** | Pure data structure + serialization for checkpoint persistence; reusable for any indexing system |
| `bootstrap_session.py` | **APP** | Orchestrates bootstrap during daemon startup; references "publish_public_state", "mark_ready", "schedule_embedding_warmup" — daemon lifecycle callbacks |
| `bootstrap_snapshot.py` | **LIBRARY** | Pure state derivation (`PublicIndexStateSnapshot`, `BootstrapReadinessSnapshot`); no daemon knowledge |
| `discovery.py` | **LIBRARY** | Generic file discovery, pattern matching, directory walking; source-agnostic |
| `embedding_cache.py` | **LIBRARY** | `SQLiteEmbeddingCache`: content-hash-keyed vector storage; generic cache implementation |
| `git_ingestion.py` | **LIBRARY** | Thin adapter: iterate `ContentSource` records → `index_record` calls; pure ingestion plumbing |
| `git_refresh_state.py` | **APP** | Durable cursors for file-watching-driven git refreshes; app-specific task state |
| `implicit_graph.py` | **LIBRARY** | Generic graph edge building (directory siblings, shared tags); works on any `GraphStore` |
| `manifest.py` | **LIBRARY** | `IndexManifest`: versioning, spec tracking; detects rebuild need; reusable configuration |
| `migration.py` | **LIBRARY** | Detects/cleans legacy index formats (Whoosh, snapshots); filesystem cleanup utility |
| `reconciler.py` | **LIBRARY** | Reconciles discovered files vs. manifest state; pure diff computation |
| `runtime_readiness.py` | **LIBRARY** | `SearchAvailability`, readiness/availability state derivation; reusable readiness policy |
| `semantic.py` | **LIBRARY** | `SemanticWorkPlanner`, encoder fingerprinting, cache-aware embedding orchestration; generic semantic work |
| `stages.py` | **LIBRARY** | `PreparedIndexBatch`, batch-oriented stage protocol, `build_graph_payload`; generic stage construction |
| `tasks.py` | **SPLIT** | **`TaskSubmissionResult`, `TaskBatchSubmissionResult`** → LIBRARY (generic task-queue wrapper); **task definitions (`submit_index_batch`, etc.)** → APP (huey-specific, daemon-owned) |
| `manager.py` | **SPLIT** | **Core indexing logic** (`index_document`, `index_record`, chunking/embedding/indexing sequence) → LIBRARY; **file-watching, delta tracking, bootstrap orchestration, task submission** → APP |
| `watcher.py` | **APP** | File-watching daemon using `watchdog.Observer`; inherently app-specific |
| `rebuild_service.py` | **APP** | Rebuild command orchestration; daemon runtime state, project scoping, progress reporting |

---

## Workstreams

### X1 — Finalize the library boundary (in progress, being completed by this doc)

**Goal:** Classify every file under `searchkernel/indexing/` and resolve the two mixed files (`manager.py`, `tasks.py`).

**Depends on:** None (this is the prerequisite).

**Deliverables**
1. Finalized library-vs-app boundary for all `searchkernel/indexing/` files (above table).
2. Documented split strategy for `manager.py` and `tasks.py`.
3. This plan document as reference for all following workstreams.

**Primary files**
- This plan doc.
- `searchkernel/indexing/*` — all files audited.

**Tasks**
1. Read all `searchkernel/indexing/` files and classify (DONE).
2. Identify mixed files and design split (DONE: manager.py, tasks.py need splitting).
3. Document boundary table with one-line rationale per file (DONE).
4. Document split strategy (see below).

**Split strategy for `manager.py` (36K)**
- **LIBRARY** → new file `searchkernel/indexing/core.py`: `IndexCore` class owning chunking, embedding, indexing steps (pure transforms on `Record`/`Chunk`/vectors).
- **APP** → keep in `manager.py`: `IndexManager` wrapping `IndexCore` + `FileWatcher` coordination, delta tracking (`_hash_store`), bootstrap session integration, task-submission callbacks.

**Split strategy for `tasks.py` (29.7K)**
- **LIBRARY** → new file `searchkernel/indexing/submission.py`: `TaskSubmissionResult`, `TaskBatchSubmissionResult` data classes; abstract submission interface.
- **APP** → keep in `tasks.py`: huey task definitions, task enqueue logic, queue coordination (`submit_index_batch`, `submit_refresh_git_batch`, etc.).

**Acceptance**
- Every file in `searchkernel/indexing/` is classified as LIBRARY, APP, or explicitly SPLIT with a split design documented.
- The rationale table is correct and matches actual code dependencies.

**Tests**
- (None — this is planning only.)

---

### X2 — Scaffold new repo `andnp-searchkernel`

**Goal:** Create a minimal new repository with pyproject.toml, CI, and initial commit from a clean state.

**Depends on:** X1.

**Deliverables**
1. New GitHub repo `andnp/andnp-searchkernel` (or similar, final name TBD).
2. `pyproject.toml`: dist name `andnp-searchkernel`, import name `searchkernel`, hatchling build, `py3.9+`.
3. `searchkernel/__init__.py`: public API re-exports (empty initially, filled by X3).
4. CI: GitHub Actions for `pytest`, `ruff check --fix` (linting + autofix), `import-linter` (inward dependencies), tag-push trusted publishing to PyPI.
5. README + minimal docs structure.

**Primary files**
- new repo root: `pyproject.toml`, `.github/workflows/lint-test-publish.yml`, `README.md`, `searchkernel/__init__.py`, `CHANGELOG.md`.

**Tasks**
1. Create repo on GitHub (manual, outside this plan).
2. Clone locally and add pyproject.toml (dist: `andnp-searchkernel`, import: `searchkernel`, version: `0.1.0.dev0`).
3. Add CI workflow: lint (`ruff check --fix`) + test (`pytest`) + import-linter check + tag-push trusted publish.
4. Add README with "coming soon" note (no content before X3 moves code).

**Acceptance**
- Repo is public and cloneable.
- `pytest` runs (even if suite is empty).
- `ruff check --fix` passes.
- `import-linter` passes (no imports to lint yet).
- CI is green on the initial commit.

**Tests**
- (CI workflow test — lint + pytest collect.)

---

### X3 — History-preserving move: `git subtree split` LIBRARY directories

**Goal:** Extract LIBRARY-classified files out of `mcp-markdown-ragdocs` into the new repo, preserving Git history via `git subtree split` or `git filter-repo`.

**Depends on:** X2 (target repo exists).

**Deliverables**
1. Filtered git history: every commit touching a LIBRARY file is replayed into the new repo.
2. All LIBRARY dirs + files are now in `andnp-searchkernel/searchkernel/`.
3. History is preserved: `git log searchkernel/domain/` in the new repo shows all original commits.
4. Commit SHAs are new (subtree split is a rebase), parent chain is rooted at a synthetic commit with a note about origin.

**Primary files**
- `andnp-searchkernel/searchkernel/domain/`, `ports/`, `adapters/`, `indices/`, `pipeline/`, `search/`, `eval/`, `chunking/`, `storage/`, `compression/`, `runtime/`, `utils/`.
- `andnp-searchkernel/searchkernel/indexing/` (LIBRARY files only: `bootstrap_checkpoint.py`, `bootstrap_snapshot.py`, `discovery.py`, `embedding_cache.py`, `git_ingestion.py`, `implicit_graph.py`, `manifest.py`, `migration.py`, `reconciler.py`, `runtime_readiness.py`, `semantic.py`, `stages.py`, plus LIBRARY half of `manager.py` and `tasks.py` after X1 splits).
- (NOT moved: APP files: `bootstrap_session.py`, `git_refresh_state.py`, `tasks.py` (APP half), `watcher.py`, `rebuild_service.py`, `manager.py` (APP half).)

**Tasks**
1. In `mcp-markdown-ragdocs`, run `git subtree split --prefix=searchkernel --branch library-only` to create a new branch with only commits touching library paths.
   - Alternative: use `git filter-repo --path searchkernel/domain --path searchkernel/ports ...` (list every LIBRARY dir).
2. Push the split tree into the new repo as the initial commit (or rebase atop X2's initial commit).
3. Verify history: `git log --oneline` shows each original commit; `git show <commit>` shows only LIBRARY changes.
4. Clean up: push merged branch to `andnp-searchkernel/main`.

**Acceptance**
- Every commit in `andnp-searchkernel/main` touches only LIBRARY files.
- `git log --all | grep searchkernel` in the new repo shows commits.
- No APP-layer code is present (no `bootstrap_session.py`, `watcher.py`, etc. in the new repo).
- Import-linter + tests pass in the new repo (see X6).

**Tests**
- `pytest` in new repo (unit tests for library modules).
- `import-linter` in new repo (inward dependencies).

---

### X4 — Cut over `mcp-markdown-ragdocs`: delete moved directories, add `andnp-searchkernel` as editable dependency

**Goal:** Remove LIBRARY code from ragdocs, add it back as a local path/editable dependency for iteration, fix imports repo-wide.

**Depends on:** X3 (library repo exists with code).

**Deliverables**
1. Delete all LIBRARY directories from `mcp-markdown-ragdocs/searchkernel/`.
2. Add `andnp-searchkernel` to `pyproject.toml` as `path = "../andnp-searchkernel"` (editable, for local development).
3. Fix all import statements: `from searchkernel.domain import ...` now resolves from the external package, no changes needed to imports.
4. App-layer `searchkernel/` now contains ONLY app-owned files (config, models, daemon/cli/bootstrap/watcher/rebuild, source adapters).

**Primary files**
- `mcp-markdown-ragdocs/pyproject.toml`: add `andnp-searchkernel = { path = "../andnp-searchkernel", editable = true }`.
- Delete from `mcp-markdown-ragdocs/searchkernel/`: domain/, ports/, adapters/ (except sources/), indices/, pipeline/, search/, eval/, chunking/, storage/, compression/, runtime/, utils/.
- Keep in `mcp-markdown-ragdocs/searchkernel/`: app-owned dirs + config.py, models.py, __init__.py, cli.py, server.py, context.py, lifecycle.py, daemon/, coordination/, cli_utils/, app/, git/, parsers/, adapters/sources/.

**Tasks**
1. In local worktree, delete all LIBRARY dirs from ragdocs.
2. Update `pyproject.toml` to list `andnp-searchkernel` as editable path dependency.
3. Run `ruff check --fix` to reformat any straggler imports (they should not change, just verify).
4. Verify all imports still resolve: `python -c "from searchkernel.domain import Record"` (should resolve from the external package).
5. Run unit tests on ragdocs: anything that imports library modules should still work.

**Acceptance**
- `mcp-markdown-ragdocs/searchkernel/` has only app-owned directories and config/models.
- `from searchkernel.domain import Record` resolves to `andnp-searchkernel` (editable installation).
- No import errors in the app.
- `ruff check` passes.
- Tests pass (or fail identically to pre-deletion state).

**Tests**
- `pytest tests/ -k "not integration"` (unit suite, fast smoke test).
- `ruff check` + `import-linter` in ragdocs.

---

### X5 — Rewire source adapters to depend only on library public surface

**Goal:** Ensure `mcp-markdown-ragdocs`' app-owned source adapters (git, markdown, plaintext, future: gdrive) depend only on the library's ports, not private internals.

**Depends on:** X4 (library is now external).

**Deliverables**
1. `adapters/sources/git.py` → depends only on `searchkernel.ports.ContentSource`, `searchkernel.domain.Record`, and git-specific logic; no imports of `searchkernel.indexing`, `searchkernel.indices`, etc.
2. `adapters/sources/local.py` (future) → same shape: only ports + domain.
3. `parsers/{markdown,plaintext}.py` → same: only ports + domain.
4. `import-linter` contract in both repos: forbid reaching past `searchkernel`' top-level `__init__.py` exports (i.e., app code can only `from searchkernel import ...` or `from searchkernel.ports import ...`, no `from searchkernel.indexing.manager import IndexManager`).

**Primary files**
- `mcp-markdown-ragdocs/searchkernel/adapters/sources/git.py`: already correct (only uses `ContentSource`, `Record` ports).
- `mcp-markdown-ragdocs/.claude/settings.json`: add import-linter contract rule or update existing.

**Tasks**
1. Audit `adapters/sources/git.py` → verify it only imports `ContentSource`, `Record`, no indexing internals (DONE: already correct).
2. Check that `local.py` + parsers follow the same pattern (if they exist; they may not yet).
3. Add import-linter rule to `.claude/settings.json` if not present: forbid `from searchkernel.indexing import ...` from app-owned code.
4. Verify the rule is enforced in CI.

**Acceptance**
- No app-owned file imports `searchkernel.indexing.*` or `searchkernel.indices.*` directly.
- `import-linter` enforces this (fails the build if violated).
- Source adapters are pluggable at the `ContentSource` port level; adding a new source (e.g., Google Drive) requires no library changes.

**Tests**
- `import-linter` pass.
- Lint rule test: add a deliberate bad import and verify the rule fails.

---

### X6 — Split tests: move generic unit tests to library, keep app/e2e/daemon tests in ragdocs

**Goal:** Partition the test suite so library unit tests live in `andnp-searchkernel`, app-specific tests stay in `mcp-markdown-ragdocs`.

**Depends on:** X4 (code is already split).

**Deliverables**
1. Move all unit tests for LIBRARY modules to `andnp-searchkernel/tests/`.
2. Keep in `mcp-markdown-ragdocs/tests/`: app integration, daemon, MCP, file-watching, bootstrap, git-refresh, CLI tests.
3. Both repos have independent green test suites.
4. CI runs both suites independently (library CI tests library only, app CI tests app only).

**Primary files**
- `andnp-searchkernel/tests/unit/{domain,ports,adapters,indices,pipeline,search,eval,chunking,storage,compression,runtime,utils,indexing}/*.py` — migrated from ragdocs.
- `mcp-markdown-ragdocs/tests/` — app-specific tests only.

**Tasks**
1. In ragdocs, identify tests that touch LIBRARY-only modules (`test_domain_models.py`, `test_record.py`, etc.).
2. Copy those test files to the new repo's `tests/unit/` tree, mirroring the source structure.
3. Verify they pass in the new repo (may need fixture/conftest adjustments for imports).
4. Delete them from ragdocs.
5. Keep in ragdocs: `test_file_discovery.py` (touches indexing manager), `test_daemon.py`, `test_mcp_tools.py`, `test_bootstrap_session.py`, etc.

**Acceptance**
- `andnp-searchkernel pytest tests/` passes (unit suite only).
- `mcp-markdown-ragdocs pytest tests/` passes (app suite only, no library unit tests).
- Combined test coverage still covers the library (from both repos' suites, together).

**Tests**
- Both suites green independently.

---

### X7 — First real PyPI release: tag `v0.1.0`, publish via trusted GitHub Actions

**Goal:** Release the library to PyPI, then flip `mcp-markdown-ragdocs` dependency from local path to published version.

**Depends on:** X6 (tests pass).

**Deliverables**
1. Tag `v0.1.0` in `andnp-searchkernel` (on main).
2. GitHub Actions triggered by tag: build wheel + sdist, publish to PyPI via trusted publisher (OIDC, no secrets in repo).
3. Verify `pip install andnp-searchkernel==0.1.0` installs from PyPI.
4. Update `mcp-markdown-ragdocs/pyproject.toml`: change `andnp-searchkernel = { path = ... }` to `andnp-searchkernel = ">=0.1.0,<1.0"`.
5. Test ragdocs still works with published version.

**Primary files**
- `andnp-searchkernel/.github/workflows/lint-test-publish.yml`: add publish step triggered by tags matching `v*`.
- `andnp-searchkernel/CHANGELOG.md`: document 0.1.0 release notes.
- `mcp-markdown-ragdocs/pyproject.toml`: switch dependency.

**Tasks**
1. Ensure CI in new repo is fully green (lint, tests, import-linter).
2. Create tag `v0.1.0` locally and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. GitHub Actions runs: build + publish.
4. Verify on PyPI: `https://pypi.org/project/andnp-searchkernel/0.1.0/`.
5. In ragdocs, update `pyproject.toml`, run `uv sync`, verify import still works: `python -c "from searchkernel import Record"`.
6. Run ragdocs unit tests one more time.

**Acceptance**
- `andnp-searchkernel==0.1.0` is on PyPI.
- `pip install andnp-searchkernel` downloads it.
- `mcp-markdown-ragdocs` works with the published version.
- No local-path fallback needed.

**Tests**
- CI passes in both repos.
- Manual `pip install` test.

---

### X8 — Prove extensibility: Google Drive ContentSource adapter

**Goal:** Implement a new `ContentSource` (Google Drive) in `mcp-markdown-ragdocs` WITHOUT touching the library, proving the port is truly extensible.

**Depends on:** X7 (library is published, app imports the library).

**Deliverables**
1. New file `mcp-markdown-ragdocs/searchkernel/adapters/sources/gdrive.py` implementing `ContentSource` protocol.
2. `GDriveSource` yields `Record`s for Google Drive documents (fetches via Google Drive API, yields structured records).
3. Wired into `ApplicationContext.content_sources` registry alongside git/markdown/plaintext.
4. Test: `test_gdrive_source.py` verifies records are yielded correctly.
5. **Zero changes to library or library-app wiring** (no new ports, no new stages, pure adapter).

**Primary files**
- new `mcp-markdown-ragdocs/searchkernel/adapters/sources/gdrive.py`.
- update `mcp-markdown-ragdocs/searchkernel/app/composition.py` to register the source.
- new `mcp-markdown-ragdocs/tests/unit/adapters/test_gdrive_source.py`.

**Tasks**
1. Design `GDriveSource` class mirroring `GitContentSource`: accept a folder ID, yield records.
2. Implement `iter_records(since)` to fetch documents modified after `since` timestamp.
3. Implement `change_signal()` to indicate polling or push-based updates.
4. Wire into the composition root.
5. Write tests (may be mocked if Google Drive API is not available in CI).

**Acceptance**
- `adapters/sources/gdrive.py` exists and implements `ContentSource`.
- No imports of `searchkernel.indexing`, `searchkernel.indices`, or any library private module.
- All imports are from `searchkernel.ports`, `searchkernel.domain`, or external (google-auth, etc.).
- Tests pass (with mocked API if needed).
- **This proves: adding a new content source requires only a new app-layer adapter, zero library changes.**

**Tests**
- Unit test of `GDriveSource.iter_records()` with mocked API.
- (Integration test of live Google Drive API can be optional / env-gated.)

---

## Milestones

- **M1:** X1 boundary finalized — library/app split is documented and reviewed.
- **M2:** X2–X3 complete — new repo scaffolded, LIBRARY code extracted with history.
- **M3:** X4–X5 complete — ragdocs cut over to external library dependency, imports working.
- **M4:** X6–X7 complete — tests split, library published to PyPI.
- **M5:** X8 complete — Google Drive adapter proves extensibility without library changes.

---

## Acceptance criteria (repo-level)

1. **`andnp-searchkernel` exists on PyPI:** `pip install andnp-searchkernel` works; library can be used standalone (no ragdocs runtime required).
2. **Library is extensible:** Adding a new `ContentSource` (Google Drive) requires only app-layer code; zero library edits.
3. **Independent releases:** Library and app ship on separate version schedules; each repo has independent CI.
4. **Boundary is enforced:** `import-linter` forbids app-owned code from reaching past the library's public surface (`searchkernel.__init__` exports).
5. **History is preserved:** `git log` in the new repo shows every commit that touched library code, with original commit messages and authorship.
6. **Tests are green:** Both repos have independent, passing test suites; CI is green for library + app.
7. **Documentation:** Library `README.md` explains how to build a search kernel; example shows a custom content source.
