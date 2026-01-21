# AI Context & Behavioral Guidelines

> **Purpose**: This file provides comprehensive context for AI agents (Claude, Gemini, ChatGPT) interacting with the `mcp-markdown-ragdocs` repository. It outlines the architectural mental model, core workflows, and coding standards to ensure high-quality, idiomatic contributions.

## 1. Project Mission
**mcp-markdown-ragdocs** is a local-first, privacy-focused RAG (Retrieval-Augmented Generation) server designed to make Markdown documentation repositories semantically searchable. It exposes its capabilities via the **Model Context Protocol (MCP)**, allowing AI assistants to query documentation and git history efficiently.

## 2. Architectural Mental Model

The system is composed of four distinct layers:

### A. Interface Layer
- **MCP Server** (`src/mcp_server.py`): The primary entry point. Handles tool registration (`query_documents`, `search_git_history`) and JSON-RPC communication.
- **REST Server** (`src/server.py`): An optional HTTP interface for standard API clients.

### B. Application Core
- **Context** (`src/context.py`): The singleton `ApplicationContext` holds the state of all indices, the configuration, and the background task manager.
- **Models** (`src/models.py`): Defines Pydantic models for `DocumentChunk`, `SearchResult`, and configuration objects.

### C. Indexing Engine
- **Watcher** (`src/indexing/watcher.py`): Debounced file system watcher that triggers updates.
- **Pipeline** (`src/indexing/manager.py`): Coordinates the flow: `Read -> Parse -> Chunk -> Index`.
- **Parsers** (`src/parsers/`): `MarkdownParser` (structure-aware) and `PlaintextParser`.
- **Chunkers** (`src/chunking/`): `HeaderChunker` preserves document hierarchy (parents/children).

### D. Search Orchestrator
- **Orchestrator** (`src/search/orchestrator.py`): The "brain" of the search.
  1. **Expansion**: Expands queries using synonyms/hypothetical questions (if enabled).
  2. **Parallel Search**: Queries Vector (FAISS), Keyword (Whoosh), and Graph (NetworkX) indices concurrently.
  3. **Fusion**: Merges results using Reciprocal Rank Fusion (RRF) and adaptive weights.
  4. **Post-Processing** (`src/search/pipeline.py`): Deduplication, MMR (Maximum Marginal Relevance) for diversity, and Re-ranking.

## 3. Key Workflows

### The Indexing Loop
1. `FileWatcher` detects a change in `docs/*.md`.
2. `IndexManager` receives the event.
3. `MarkdownParser` converts text to a structural tree.
4. `HeaderChunker` splits the tree into `DocumentChunk` objects (preserving headers as metadata).
5. Chunks are sent to:
   - `VectorIndex` (Embeddings)
   - `KeywordIndex` (BM25)
   - `GraphStore` (Links/Hierarchy)
6. `IndexState` is persisted to disk.

### The Search Loop
1. User invokes `query_documents(query="...")`.
2. `SearchOrchestrator` classifies the query (Broad vs. Specific).
3. Sub-queries are executed in parallel:
   - *Semantic*: Dense vector retrieval.
   - *Keyword*: Sparse BM25 retrieval for exact matches.
   - *Graph*: PageRank/Connectivity scoring.
4. `FusionRanker` combines scores: $Score = W_v \cdot V + W_k \cdot K + W_g \cdot G$.
5. Results are returned to the MCP client.

## 4. Coding Standards & Conventions

- **Asyncio**: The core is async-first. Use `async def` and `await` for all I/O bound operations (file reading, searching).
- **Typing**: Strict Python type hints are required. Use modern (python 3.13+) typing standards, and Pydantic models.
- **Error Handling**: specialized exceptions in `src/utils.py`. Never swallow errors silently; log them using the centralized logger.
- **Index Resilience**: All indices implement self-healing via `_reinitialize_after_corruption()`. On corruption detection, indices reinitialize to a clean state and return empty results (graceful degradation). Reconciliation rebuilds indices automatically.
- **Thread Safety**: VectorIndex uses fine-grained locking (`_index_lock`) to protect internal state during concurrent operations. Lock is held only during dict access + mutation, not during expensive I/O (embeddings, docstore). See [docs/specs/20-vector-index-thread-safety.md](docs/specs/20-vector-index-thread-safety.md).
- **Testing**:
  - `pytest` is the runner.
  - `tests/unit`: Fast, isolated tests.
  - `tests/integration`: Tests involving the real file system or multiple components.
  - `tests/e2e`: Full MCP server tests.
- **Configuration**: All config is managed via `pydantic-settings` in `src/config.py` and loaded from `pyproject.toml` or `config.toml`.

## 5. Critical Files Map
- `src/mcp_server.py`: **Start here** to add new tools.
- `src/search/orchestrator.py`: **Start here** to modify search logic/ranking.
- `src/indexing/manager.py`: **Start here** to change how files are processed.
- `src/git/commit_indexer.py`: **Start here** for git history features.
- `pyproject.toml`: Dependency and build management (uv/hatch).

## 6. Tool Usage Guidelines

### Document Search vs. Memory Management

**CRITICAL DISTINCTION**: This system provides TWO separate tool sets with different purposes:

#### A. Document Search Tools (Query the indexed documentation)
Use these to **search and read** existing project documentation:

- `query_documents`: Search documentation using hybrid search
- `query_unique_documents`: Search with one result per document
- `search_git_history`: Search commit history
- `search_with_hypothesis`: Search using HyDE technique

**Purpose**: Finding and retrieving information from the project's documentation corpus (README, specs, guides, etc.)
**When to use**: When you need to understand existing documentation, find API references, locate implementation details

#### B. Memory Management Tools (AI persistent storage)
Use these to **create, modify, and search AI memories** (persistent notes that survive across sessions):

- `create_memory`: Create a new memory file in the Memory Bank
- `read_memory`: Read a specific memory file
- `update_memory`: Replace memory content
- `append_memory`: Append to a memory file
- `delete_memory`: Delete a memory (moves to trash)
- `search_memories`: Search the Memory Bank
- `search_linked_memories`: Find memories linking to a document
- `get_memory_stats`: Get memory statistics
- `merge_memories`: Consolidate multiple memories

**Purpose**: Storing AI decisions, plans, observations, and cross-session knowledge
**When to use**: When recording decisions, creating plans, noting observations for future reference

**Storage locations:**
- Memories: `.memories/` (project) or `~/.local/share/mcp-markdown-ragdocs/memories/` (user)
- Documentation: Anywhere under the configured `documents_path` (typically project root)

### File System Operations

For **editing regular project files** (code, documentation, configuration):
- Use standard file system tools provided by your environment
- **DO NOT** use memory tools for editing project documentation files
- **DO NOT** use `create_memory` when asked to create/edit markdown documentation files

**Example scenarios:**
- "Add a section to README.md" → Use file editor, NOT `create_memory`
- "Create a new spec document" → Use file editor, NOT `create_memory`
- "Record this architectural decision" → Use `create_memory` with type="reflection"
- "Remember this plan for later" → Use `create_memory` with type="plan"

### Query vs. Content Distinction

**Query tools** (`query_documents`, `search_memories`) are for **finding existing content**:
- Returns search results with relevance scores
- Does not modify content
- Use when you need to locate information

**CRUD tools** (`create_memory`, `update_memory`) are for **managing content**:
- Create, read, update, delete operations
- Modify the file system
- Use when you need to persist information

## 7. Common Tasks (for AI)

**Task**: Add a new tool to the MCP server.
**Action**:
1. Define the input schema (Pydantic) in `src/mcp_server.py`.
2. Implement the handler method in `MCPServer`.
3. Register it in the `list_tools` method.

**Task**: Improve search quality.
**Action**:
1. Check `src/search/weights.py` (if it exists) or `orchestrator.py` for scoring constants.
2. Consider adjusting the RRF constant ($k=60$) in `src/search/fusion.py`.
