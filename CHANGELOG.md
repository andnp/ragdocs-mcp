# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Search Infrastructure Overhaul (Spec 17):**
  - **Community Detection:** Louvain algorithm clusters documents by wikilink connectivity; co-community results receive configurable score boost (default 1.1×)
  - **Score-Aware Fusion:** Dynamic weight adjustment based on per-query score variance; low-variance strategies automatically down-weighted
  - **HyDE Search:** `search_with_hypothesis` MCP tool for hypothesis-driven document embeddings; improves retrieval for vague queries
  - **Edge Types:** Graph edges now carry semantic types (`links_to`, `implements`, `tests`, `related`)
  - New config section: `[search.advanced]` with `community_detection_enabled`, `community_boost_factor`, `dynamic_weights_enabled`, `variance_threshold`, `hyde_enabled`, `default_edge_type`
- **Memory Management System:** Persistent AI memory bank with CRUD operations, hybrid search, and cross-corpus linking
  - 9 MCP tools: `create_memory`, `read_memory`, `update_memory`, `append_memory`, `delete_memory`, `search_memories`, `search_linked_memories`, `get_memory_stats`, `merge_memories`
  - Ghost node pattern for cross-corpus graph traversal (`[[doc.md]]` creates `ghost:doc.md` node)
  - Memory-specific recency boost (configurable days/factor)
  - Dual storage strategies: `"project"` (`.memories/`) or `"user"` (`~/.local/share/`)
  - New config section: `[memory]` with `enabled`, `storage_strategy`, `recency_boost_days`, `recency_boost_factor`
- Query expansion via embeddings: `build_concept_vocabulary()` extracts terms during indexing, `expand_query()` finds top-3 nearest terms to query embedding for improved recall
- Cross-encoder re-ranking with lazy model loading (loaded on first `rerank()` call)
  - Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB, ~50ms/10 docs)
  - New config options: `rerank_enabled`, `rerank_model`, `rerank_top_n`
- Concept vocabulary persisted as `concept_vocabulary.json` with index
- Heading-weighted embeddings: chunks prepend `header_path` to content before embedding for improved semantic context
- Extended frontmatter extraction: title, description, summary, keywords, author, category, type, related fields
- BM25F field boosting in keyword index:
  - title (3.0x), headers (2.5x), keywords (2.5x), description (2.0x), tags (2.0x), aliases (1.5x), author (1.0x), category
  - MultifieldParser searches all TEXT fields
- Result filtering pipeline with `CompressionStats` tracking:
  - `min_confidence`: score threshold filtering (default: 0.0 = disabled)
  - `max_chunks_per_doc`: per-document chunk limit (default: 0 = disabled)
  - `dedup_enabled` / `dedup_similarity_threshold`: semantic deduplication via cosine similarity clustering

### Changed
- `QueryOrchestrator.query()` returns `tuple[list[ChunkResult], CompressionStats]`
- Processing pipeline order: normalize → threshold → doc limit → dedup → re-rank → top_n
- Graph index enhanced with `related` frontmatter field edges
- Vocabulary built during `persist()` after indexing completes
- `rebuild-index` CLI command now includes three phases:
  1. Document indexing with progress bar
  2. Git commit indexing (if enabled) with repository discovery and progress tracking
  3. Concept vocabulary building (if enabled) with term count display
- All phases non-fatal: failures logged but do not prevent subsequent phases
- Enhanced progress output with emoji indicators and detailed phase summaries

### Migration
- **Reindexing required** to build concept vocabulary. Run: `uv run mcp-markdown-ragdocs rebuild-index`

### Migration
- **Reindexing required** for schema changes. Run: `uv run mcp-markdown-ragdocs rebuild-index`

## [0.1.0] - 2025-12-22

### Added
- Initial implementation of MCP server for Markdown documentation
- Hybrid search combining semantic embeddings (FAISS), keyword search (Whoosh), and graph traversal (NetworkX)
- Reciprocal Rank Fusion (RRF) for multi-strategy result merging
- Recency bias for recently modified documents
- LLM synthesis using llama-index for answer generation
- Automatic file watching with debounced incremental indexing
- Index versioning with automatic rebuild on configuration changes
- Rich Markdown parsing with tree-sitter:
  - Frontmatter extraction
  - Wikilink resolution
  - Tag extraction
  - Transclusion support
- CLI commands:
  - `run`: Start MCP server with optional host/port overrides
  - `rebuild-index`: Force full index rebuild
  - `check-config`: Validate and display configuration
- FastAPI server with endpoints:
  - `POST /query_documents`: Query interface with LLM synthesis
  - `GET /health`: Health check endpoint
  - `GET /status`: Operational status with document count, queue size, and failed files
- Zero-configuration operation with sensible defaults
- TOML configuration support with cascading config file lookup
- Comprehensive test suite with unit, integration, and E2E tests
- Documentation:
  - Architecture overview
  - Configuration reference
  - Hybrid search algorithm details
  - Integration guide for VS Code MCP
  - Development guide

[unreleased]: https://github.com/yourusername/mcp-markdown-ragdocs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/mcp-markdown-ragdocs/releases/tag/v0.1.0
