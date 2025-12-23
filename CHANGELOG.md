# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
