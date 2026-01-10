# Development Guide

This document describes how to set up a development environment, run tests, understand the code structure, and contribute to the project.

## Development Setup

### Prerequisites

- Python 3.13+
- uv package manager
- Git

### Clone and Install

```zsh
git clone https://github.com/yourusername/mcp-markdown-ragdocs.git
cd mcp-markdown-ragdocs
uv sync
```

This installs all dependencies including development tools (pytest, pyright, ruff).

### Install as Editable Package

```zsh
uv pip install -e .
```

This allows you to run `mcp-markdown-ragdocs` commands while actively developing.

## Running Tests

### Full Test Suite

```zsh
uv run pytest
```

### By Test Category

**Unit tests:**

```zsh
uv run pytest tests/unit/
```

**Integration tests:**

```zsh
uv run pytest tests/integration/
```

**End-to-end tests:**

```zsh
uv run pytest tests/e2e/
```

**Performance tests:**

```zsh
uv run pytest tests/performance/
```

### With Coverage

```zsh
uv run pytest --cov=src --cov-report=term-missing
```

### Specific Test File

```zsh
uv run pytest tests/unit/test_config.py
```

### Specific Test Function

```zsh
uv run pytest tests/unit/test_config.py::test_load_config_defaults
```

## Code Quality

### Linting

Check for lint errors:

```zsh
uv run ruff check .
```

Auto-fix errors:

```zsh
uv run ruff check --fix .
```

### Formatting

Format code:

```zsh
uv run ruff format .
```

### Type Checking

```zsh
uv run pyright
```

### Pre-Commit Checks

Run all checks before committing:

```zsh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Code Structure

### Directory Layout

```
src/
├── cli.py              # CLI commands (run, rebuild-index, check-config)
├── config.py           # Configuration loading and dataclasses
├── models.py           # Shared data models (Document)
├── server.py           # FastAPI application and endpoints
├── indexing/
│   ├── manager.py      # IndexManager (coordinates all indices)
│   ├── manifest.py     # Index versioning and rebuild logic
│   └── watcher.py      # File watching with debouncing
├── indices/
│   ├── graph.py        # NetworkX graph store
│   ├── keyword.py      # Whoosh keyword index
│   └── vector.py       # FAISS vector index
├── parsers/
│   ├── base.py         # DocumentParser protocol
│   ├── dispatcher.py   # ParserRegistry and dispatch logic
│   └── markdown.py     # MarkdownParser with tree-sitter
└── search/
    ├── fusion.py       # RRF fusion algorithm
    └── orchestrator.py # Query orchestration and synthesis
```

### Key Components

#### [src/server.py](../src/server.py)

FastAPI application with lifespan context manager. Entry point for HTTP API.

**Responsibilities:**
- Load configuration
- Initialize indices
- Check manifest and rebuild if needed
- Start file watcher
- Define API endpoints
- Shutdown cleanup

#### [src/indexing/manager.py](../src/indexing/manager.py)

Coordinates updates across vector, keyword, and graph indices.

**Key Methods:**
- `index_document(file_path)`: Parse and update all indices
- `remove_document(doc_id)`: Remove from all indices
- `persist()`: Save all indices to disk
- `load()`: Load existing indices from disk

#### [src/parsers/markdown.py](../src/parsers/markdown.py)

Markdown parser using tree-sitter for AST-based extraction.

**Extraction Logic:**
- YAML frontmatter (metadata, tags, aliases)
- Wikilinks `[[Note]]` and `[[Note|Display]]`
- Transclusions `![[Note]]`
- Inline tags `#tag`
- Content (excluding code blocks)

#### [src/search/orchestrator.py](../src/search/orchestrator.py)

Query orchestration with parallel searches and RRF fusion.

**Query Flow:**
1. Execute semantic, keyword searches in parallel
2. Apply 1-hop graph neighbor boosting
3. Compute RRF scores with weights
4. Apply recency bias
5. Return top-k document IDs

#### [src/search/fusion.py](../src/search/fusion.py)

Reciprocal Rank Fusion algorithm implementation.

**Core Function:**
```python
def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    weights: dict[str, float],
    k: int = 60
) -> list[str]:
    # Returns fused ranked list of document IDs
```

## Testing Strategy

### Test Categories

**Unit Tests:**
- Test individual components in isolation
- Use `tmp_path` fixtures for ephemeral storage
- Fast execution (< 1 second per test)
- Examples: config loading, parser extraction, fusion algorithm

**Integration Tests:**
- Test component interactions
- Use real indices (not mocks)
- Moderate execution time (1-10 seconds per test)
- Examples: multi-index updates, hybrid search, persistence

**End-to-End Tests:**
- Test complete workflows via HTTP API
- Use FastAPI TestClient
- Realistic test data
- Examples: server lifecycle, query flow, file watching

**Performance Tests:**
- Benchmark indexing speed and query latency
- Large corpus tests (100-1000 documents)
- Track performance regressions

### Fixture Architecture

**Ephemeral Fixtures (default):**
- Use pytest's `tmp_path` for isolated test storage
- Fast, no cleanup needed
- Preferred for unit tests

**Persistent Fixtures:**
- Session/module/function scoped fixtures with shared storage
- Used for integration tests testing persistence/loading
- See [tests/infrastructure/README.md](../tests/infrastructure/README.md)

**Example Unit Test:**

```python
def test_index_document(tmp_path):
    """Test indexing a single document."""
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
        )
    )
    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)

    doc_file = tmp_path / "docs" / "test.md"
    doc_file.parent.mkdir(parents=True)
    doc_file.write_text("# Test\nContent here")

    manager.index_document(str(doc_file))
    assert manager.get_document_count() == 1
```

**Example Integration Test:**

```python
@pytest.mark.asyncio
async def test_hybrid_search_fusion(tmp_path):
    """Test RRF fusion across semantic, keyword, graph search."""
    config = create_test_config(tmp_path)
    manager = create_test_manager(config)
    orchestrator = QueryOrchestrator(
        manager._vector, manager._keyword, manager._graph, config, manager
    )

    # Create test corpus with links
    create_test_documents(tmp_path)

    # Query should combine all strategies
    results = await orchestrator.query("authentication", top_k=5)

    assert len(results) > 0
    assert "authentication" in results[0]  # Keyword match
    # Verify graph traversal added linked docs
```

## Adding New Features

### Adding a New Parser

The server supports pluggable parsers. Two parsers are included out-of-the-box:
- `MarkdownParser`: Full Markdown support with tree-sitter AST parsing, frontmatter, wikilinks, tags
- `PlainTextParser`: Plain text (.txt) files with UTF-8 and fallback encoding support

To add a new parser:

1. **Create parser class** in `src/parsers/`:

```python
# src/parsers/csv_parser.py
from pathlib import Path
from datetime import datetime, timezone
from src.models import Document
from src.parsers.base import DocumentParser

class CSVParser(DocumentParser):
    def parse(self, file_path: str):
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        return Document(
            id=path.stem,
            content=content,
            metadata={"source": str(path)},
            links=[],
            tags=[],
            file_path=str(path),
            modified_time=modified_time,
        )
```

2. **Register parser** in `src/parsers/__init__.py`:

```python
from src.parsers.csv_parser import CSVParser

__all__ = ["MarkdownParser", "PlainTextParser", "CSVParser"]
```

3. **Update dispatcher** in `src/parsers/dispatcher.py`:

Add to `_instantiate_parser()` function:

```python
def _instantiate_parser(parser_name: str, file_path: str):
    if parser_name == "MarkdownParser":
        return MarkdownParser()
    elif parser_name == "PlainTextParser":
        return PlainTextParser()
    elif parser_name == "CSVParser":
        return CSVParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name} for {file_path}")
```

4. **Configure pattern** in `config.toml`:

```toml
[parsers]
"**/*.md" = "MarkdownParser"
"**/*.txt" = "PlainTextParser"
"**/*.csv" = "CSVParser"
```

5. **Add tests** in `tests/unit/test_csv_parser.py`:

```python
def test_csv_parser_basic(tmp_path):
    parser = CSVParser()
    file_path = tmp_path / "test.csv"
    file_path.write_text("col1,col2\nval1,val2")

    doc = parser.parse(str(file_path))
    assert doc.content == "col1,col2\nval1,val2"
    assert doc.id == "test"
```

**Notes:**
- PlainTextParser handles encoding fallback gracefully (UTF-8 → latin-1 → cp1252 → iso-8859-1)
- Plain text files use paragraph-based chunking (reuses `HeaderBasedChunker._chunk_plain_text()`)
- See [specs/13-txt-file-chunking.md](../specs/13-txt-file-chunking.md) for implementation details

### Adding a New Search Strategy

1. **Implement strategy** in new index class or orchestrator method
2. **Add to parallel search** in `QueryOrchestrator.query()`
3. **Add weight configuration** in `SearchConfig`
4. **Update RRF fusion** to include new ranked list
5. **Add integration tests** for new strategy
6. **Document** in [docs/hybrid-search.md](hybrid-search.md)

### Adding a New CLI Command

1. **Add command** in `src/cli.py`:

```python
@cli.command("stats")
@click.option("--project", default=None, help="Override project detection")
def stats_cmd(project: str | None):
    """Display index statistics."""
    config = load_config()
    config = _apply_project_detection(config, project)
    # Implementation here
    click.echo(f"Total documents: {count}")
```

2. **Add tests** in `tests/e2e/test_cli.py`:

```python
def test_cli_stats(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "Total documents:" in result.output
```

3. **Update documentation** in `docs/configuration.md` CLI Commands section.

## Manual Testing

### Testing MCP Server (stdio)

Start the MCP server manually:

```zsh
uv run mcp-markdown-ragdocs mcp
```

The server will start and wait for MCP protocol messages on stdin. Useful for:
- Verifying server startup without errors
- Testing with manual JSON-RPC messages
- Debugging stdio transport issues

Press Ctrl+C to stop.

### Testing HTTP Server

Start the HTTP server:

```zsh
uv run mcp-markdown-ragdocs run
```

Query the API:

```zsh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query_documents \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_n": 3}'
```

### Testing CLI Query

Run a direct query:

```zsh
uv run mcp-markdown-ragdocs query "authentication"
```

With JSON output for parsing:

```zsh
uv run mcp-markdown-ragdocs query "test" --json | jq
```

## Testing Patterns

### Anti-Pattern: Mocks

This project avoids mocks in favor of real implementations:

**Bad:**

```python
def test_index_document_mock():
    mock_vector = Mock()
    mock_keyword = Mock()
    manager = IndexManager(config, mock_vector, mock_keyword, graph)
    manager.index_document("test.md")
    mock_vector.add.assert_called_once()
```

**Good:**

```python
def test_index_document_real(tmp_path):
    vector = VectorIndex()  # Real instance
    keyword = KeywordIndex()
    manager = IndexManager(config, vector, keyword, graph)
    manager.index_document(str(test_file))
    assert manager.get_document_count() == 1
```

### Pattern: Realistic Test Data

Use realistic document content in tests:

```python
markdown_content = """---
title: Authentication Guide
tags: [security, api]
aliases: [auth, credentials]
---

# Authentication

This guide covers authentication methods:

- [[OAuth 2.0]]
- [[API Keys]]

See also: [[Security Best Practices]]
"""
```

### Pattern: Fixture Reuse

Extract common setup to fixtures:

```python
@pytest.fixture
def test_corpus(tmp_path):
    """Create realistic documentation corpus."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    (docs_path / "auth.md").write_text("# Auth\n...")
    (docs_path / "api.md").write_text("# API\n...")

    return docs_path

def test_index_corpus(tmp_path, test_corpus):
    # Use test_corpus fixture
    config = Config(indexing=IndexingConfig(documents_path=str(test_corpus)))
    # ...
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Index Contents

```python
# In pytest with breakpoint
def test_debug_index(tmp_path):
    manager = create_manager(tmp_path)
    manager.index_document("test.md")

    # Inspect vector index
    import pdb; pdb.set_trace()
    results = manager._vector.search("query", top_k=10)
```

### Verify Manifest

```zsh
cat .index_data/index.manifest.json | jq
```

### Check Index Files

```zsh
ls -lh .index_data/
ls -lh .index_data/vector/
ls -lh .index_data/keyword/
```

## Contributing Guidelines

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-parser`
3. Make changes with tests
4. Run quality checks: `ruff check`, `pyright`, `pytest`
5. Commit with clear messages
6. Push and open a pull request

### Commit Messages

Use conventional commit format:

```
feat: add PlainTextParser for .txt files
fix: handle empty frontmatter in MarkdownParser
docs: update configuration reference
test: add integration test for graph traversal
```

### Pull Request Checklist

- [ ] Tests pass: `pytest`
- [ ] Lint clean: `ruff check`
- [ ] Type clean: `pyright`
- [ ] Format clean: `ruff format --check`
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG updated (if user-facing change)

### Code Style

- Follow PEP 8 (enforced by ruff)
- Type hints on all function parameters
- Docstrings for public APIs (Google style)
- Prefer composition over inheritance
- Keep functions focused (single responsibility)

### Testing Requirements

- New features require tests (unit + integration)
- Bug fixes require regression tests
- Tests must be deterministic (no random failures)
- Use realistic test data
- Avoid mocks when possible

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Tag release: `git tag v0.2.0`
5. Push tag: `git push --tags`
6. Build package: `uv build`
7. Publish: `uv publish` (if configured)

## Architecture Decision Records

Significant architecture decisions are documented in the `specs/` directory:

- [specs/02-architecture-and-tech-stack.md](../specs/02-architecture-and-tech-stack.md): System architecture and technology choices
- [specs/10-multi-project-support.md](../specs/10-multi-project-support.md): Project detection and index isolation
- [specs/11-search-quality-improvements.md](../specs/11-search-quality-improvements.md): Search pipeline enhancements (BM25F, dedup, re-ranking)
- [specs/12-context-compression.md](../specs/12-context-compression.md): Compression strategy decision (threshold + dedup)

Each spec includes:
- Executive summary
- Current state analysis
- Decision matrix with alternatives
- Implementation details
- Risk register
