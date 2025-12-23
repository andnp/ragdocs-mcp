# Contributing Guide

This guide covers everything needed to contribute to mcp-markdown-ragdocs.

## Quick Start

```zsh
git clone https://github.com/yourusername/mcp-markdown-ragdocs.git
cd mcp-markdown-ragdocs
uv sync
uv pip install -e .
```

Prerequisites: Python 3.13+, uv package manager.

## Development Setup

### Install Dependencies

```zsh
uv sync
```

Installs all runtime and development dependencies (pytest, pyright, ruff).

### Install as Editable Package

```zsh
uv pip install -e .
```

Allows running `mcp-markdown-ragdocs` commands while actively developing.

## Running Tests

Run all tests:

```zsh
uv run pytest
```

By category:

```zsh
uv run pytest tests/unit/           # Unit tests (fast)
uv run pytest tests/integration/    # Integration tests
uv run pytest tests/e2e/           # End-to-end tests
uv run pytest tests/performance/   # Performance benchmarks
```

Specific test file or function:

```zsh
uv run pytest tests/unit/test_config.py
uv run pytest tests/unit/test_config.py::test_load_config_defaults
```

With coverage:

```zsh
uv run pytest --cov=src --cov-report=term-missing
```

## Code Quality

### Linting

Check for lint errors:

```zsh
uv run ruff check .
```

Auto-fix where possible:

```zsh
uv run ruff check --fix .
```

### Type Checking

```zsh
uv run pyright
```

### Formatting

Check formatting:

```zsh
uv run ruff format --check .
```

Format code:

```zsh
uv run ruff format .
```

## Code Style

- Follow PEP 8 (enforced by ruff)
- Type hints on all function parameters
- Prefer return type inference over explicit annotations
- Docstrings on public APIs (Google style)
- Keep functions focused (single responsibility)
- Prefer composition over inheritance

## Commit Message Conventions

Use conventional commit format:

```
feat: add PlainTextParser for .txt files
fix: handle empty frontmatter in MarkdownParser
docs: update configuration reference
test: add integration test for graph traversal
refactor: extract query orchestration to separate module
perf: optimize keyword index rebuilds
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

## Pull Request Process

### Before Submitting

Run all quality checks:

```zsh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All must pass with zero errors.

### PR Checklist

- [ ] Tests pass: `pytest`
- [ ] Lint clean: `ruff check`
- [ ] Type clean: `pyright`
- [ ] Format clean: `ruff format --check`
- [ ] Tests added for new features
- [ ] Regression tests added for bug fixes
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG updated (if user-facing change)

### PR Description

Include:

- Summary of changes (what and why)
- Related issues (fixes #123)
- Testing approach
- Breaking changes (if any)

## Testing Requirements

### Test Coverage

- New features require tests (unit + integration)
- Bug fixes require regression tests
- Tests must be deterministic (no random failures)
- Use realistic test data

### Anti-Pattern: Avoid Mocks

Prefer real implementations over mocks:

**Bad:**

```python
def test_with_mock():
    mock_vector = Mock()
    manager = IndexManager(config, mock_vector, keyword, graph)
    mock_vector.add.assert_called_once()
```

**Good:**

```python
def test_with_real_index(tmp_path):
    vector = VectorIndex()  # Real instance
    manager = IndexManager(config, vector, keyword, graph)
    assert manager.get_document_count() == 1
```

### Fixture Patterns

Use pytest's `tmp_path` for isolated test storage:

```python
def test_index_document(tmp_path):
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "indices"),
        )
    )
    # Test implementation
```

See [docs/development.md](docs/development.md) for comprehensive testing patterns.

## Project Structure

```
src/
├── cli.py              # CLI commands
├── config.py           # Configuration dataclasses
├── models.py           # Shared data models
├── server.py           # FastAPI application
├── indexing/           # Index coordination
├── indices/            # Vector, keyword, graph indices
├── parsers/            # Document parsers
└── search/             # Query orchestration and fusion
```

Key components:

- **[src/server.py](src/server.py)**: FastAPI app with lifespan management
- **[src/indexing/manager.py](src/indexing/manager.py)**: Coordinates all indices
- **[src/parsers/markdown.py](src/parsers/markdown.py)**: Tree-sitter markdown parser
- **[src/search/orchestrator.py](src/search/orchestrator.py)**: Query execution and RRF fusion
- **[src/search/fusion.py](src/search/fusion.py)**: Reciprocal Rank Fusion algorithm

## Adding Features

### Adding a New Parser

1. Create parser in `src/parsers/`:

```python
# src/parsers/plaintext.py
from pathlib import Path
from src.models import Document

class PlainTextParser:
    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return Document(
            content=content,
            metadata={},
            links=[],
            tags=[],
            doc_id=path.stem,
            modified_time=path.stat().st_mtime,
        )
```

2. Register in `src/parsers/__init__.py`

3. Add pattern to `config.toml`:

```toml
[parsers]
"**/*.txt" = "PlainTextParser"
```

4. Add tests in `tests/unit/test_parsers.py`

### Adding a New CLI Command

1. Add command in `src/cli.py`:

```python
@cli.command("stats")
def stats_cmd():
    """Display index statistics."""
    config = load_config()
    # Implementation
```

2. Add tests in `tests/e2e/test_cli.py`

See [docs/development.md](docs/development.md) for detailed examples.

## Documentation

Update documentation when:

- Adding new features
- Changing configuration options
- Modifying API behavior
- Adding new parsers or search strategies

Documentation files:

- [README.md](README.md): Overview and quick start
- [docs/architecture.md](docs/architecture.md): System design
- [docs/configuration.md](docs/configuration.md): Config reference
- [docs/integration.md](docs/integration.md): VS Code and Claude setup
- [docs/hybrid-search.md](docs/hybrid-search.md): Search strategy details
- [docs/development.md](docs/development.md): Development guide

## Getting Help

- **Issues**: Check [GitHub issues](https://github.com/yourusername/mcp-markdown-ragdocs/issues) for known problems
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Read [docs/](docs/) for comprehensive guides
- **Code Examples**: See tests/ for usage patterns

## Release Process

Maintainers only:

1. Update version in `pyproject.toml`
2. Update [CHANGELOG.md](CHANGELOG.md)
3. Run full test suite
4. Tag release: `git tag v0.2.0`
5. Push tag: `git push --tags`
6. Build: `uv build`
7. Publish: `uv publish` (if configured)

## Code of Conduct

Be respectful and constructive. Focus on technical merit. Welcome newcomers.
