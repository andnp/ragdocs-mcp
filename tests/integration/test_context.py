from pathlib import Path

import pytest

from src.config import Config, ChunkingConfig, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.context import ApplicationContext
from src.indexing.manifest import save_manifest, IndexManifest


@pytest.fixture
def test_config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(index_path),
        ),
        search=SearchConfig(),
        chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
    )


@pytest.fixture
def context_with_config(test_config, monkeypatch):
    monkeypatch.setattr("src.context.load_config", lambda: test_config)
    return ApplicationContext.create(
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )


def test_create_initializes_components(context_with_config):
    ctx = context_with_config
    assert ctx.config is not None
    assert ctx.index_manager is not None
    assert ctx.orchestrator is not None
    assert ctx.watcher is None  # disabled


def test_create_with_watcher_enabled(test_config, monkeypatch):
    monkeypatch.setattr("src.context.load_config", lambda: test_config)
    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=True,
        lazy_embeddings=True,
    )
    assert ctx.watcher is not None


def test_discover_files_returns_markdown_files(context_with_config):
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    (docs_path / "test1.md").write_text("# Test 1")
    (docs_path / "test2.md").write_text("# Test 2")
    (docs_path / "not_markdown.txt").write_text("Not markdown")

    files = ctx.discover_files()
    assert len(files) == 2
    assert all(f.endswith(".md") for f in files)


def test_discover_files_excludes_hidden_dirs(context_with_config):
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)

    (docs_path / "visible.md").write_text("# Visible")
    hidden_dir = docs_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.md").write_text("# Hidden")

    files = ctx.discover_files()
    assert len(files) == 1
    assert "visible.md" in files[0]


def test_build_manifest_creates_correct_structure(context_with_config):
    ctx = context_with_config
    manifest = ctx._build_manifest()

    assert manifest.spec_version == "1.0.0"
    assert manifest.embedding_model == ctx.config.llm.embedding_model
    assert manifest.parsers == ctx.config.parsers
    assert "strategy" in manifest.chunking_config


def test_check_and_rebuild_if_needed_returns_true_for_new_index(context_with_config):
    ctx = context_with_config
    needs_rebuild = ctx._check_and_rebuild_if_needed()
    assert needs_rebuild is True


def test_check_and_rebuild_if_needed_returns_false_for_existing_index(context_with_config):
    ctx = context_with_config
    ctx.index_path.mkdir(parents=True, exist_ok=True)

    manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model=ctx.config.llm.embedding_model,
        parsers=ctx.config.parsers,
        chunking_config={
            "strategy": ctx.config.chunking.strategy,
            "min_chunk_chars": ctx.config.chunking.min_chunk_chars,
            "max_chunk_chars": ctx.config.chunking.max_chunk_chars,
            "overlap_chars": ctx.config.chunking.overlap_chars,
        },
        indexed_files={},
    )
    save_manifest(ctx.index_path, manifest)

    needs_rebuild = ctx._check_and_rebuild_if_needed()
    assert needs_rebuild is False


@pytest.mark.asyncio
async def test_startup_indexes_documents(context_with_config):
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)
    (docs_path / "test.md").write_text("# Test\n\nSome content here for indexing.")

    await ctx.start(background_index=False)

    assert ctx.index_manager.get_document_count() == 1


@pytest.mark.asyncio
async def test_startup_loads_existing_index(context_with_config, shared_embedding_model):
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)
    (docs_path / "test.md").write_text("# Test\n\nSome content here for indexing.")

    await ctx.start(background_index=False)
    doc_count = ctx.index_manager.get_document_count()
    assert doc_count == 1

    ctx2 = ApplicationContext.create.__func__(
        ApplicationContext,
        project_override=None,
        enable_watcher=False,
        lazy_embeddings=True,
    )
    ctx2.config = ctx.config
    ctx2.index_path = ctx.index_path

    needs_rebuild = ctx2._check_and_rebuild_if_needed()
    assert needs_rebuild is False


@pytest.mark.asyncio
async def test_shutdown_persists_indices(context_with_config):
    ctx = context_with_config
    docs_path = Path(ctx.config.indexing.documents_path)
    (docs_path / "test.md").write_text("# Test\n\nContent for persistence test.")

    await ctx.start(background_index=False)
    await ctx.stop()

    vector_path = ctx.index_path / "vector"
    keyword_path = ctx.index_path / "keyword"
    graph_path = ctx.index_path / "graph"

    assert vector_path.exists()
    assert keyword_path.exists()
    assert graph_path.exists()


@pytest.mark.asyncio
async def test_startup_with_watcher_starts_watcher(test_config, monkeypatch):
    monkeypatch.setattr("src.context.load_config", lambda: test_config)
    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=True,
        lazy_embeddings=True,
    )

    await ctx.start(background_index=False)

    assert ctx.watcher is not None
    assert ctx.watcher._running is True

    await ctx.stop()
    assert ctx.watcher._running is False
