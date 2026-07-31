"""Recall proof: the LIVE IndexManager + SearchOrchestrator run end-to-end on
pgvector when `store.backend = "pgvector"`.

This is the acceptance test for the pgvector-cutover workstream: it goes
through `ApplicationContext.create()` (the composition root), not a
hand-wired IndexManager, so it proves `store.backend` genuinely drives the
live index/search path rather than being read only by a purpose-built test
double. Needs a live Postgres (SEARCHKERNEL_PG_DSN) and loads the real
HuggingFace embedding model, so it is marked slow + serial.
"""

import os

import pytest
from searchkernel.search.pipeline import SearchPipelineConfig

from ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
    StoreConfig,
)
from ragdocs.context import ApplicationContext

pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.fixture
def pg_dsn():
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")
    return dsn


@pytest.fixture
def test_config(tmp_path, pg_dsn):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()
    return Config(
        indexing=IndexingConfig(
            documents_path=str(docs_path), index_path=str(index_path)
        ),
        search=SearchConfig(),
        chunking=ChunkingConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        store=StoreConfig(backend="pgvector", pg_dsn=pg_dsn),
    )


@pytest.mark.asyncio
async def test_pgvector_backend_indexes_and_searches_live(test_config, monkeypatch):
    monkeypatch.setattr("ragdocs.context.load_config", lambda: test_config)
    ctx = ApplicationContext.create(
        project_override=None, enable_watcher=False, lazy_embeddings=True
    )

    from searchkernel.adapters.stores.pgvector_index import PGVectorIndex

    assert isinstance(ctx.index_manager.vector, PGVectorIndex)

    docs_path = ctx.documents_roots[0]
    (docs_path / "photosynthesis.md").write_text(
        "# Photosynthesis\n\n"
        "Plants convert sunlight, water, and carbon dioxide into glucose "
        "and oxygen through photosynthesis in their chloroplasts."
    )
    (docs_path / "database.md").write_text(
        "# Relational Databases\n\n"
        "PostgreSQL is an open-source relational database that stores data "
        "in tables and answers queries written in SQL."
    )
    (docs_path / "volcano.md").write_text(
        "# Volcanoes\n\n"
        "A volcano is a rupture in the crust of a planet that allows molten "
        "rock, ash, and gases to escape from a magma chamber below."
    )

    for file_path in ctx.discover_files():
        ctx.index_manager.index_document(file_path)

    assert set(ctx.index_manager.vector.get_document_ids()) == {
        "photosynthesis",
        "database",
        "volcano",
    }

    results, _stats, _strategy_stats = await ctx.orchestrator.query(
        "How do plants turn sunlight into energy?",
        top_k=3,
        pipeline_config=SearchPipelineConfig(reranking_enabled=False),
    )

    assert results
    assert results[0].record_id == "photosynthesis"
