"""Tests for daemon semantic-search startup warmup."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_markdown_ragdocs.context import ApplicationContext, ContextIndexingPort


@pytest.mark.asyncio
async def test_warmup_semantic_search_loads_vector_state_for_daemon() -> None:
    """
    Warmup embeds a sentinel query and performs one vector lookup.

    The lookup is the contract that forces a lazy FAISS state to load.
    """
    provider = SimpleNamespace(
        model_name="test-model",
        dim=3,
        embed_query=MagicMock(return_value=[0.1, 0.2, 0.3]),
    )
    events: list[str] = []
    vector_store = SimpleNamespace(
        search=MagicMock(side_effect=lambda *args, **kwargs: events.append("search")),
        migrate_legacy_persistence=MagicMock(
            side_effect=lambda *args, **kwargs: events.append("migration") or True
        ),
    )
    index_manager = MagicMock(spec=ContextIndexingPort)
    index_manager.embedding_provider = provider
    index_manager.vector = vector_store
    index_manager.kernel = SimpleNamespace(
        backend=SimpleNamespace(vector_epoch=MagicMock(return_value=1))
    )
    context = ApplicationContext(
        config=MagicMock(),
        index_manager=index_manager,
        orchestrator=MagicMock(),
        use_tasks=True,
    )

    await context.warmup_semantic_search()

    provider.embed_query.assert_called_once_with("__ragdocs_startup_warmup__")
    vector_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        1,
        model_name="test-model",
        dim=3,
    )
    vector_store.migrate_legacy_persistence.assert_called_once_with(
        "test-model", 3
    )
    assert events == ["search", "migration"]


@pytest.mark.asyncio
async def test_warmup_migration_false_is_nonfatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep semantic warmup successful when migration reports no change.

    A false result can mean that no applicable FAISS legacy artifact exists;
    the existing warmup and epoch readiness contract must still complete.
    """
    provider = SimpleNamespace(
        model_name="test-model",
        dim=3,
        embed_query=MagicMock(return_value=[0.1, 0.2, 0.3]),
    )
    vector_store = SimpleNamespace(
        search=MagicMock(return_value=[]),
        migrate_legacy_persistence=MagicMock(return_value=False),
    )
    index_manager = MagicMock(spec=ContextIndexingPort)
    index_manager.embedding_provider = provider
    index_manager.vector = vector_store
    index_manager.kernel = SimpleNamespace(
        backend=SimpleNamespace(vector_epoch=MagicMock(return_value=1))
    )
    context = ApplicationContext(
        config=MagicMock(),
        index_manager=index_manager,
        orchestrator=MagicMock(),
        use_tasks=True,
    )

    with caplog.at_level(logging.WARNING):
        await context.warmup_semantic_search()

    assert context.is_semantic_search_ready() is True
    assert "migration was not completed" in caplog.text


@pytest.mark.asyncio
async def test_warmup_migration_exception_is_nonfatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep semantic warmup successful when migration raises.

    Migration is an optional startup optimization; its failure must not turn a
    valid warmed search state into a readiness failure.
    """
    provider = SimpleNamespace(
        model_name="test-model",
        dim=3,
        embed_query=MagicMock(return_value=[0.1, 0.2, 0.3]),
    )
    vector_store = SimpleNamespace(
        search=MagicMock(return_value=[]),
        migrate_legacy_persistence=MagicMock(
            side_effect=RuntimeError("migration unavailable")
        ),
    )
    index_manager = MagicMock(spec=ContextIndexingPort)
    index_manager.embedding_provider = provider
    index_manager.vector = vector_store
    index_manager.kernel = SimpleNamespace(
        backend=SimpleNamespace(vector_epoch=MagicMock(return_value=1))
    )
    context = ApplicationContext(
        config=MagicMock(),
        index_manager=index_manager,
        orchestrator=MagicMock(),
        use_tasks=True,
    )

    with caplog.at_level(logging.WARNING):
        await context.warmup_semantic_search()

    assert context.is_semantic_search_ready() is True
    assert "migration failed" in caplog.text


@pytest.mark.asyncio
async def test_warmup_without_migration_capability_remains_compatible() -> None:
    """Support installed SearchKernel versions without the optional method.

    The existing one-shot search remains sufficient when the vector store does
    not expose explicit legacy migration yet.
    """
    provider = SimpleNamespace(
        model_name="test-model",
        dim=3,
        embed_query=MagicMock(return_value=[0.1, 0.2, 0.3]),
    )
    vector_store = SimpleNamespace(search=MagicMock(return_value=[]))
    index_manager = MagicMock(spec=ContextIndexingPort)
    index_manager.embedding_provider = provider
    index_manager.vector = vector_store
    index_manager.kernel = SimpleNamespace(
        backend=SimpleNamespace(vector_epoch=MagicMock(return_value=1))
    )
    context = ApplicationContext(
        config=MagicMock(),
        index_manager=index_manager,
        orchestrator=MagicMock(),
        use_tasks=True,
    )

    await context.warmup_semantic_search()

    vector_store.search.assert_called_once()
    assert context.is_semantic_search_ready() is True


@pytest.mark.asyncio
async def test_warmup_semantic_search_skips_one_shot_runtime() -> None:
    """
    One-shot runtimes do not pay the daemon-only warmup cost.

    Their normal query path remains responsible for loading semantic state.
    """
    provider = SimpleNamespace(embed_query=MagicMock())
    vector_store = SimpleNamespace(search=MagicMock())
    index_manager = MagicMock(spec=ContextIndexingPort)
    index_manager.embedding_provider = provider
    index_manager.vector = vector_store
    context = ApplicationContext(
        config=MagicMock(),
        index_manager=index_manager,
        orchestrator=MagicMock(),
        use_tasks=False,
    )

    await context.warmup_semantic_search()

    provider.embed_query.assert_not_called()
    vector_store.search.assert_not_called()
