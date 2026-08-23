"""Tests for daemon semantic-search startup warmup."""

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

    provider.embed_query.assert_called_once_with("__ragdocs_startup_warmup__")
    vector_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        1,
        model_name="test-model",
        dim=3,
    )


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
