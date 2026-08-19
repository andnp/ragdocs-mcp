from pathlib import Path

import pytest

import mcp_markdown_ragdocs.indexing.record_manager as record_manager_module
from tests.conftest import create_test_document


@pytest.mark.asyncio
async def test_async_index_document_avoids_sync_bridge(
    record_manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async document indexing stays on the caller's event-loop path."""
    document = create_test_document(
        tmp_path / "docs",
        "async-document",
        "# Async document\n\nIndexed without a sync bridge.",
    )

    def fail_if_called(_awaitable):
        raise AssertionError("async indexing must not use the sync bridge")

    monkeypatch.setattr(record_manager_module, "_run_async", fail_if_called)

    assert await record_manager.async_index_document(document, update_graph=False)
    assert await record_manager.async_index_document(
        document,
        force=True,
        update_graph=False,
    )
