from types import SimpleNamespace

import pytest

from mcp_markdown_ragdocs.app.services import (
    ContextLifecycleService,
    ManagerIndexingService,
)


def test_manager_indexing_service_hides_manager_implementation() -> None:
    manager = SimpleNamespace(
        load=lambda: "loaded",
        persist=lambda: "persisted",
        index_document=lambda path: path,
        index_record=lambda record: record,
        get_document_count=lambda: 3,
        is_ready=lambda: True,
    )
    service = ManagerIndexingService(manager)

    assert service.load() is None
    assert service.persist() is None
    assert service.index_document("doc.md") == "doc.md"
    assert service.get_document_count() == 3
    assert service.is_ready() is True


@pytest.mark.asyncio
async def test_context_lifecycle_service_delegates() -> None:
    calls = []

    class FakeContext:
        async def start(self, *, background_index):
            calls.append(("start", background_index))

        async def stop(self):
            calls.append(("stop",))

        async def ensure_ready(self, *, timeout):
            calls.append(("ready", timeout))

        def is_ready(self):
            return True

    service = ContextLifecycleService(FakeContext())
    await service.start(background_index=True)
    await service.ensure_ready(timeout=2)
    await service.stop()

    assert calls == [("start", True), ("ready", 2), ("stop",)]
    assert service.is_ready() is True
