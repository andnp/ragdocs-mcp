import pytest
from searchkernel.domain import Record

from mcp_markdown_ragdocs.app.services import (
    ContextLifecycleService,
    ManagerIndexingService,
)


def test_manager_indexing_service_hides_manager_implementation() -> None:
    """The service delegates only the manager operations in its application port."""
    class FakeManager:
        def load(self) -> None:
            return None

        def persist(self) -> None:
            return None

        def index_document(self, file_path: str) -> bool:
            return file_path == "doc.md"

        def index_record(self, record: Record) -> bool:
            return record.source_id == "record"

        def get_document_count(self) -> int:
            return 3

        def is_ready(self) -> bool:
            return True

    manager = FakeManager()
    service = ManagerIndexingService(manager)

    assert service.load() is None
    assert service.persist() is None
    assert service.index_document("doc.md") is True
    assert service.get_document_count() == 3
    assert service.is_ready() is True
    assert service.task_target() is manager


@pytest.mark.asyncio
async def test_context_lifecycle_service_delegates() -> None:
    calls = []

    class FakeContext:
        async def start(self, background_index=False):
            calls.append(("start", background_index))

        async def stop(self):
            calls.append(("stop",))

        async def ensure_ready(self, timeout=60.0):
            calls.append(("ready", timeout))

        def is_ready(self):
            return True

    service = ContextLifecycleService(FakeContext())
    await service.start(background_index=True)
    await service.ensure_ready(timeout=2)
    await service.stop()

    assert calls == [("start", True), ("ready", 2), ("stop",)]
    assert service.is_ready() is True
