"""Application service protocols and composition adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from searchkernel.api import Record

from mcp_markdown_ragdocs.app.search import ApplicationSearchUseCase

if TYPE_CHECKING:
    from mcp_markdown_ragdocs.context import ApplicationContext


class IndexingService(Protocol):
    """Operations needed by transports and worker coordination."""

    def load(self) -> None: ...
    def persist(self) -> None: ...
    def index_document(self, file_path: str) -> bool: ...
    def index_record(self, record: Record) -> bool: ...
    def get_document_count(self) -> int: ...
    def is_ready(self) -> bool: ...
    def task_target(self) -> Any: ...


class LifecycleService(Protocol):
    """Lifecycle operations exposed outside the composition root."""

    async def start(self, background_index: bool = False) -> None: ...
    async def stop(self) -> None: ...
    async def ensure_ready(self, timeout: float = 60.0) -> None: ...
    def is_ready(self) -> bool: ...


class ManagerIndexingService:
    """Expose record manager operations without leaking its implementation."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    def load(self) -> None:
        self._manager.load()

    def persist(self) -> None:
        self._manager.persist()

    def index_document(self, file_path: str) -> bool:
        return self._manager.index_document(file_path)

    def index_record(self, record: Record) -> bool:
        return self._manager.index_record(record)

    def get_document_count(self) -> int:
        return self._manager.get_document_count()

    def is_ready(self) -> bool:
        return self._manager.is_ready()

    def task_target(self) -> Any:
        return self._manager


class ContextLifecycleService:
    """Delegate lifecycle calls to the composed application context."""

    def __init__(self, context: LifecycleService) -> None:
        self._context = context

    async def start(self, background_index: bool = False) -> None:
        await self._context.start(background_index=background_index)

    async def stop(self) -> None:
        await self._context.stop()

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        await self._context.ensure_ready(timeout=timeout)

    def is_ready(self) -> bool:
        return self._context.is_ready()


class ApplicationServices:
    """Stable service bundle owned by the application composition root."""

    def __init__(
        self,
        *,
        search: ApplicationSearchUseCase,
        indexing: IndexingService,
        lifecycle: LifecycleService,
    ) -> None:
        self.search = search
        self.indexing = indexing
        self.lifecycle = lifecycle


def compose_services(
    context: ApplicationContext,
    *,
    manager: Any,
    search: ApplicationSearchUseCase,
) -> ApplicationServices:
    return ApplicationServices(
        search=search,
        indexing=ManagerIndexingService(manager),
        lifecycle=ContextLifecycleService(context),
    )


__all__ = [
    "ApplicationServices",
    "ContextLifecycleService",
    "IndexingService",
    "LifecycleService",
    "ManagerIndexingService",
    "compose_services",
]
