"""Instance-owned task runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
import time
from typing import TYPE_CHECKING, Callable, Protocol

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeasePort
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentPort
from mcp_markdown_ragdocs.config import Config

if TYPE_CHECKING:
    from searchkernel.api import Record


class RegisteredTaskFunction(Protocol):
    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class RegisteredTaskHandle(Protocol):
    func: RegisteredTaskFunction

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class TaskIndexManager(Protocol):
    """Indexing capabilities used directly by task execution."""

    @property
    def index_path(self) -> Path: ...

    def index_document(self, file_path: str, force: bool = False) -> bool: ...

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ) -> None: ...

    def index_record(self, record: Record) -> bool: ...

    def index_records(self, records: Sequence[Record]) -> bool: ...

    def remove_document(self, doc_id: str) -> None: ...

    def remove_documents(
        self, doc_ids: list[str], persist: bool = False
    ) -> None: ...

    def persist(self) -> None: ...


class TaskEmbeddingCachePort(Protocol):
    """Embedding-cache maintenance exposed to indexing tasks."""

    def prune_to_active_records(self) -> int: ...

    def metrics(self) -> Mapping[str, int]: ...


@dataclass
class TaskRuntime:
    """Own one queue, its submission port, and registered task handles."""

    queue_runtime: QueueRuntime
    submission: TaskSubmissionPort
    task_handles: Mapping[str, RegisteredTaskHandle] = field(default_factory=dict)
    gdrive_task_handles: Mapping[str, object] = field(default_factory=dict)
    config: Config = field(kw_only=True)
    index_manager: TaskIndexManager | None = field(default=None, repr=False)
    embedding_cache: TaskEmbeddingCachePort | None = field(default=None, repr=False)
    task_backpressure_limit: int = field(default=100, repr=False)
    embedding_cache_prune_cooldown_seconds: int = field(default=86400, repr=False)
    time_provider: Callable[[], float] = field(default=time.time, repr=False)
    bootstrap_index_path: Path | None = field(default=None, repr=False)
    bootstrap_documents_roots: list[Path] = field(default_factory=list, repr=False)
    schedule_vocabulary_catch_up: Callable[[], bool] | None = field(
        default=None, repr=False
    )
    task_lease_store: TaskLeasePort | None = field(default=None, repr=False)
    work_intent_store: WorkIntentPort | None = field(default=None, repr=False)
    git_refresh_in_flight: set[str] = field(default_factory=set, repr=False)
    git_refresh_pending: set[str] = field(default_factory=set, repr=False)
    git_refresh_deferred: set[str] = field(default_factory=set, repr=False)
    git_refresh_lock: Lock = field(default_factory=Lock, repr=False)
