"""Instance-owned task runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol

from mcp_markdown_ragdocs.coordination.queue import QueueRuntime
from mcp_markdown_ragdocs.coordination.task_submission import TaskSubmissionPort
from mcp_markdown_ragdocs.coordination.task_leases import TaskLeasePort
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentPort


class RegisteredTaskFunction(Protocol):
    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class RegisteredTaskHandle(Protocol):
    func: RegisteredTaskFunction

    def __call__(self, *args: object, **kwargs: object) -> object: ...


@dataclass
class TaskRuntime:
    """Own one queue, its submission port, and registered task handles."""

    queue_runtime: QueueRuntime
    submission: TaskSubmissionPort
    task_handles: Mapping[str, RegisteredTaskHandle] = field(default_factory=dict)
    gdrive_task_handles: Mapping[str, object] = field(default_factory=dict)
    index_manager: Any | None = field(default=None, repr=False)
    task_backpressure_limit: int = field(default=100, repr=False)
    bootstrap_index_path: Path | None = field(default=None, repr=False)
    bootstrap_documents_roots: list[Path] = field(default_factory=list, repr=False)
    schedule_vocabulary_catch_up: Callable[[], bool] | None = field(
        default=None, repr=False
    )
    task_lease_store: TaskLeasePort | None = field(default=None, repr=False)
    work_intent_store: WorkIntentPort | None = field(default=None, repr=False)
