"""Generic task-submission result types.

Pure data classes describing the outcome of enqueuing work onto a task
queue. No huey/queue coupling — reusable by any indexing system that needs
to report enqueue/backpressure/already-pending outcomes to its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TaskSubmissionResult:
    status: Literal["enqueued", "already_pending", "backpressured", "unavailable"]

    @property
    def accepted_by_queue(self) -> bool:
        return self.status in {"enqueued", "already_pending"}

    @property
    def should_retry_later(self) -> bool:
        return self.status == "backpressured"

    @property
    def queue_available(self) -> bool:
        return self.status != "unavailable"

    @property
    def enqueued(self) -> bool:
        return self.status == "enqueued"


@dataclass(frozen=True)
class TaskBatchSubmissionResult:
    queue_available: bool
    requested_unique_count: int
    enqueued_count: int
    already_pending_count: int = 0
    backpressured_items: tuple[str, ...] = ()

    @property
    def backpressured_count(self) -> int:
        return len(self.backpressured_items)

    @property
    def should_retry_later(self) -> bool:
        return bool(self.backpressured_items)

    @property
    def all_represented(self) -> bool:
        if not self.queue_available:
            return False
        if self.should_retry_later:
            return False
        return (
            self.enqueued_count + self.already_pending_count
            >= self.requested_unique_count
        )
