from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from huey.utils import Error

if TYPE_CHECKING:
    from huey import SqliteHuey


@dataclass(frozen=True)
class QueueFailure:
    task_id: str
    task_name: str | None
    error: str
    retries: int = 0
    traceback: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueueStats:
    pending_count: int
    scheduled_count: int
    running_count: int = 0
    failed_count: int = 0
    worker_running: bool = False
    backpressure_limit: int | None = None
    backpressure_utilization: float | None = None
    task_counts: dict[str, int] = field(default_factory=dict)
    recent_failures: list[QueueFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "pending_count": self.pending_count,
            "scheduled_count": self.scheduled_count,
            "running_count": self.running_count,
            "failed_count": self.failed_count,
            "worker_running": self.worker_running,
            "backpressure_limit": self.backpressure_limit,
            "backpressure_utilization": self.backpressure_utilization,
            "task_counts": self.task_counts,
            "recent_failures": [failure.to_dict() for failure in self.recent_failures],
        }


def get_queue_stats(
    huey: SqliteHuey,
    *,
    worker_running: bool = False,
    failure_limit: int = 10,
    backpressure_limit: int | None = None,
) -> QueueStats:
    task_counts = _collect_task_counts(huey)
    failures = _collect_failures(huey)
    pending_count = huey.pending_count()
    utilization = None
    if backpressure_limit is not None and backpressure_limit > 0:
        utilization = pending_count / backpressure_limit
    return QueueStats(
        pending_count=pending_count,
        scheduled_count=huey.scheduled_count(),
        running_count=0,
        failed_count=len(failures),
        worker_running=worker_running,
        backpressure_limit=backpressure_limit,
        backpressure_utilization=utilization,
        task_counts=task_counts,
        recent_failures=failures[-failure_limit:],
    )


def _collect_task_counts(huey: SqliteHuey) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_item in huey.storage.enqueued_items(limit=None):
        name = _decode_task_name(huey, raw_item)
        counts[name] = counts.get(name, 0) + 1
    for raw_item in huey.storage.scheduled_items(limit=None):
        name = _decode_task_name(huey, raw_item)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _collect_failures(huey: SqliteHuey) -> list[QueueFailure]:
    failures: list[QueueFailure] = []
    for task_id in huey.storage.result_items():
        failure = _decode_failure(huey, str(task_id))
        if failure is not None:
            failures.append(failure)
    return failures


def _decode_task_name(huey: SqliteHuey, raw_item: bytes) -> str:
    try:
        message = huey.serializer.deserialize(raw_item)
        raw_name = getattr(message, "name", None)
        if isinstance(raw_name, str) and raw_name:
            return raw_name.rsplit(".", 1)[-1]
    except Exception:
        pass
    return "unknown"


def _decode_failure(huey: SqliteHuey, task_id: str) -> QueueFailure | None:
    try:
        raw_item = huey.storage.peek_data(task_id)
    except Exception:
        return None
    if raw_item is None:
        return None

    try:
        payload = huey.serializer.deserialize(raw_item)
    except Exception:
        return None
    if not isinstance(payload, Error):
        return None

    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    return QueueFailure(
        task_id=str(metadata.get("task_id", task_id)),
        task_name=_normalize_failure_task_name(metadata.get("task_name")),
        error=str(metadata.get("error", "unknown error")),
        retries=int(metadata.get("retries", 0)),
        traceback=_coerce_optional_str(metadata.get("traceback")),
    )


def _normalize_failure_task_name(task_name: Any) -> str | None:
    if not isinstance(task_name, str) or not task_name:
        return None
    return task_name.rsplit(".", 1)[-1]


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
