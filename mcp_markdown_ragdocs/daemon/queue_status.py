from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from huey.utils import Error

from mcp_markdown_ragdocs.coordination.task_leases import TaskLeaseStore

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


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
class QueueTaskSummary:
    task_id: str
    task_name: str | None
    state: str
    source_queue: str
    eta: str | None = None
    priority: int | None = None
    retries: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueueStats:
    pending_count: int
    scheduled_count: int
    running_count: int = 0
    failed_count: int = 0
    historical_failure_count: int = 0
    worker_running: bool = False
    backpressure_limit: int | None = None
    backpressure_utilization: float | None = None
    task_counts: dict[str, int] = field(default_factory=dict)
    recent_failures: list[QueueFailure] = field(default_factory=list)
    pending_tasks: list[QueueTaskSummary] = field(default_factory=list)
    scheduled_tasks: list[QueueTaskSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "pending_count": self.pending_count,
            "scheduled_count": self.scheduled_count,
            "running_count": self.running_count,
            "failed_count": self.failed_count,
            "historical_failure_count": self.historical_failure_count,
            "worker_running": self.worker_running,
            "backpressure_limit": self.backpressure_limit,
            "backpressure_utilization": self.backpressure_utilization,
            "task_counts": self.task_counts,
            "recent_failures": [failure.to_dict() for failure in self.recent_failures],
            "pending_tasks": [task.to_dict() for task in self.pending_tasks],
            "scheduled_tasks": [task.to_dict() for task in self.scheduled_tasks],
        }


@dataclass(frozen=True)
class QueuePurgeResult:
    purged_state: str
    purged_counts: dict[str, int]
    queue_stats: QueueStats

    def to_dict(self) -> dict[str, object]:
        payload = self.queue_stats.to_dict()
        payload["purged_state"] = self.purged_state
        payload["purged_counts"] = dict(self.purged_counts)
        return payload


def get_queue_stats(
    huey: SqliteHuey,
    *,
    worker_running: bool = False,
    failure_limit: int = 10,
    backpressure_limit: int | None = None,
) -> QueueStats:
    task_counts = _collect_task_counts(huey)
    failures = _collect_failures(huey)
    lease_store = TaskLeaseStore(cast(Any, huey.storage).filename)
    pending_count = huey.pending_count()
    utilization = None
    if backpressure_limit is not None and backpressure_limit > 0:
        utilization = pending_count / backpressure_limit

    pending_tasks = _collect_task_summaries(
        huey,
        huey.storage.enqueued_items(limit=None),
        state="pending",
        source_queue="pending",
    )
    scheduled_tasks = _collect_task_summaries(
        huey,
        huey.storage.scheduled_items(limit=None),
        state="scheduled",
        source_queue="scheduled",
    )

    return QueueStats(
        pending_count=pending_count,
        scheduled_count=huey.scheduled_count(),
        running_count=lease_store.active_count(),
        failed_count=len(failures),
        historical_failure_count=len(failures),
        worker_running=worker_running,
        backpressure_limit=backpressure_limit,
        backpressure_utilization=utilization,
        task_counts=task_counts,
        recent_failures=failures[-failure_limit:],
        pending_tasks=pending_tasks,
        scheduled_tasks=scheduled_tasks,
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
    # result_items() already performs one bulk read of every kv row; reuse
    # the values it returns instead of re-querying each key individually
    # with peek_data (an N+1 that dominated queue-status latency).
    failures: list[QueueFailure] = []
    for task_id, raw_item in huey.storage.result_items().items():
        failure = _decode_failure(huey, str(task_id), raw_item)
        if failure is not None:
            failures.append(failure)
    return failures


def _collect_task_summaries(
    huey: SqliteHuey,
    raw_items: list[bytes],
    *,
    state: str,
    source_queue: str,
) -> list[QueueTaskSummary]:
    summaries: list[QueueTaskSummary] = []
    for raw_item in raw_items:
        summary = _decode_task_summary(
            huey,
            raw_item,
            state=state,
            source_queue=source_queue,
        )
        if summary is not None:
            summaries.append(summary)
    return summaries


def _decode_task_name(huey: SqliteHuey, raw_item: bytes) -> str:
    try:
        message = huey.serializer.deserialize(raw_item)
        task_name = _normalize_task_name(getattr(message, "name", None))
        if task_name is not None:
            return task_name
    except Exception:
        logger.debug("Failed to decode task name", exc_info=True)
    return "unknown"


def _decode_task_summary(
    huey: SqliteHuey,
    raw_item: bytes,
    *,
    state: str,
    source_queue: str,
) -> QueueTaskSummary | None:
    try:
        message = huey.serializer.deserialize(raw_item)
    except Exception:  # noqa: BLE001 -- serializer backend errors vary; deserializing untrusted queue data
        return None

    task_id = _coerce_optional_str(getattr(message, "id", None))
    if task_id is None:
        return None

    return QueueTaskSummary(
        task_id=task_id,
        task_name=_normalize_task_name(getattr(message, "name", None)),
        state=state,
        source_queue=source_queue,
        eta=_serialize_optional_datetime(getattr(message, "eta", None)),
        priority=_coerce_optional_int(getattr(message, "priority", None)),
        retries=_coerce_optional_int(getattr(message, "retries", None), default=0) or 0,
    )


def _decode_failure(
    huey: SqliteHuey, task_id: str, raw_item: Any
) -> QueueFailure | None:
    if raw_item is None:
        return None

    try:
        payload = huey.serializer.deserialize(raw_item)
    except Exception:  # noqa: BLE001 -- serializer backend errors vary; deserializing untrusted queue data
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


def _normalize_task_name(task_name: Any) -> str | None:
    if not isinstance(task_name, str) or not task_name:
        return None
    return task_name.rsplit(".", 1)[-1]


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _coerce_optional_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _serialize_optional_datetime(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    return value.isoformat()


def purge_queue_state(
    huey: SqliteHuey,
    *,
    state: str,
    worker_running: bool = False,
    backpressure_limit: int | None = None,
) -> QueuePurgeResult:
    normalized_state = state.lower()
    if normalized_state not in {"pending", "scheduled", "failed", "all"}:
        raise ValueError(
            "state must be one of: pending, scheduled, failed, all"
        )

    purged_counts = {
        "pending": 0,
        "scheduled": 0,
        "failed": 0,
    }

    if normalized_state in {"pending", "all"}:
        purged_counts["pending"] = huey.pending_count()
        huey.storage.flush_queue()

    if normalized_state in {"scheduled", "all"}:
        purged_counts["scheduled"] = huey.scheduled_count()
        huey.storage.flush_schedule()

    if normalized_state in {"failed", "all"}:
        purged_counts["failed"] = len(_collect_failures(huey))
        huey.storage.flush_results()

    return QueuePurgeResult(
        purged_state=normalized_state,
        purged_counts=purged_counts,
        queue_stats=get_queue_stats(
            huey,
            worker_running=worker_running,
            backpressure_limit=backpressure_limit,
        ),
    )
