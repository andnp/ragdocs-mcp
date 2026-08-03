"""Bounded metadata and liveness checks for managed producers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProducerMetadata:
    pid: int
    start_time_ticks: int
    started_at: float
    status: str = "active"
    stop_reason: str | None = None


def write_producer_metadata(path: Path, metadata: ProducerMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), sort_keys=True), encoding="utf-8")


def _valid_pid(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_start_time_ticks(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_started_at(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_producer_metadata(path: Path) -> ProducerMetadata | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if not _valid_pid(payload.get("pid")):
            return None
        if not _valid_start_time_ticks(payload.get("start_time_ticks")):
            return None
        if not _valid_started_at(payload.get("started_at")):
            return None
        if not isinstance(payload.get("status", "active"), str):
            return None
        stop_reason = payload.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, str):
            return None
        return ProducerMetadata(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def producer_is_live(metadata: ProducerMetadata | None) -> bool:
    if (
        metadata is None
        or metadata.status != "active"
        or not _valid_pid(metadata.pid)
        or not _valid_start_time_ticks(metadata.start_time_ticks)
    ):
        return False
    try:
        os.kill(metadata.pid, 0)
    except (OSError, TypeError, ValueError, OverflowError):
        return False
    actual_start_time = read_process_start_time_ticks(metadata.pid)
    return actual_start_time is not None and actual_start_time == metadata.start_time_ticks


def read_process_start_time_ticks(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "stat"
    try:
        stat_line = path.read_text(encoding="utf-8")
    except OSError:
        return None
    closing_paren = stat_line.rfind(")")
    if closing_paren == -1:
        return None
    fields = stat_line[closing_paren + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def producer_diagnostics(
    metadata: ProducerMetadata | None,
) -> dict[str, object]:
    if metadata is None:
        return {
            "watcher_active": False,
            "producer_pid": None,
            "producer_started_at": None,
            "stop_reason": None,
        }
    return {
        "watcher_active": producer_is_live(metadata),
        "producer_pid": metadata.pid,
        "producer_started_at": metadata.started_at,
        "stop_reason": metadata.stop_reason,
    }
