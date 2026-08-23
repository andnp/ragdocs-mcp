"""Deterministic, explicitly refreshed snapshots for daemon status data."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

SnapshotValue = TypeVar("SnapshotValue")


@dataclass(frozen=True)
class SnapshotStatus:
    age_seconds: float | None
    stale: bool
    error: str | None
    stale_after_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "age_seconds": self.age_seconds,
            "stale": self.stale,
            "error": self.error,
            "stale_after_seconds": self.stale_after_seconds,
        }


class StatusSnapshot(Generic[SnapshotValue]):
    """Store one value until an explicit refresh replaces it."""

    def __init__(
        self,
        *,
        stale_after_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self._stale_after_seconds = stale_after_seconds
        self._clock = clock
        self._lock = Lock()
        self._value: SnapshotValue | None = None
        self._captured_at: float | None = None
        self._error: str | None = None

    def read(
        self,
        builder: Callable[[], SnapshotValue],
    ) -> tuple[SnapshotValue, SnapshotStatus]:
        with self._lock:
            if self._value is None:
                self._refresh_locked(builder)
            return self._value_and_status_locked()

    def refresh(
        self,
        builder: Callable[[], SnapshotValue],
    ) -> tuple[SnapshotValue, SnapshotStatus]:
        with self._lock:
            self._refresh_locked(builder)
            return self._value_and_status_locked()

    def _refresh_locked(self, builder: Callable[[], SnapshotValue]) -> None:
        try:
            value = builder()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            if self._value is None:
                raise
            return

        self._value = value
        self._captured_at = self._clock()
        self._error = None

    def _value_and_status_locked(self) -> tuple[SnapshotValue, SnapshotStatus]:
        if self._value is None:
            raise RuntimeError("status snapshot has no value")
        now = self._clock()
        age_seconds = (
            None
            if self._captured_at is None
            else max(now - self._captured_at, 0.0)
        )
        status = SnapshotStatus(
            age_seconds=age_seconds,
            stale=(
                age_seconds is None
                or age_seconds > self._stale_after_seconds
                or self._error is not None
            ),
            error=self._error,
            stale_after_seconds=self._stale_after_seconds,
        )
        return self._value, status
