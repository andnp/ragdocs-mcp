"""Tests for DriveRequestGate's bounded-concurrency and backoff contract."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.gdrive.gate import DriveRequestGate


class _ApiError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider status {status}")
        self.resp = type("Response", (), {"status": status})()


class FakeClock:
    """A manually-advanced clock whose sleep() advances it.

    Lets claim-spacing and expiry tests run deterministically without any
    real wall-clock delay.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_gate_allows_up_to_max_concurrent_claims_in_flight(tmp_path: Path) -> None:
    """max_concurrent operations may run at once; the next one waits."""
    gate = DriveRequestGate(tmp_path / "gate.db", min_interval_seconds=0, max_concurrent=2)
    entered = threading.Semaphore(0)
    release = threading.Event()

    def held_operation() -> None:
        entered.release()
        assert release.wait(timeout=5), "held operation was never released"

    held_threads = [
        threading.Thread(target=lambda: gate.run(held_operation)) for _ in range(2)
    ]
    for thread in held_threads:
        thread.start()
    assert entered.acquire(timeout=2)
    assert entered.acquire(timeout=2)

    extra_entered = threading.Event()
    extra_thread = threading.Thread(target=lambda: gate.run(extra_entered.set))
    extra_thread.start()

    assert not extra_entered.wait(timeout=0.2), "third claim proceeded past max_concurrent"

    release.set()
    for thread in held_threads:
        thread.join(timeout=2)
    extra_thread.join(timeout=2)

    assert extra_entered.is_set()


def test_gate_spaces_successive_claims_by_min_interval(tmp_path: Path) -> None:
    """Claims are spaced by min_interval_seconds even with slots free."""
    clock = FakeClock()
    gate = DriveRequestGate(
        tmp_path / "gate.db",
        min_interval_seconds=0.2,
        max_concurrent=4,
        time_source=clock.time,
        sleep=clock.sleep,
    )
    claim_times: list[float] = []

    gate.run(lambda: claim_times.append(clock.time()))
    gate.run(lambda: claim_times.append(clock.time()))

    assert claim_times[1] - claim_times[0] >= 0.2


def test_gate_reclaims_an_expired_slot_from_a_crashed_holder(tmp_path: Path) -> None:
    """A holder that crashes without releasing must not deadlock the gate."""
    clock = FakeClock()
    gate = DriveRequestGate(
        tmp_path / "gate.db",
        min_interval_seconds=0,
        max_concurrent=1,
        request_timeout_seconds=60,
        time_source=clock.time,
        sleep=clock.sleep,
    )
    # Simulate a crashed process: claim a slot and never release it.
    gate._claim()

    clock.now += 61  # past the slot's expiry

    completed = threading.Event()
    gate.run(completed.set)

    assert completed.is_set()


def test_gate_extends_cooldown_after_a_429(tmp_path: Path) -> None:
    """A 429 backs off all claimants via provider_cooldown_seconds."""
    clock = FakeClock()
    gate = DriveRequestGate(
        tmp_path / "gate.db",
        min_interval_seconds=0,
        max_concurrent=4,
        provider_cooldown_seconds=5,
        time_source=clock.time,
        sleep=clock.sleep,
    )

    def rate_limited() -> None:
        raise _ApiError(429)

    with pytest.raises(_ApiError):
        gate.run(rate_limited)

    claimed_at: list[float] = []
    gate.run(lambda: claimed_at.append(clock.time()))

    assert claimed_at[0] >= 5.0


def test_gate_migrates_legacy_in_flight_schema(tmp_path: Path) -> None:
    """A pre-slot gate database remains usable after the schema upgrade."""
    path = tmp_path / "gate.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE drive_request_gate (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_allowed_at REAL NOT NULL,
                in_flight_until REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO drive_request_gate VALUES (1, 2.0, 0.0)"
        )

    gate = DriveRequestGate(path, min_interval_seconds=0)
    gate.run(lambda: None)

    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(drive_request_gate)"
            ).fetchall()
        }
    assert columns == {"id", "next_allowed_at"}


def test_gate_preserves_unexpired_legacy_claim(tmp_path: Path) -> None:
    """An unexpired legacy claim becomes a durable slot during migration."""
    path = tmp_path / "gate.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE drive_request_gate (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_allowed_at REAL NOT NULL,
                in_flight_until REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO drive_request_gate VALUES (1, 0.0, 100.0)"
        )

    DriveRequestGate(
        path,
        min_interval_seconds=0,
        max_concurrent=1,
        time_source=lambda: 50.0,
        sleep=lambda seconds: None,
    )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT expires_at FROM drive_request_gate_slots"
        ).fetchone() == (100.0,)
