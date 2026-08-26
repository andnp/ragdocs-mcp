"""Huey consumer wrapper for running task workers."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from huey.constants import EmptyData
from huey.utils import Error

from mcp_markdown_ragdocs.coordination.task_leases import (
    DEFAULT_LEASE_TIMEOUT_SECONDS,
    TaskLeaseStore,
)
from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)

LEASE_TIMEOUT_SECONDS = DEFAULT_LEASE_TIMEOUT_SECONDS
LEASE_HEARTBEAT_INTERVAL_SECONDS = 5.0
LEASE_RECLAIM_INTERVAL_SECONDS = 5.0
NON_QUEUE_LEASE_TASK_NAMES = frozenset({"gdrive_scope_sync"})
SCHEDULE_PROMOTION_BATCH_SIZE = 32
# Retention pruning is bounded work (see prune_terminal()) but there is no
# reason to run it on every 5s reclaim tick; a coarser cadence on the same
# loop keeps this from becoming its own periodic-scan performance bug.
PRUNE_INTERVAL_SECONDS = 300.0


def _requeue_expired_leases(
    huey: SqliteHuey,
    lease_store: TaskLeaseStore,
) -> None:
    intent_store = WorkIntentStore(
        cast(Any, huey.storage).filename,
        claim_timeout_seconds=LEASE_TIMEOUT_SECONDS,
    )
    for lease in lease_store.reclaim_expired():
        if lease.task_name in NON_QUEUE_LEASE_TASK_NAMES:
            logger.info("Skipping expired non-queue lease: %s", lease.task_id)
            continue
        if lease.payload is None:
            raise RuntimeError(
                f"Expired lease {lease.task_id} has no serialized task payload"
            )
        task = huey.deserialize_task(lease.payload)
        _refresh_task_intent_claims(intent_store, task)
        huey.enqueue(task)


def _refresh_task_intent_claims(intent_store: WorkIntentStore, task: Any) -> None:
    kwargs = getattr(task, "kwargs", None)
    if not isinstance(kwargs, dict):
        return

    def _refresh_claim(intent_id: object, claim_token: object) -> str | None:
        if not isinstance(intent_id, str) or not isinstance(claim_token, str):
            return None
        reclaimed = intent_store.reclaim_stale_claim(intent_id, claim_token)
        return None if reclaimed is None else reclaimed[1]

    intent_id = kwargs.get("intent_id")
    claim_token = kwargs.get("claim_token")
    refreshed_token = _refresh_claim(intent_id, claim_token)
    if refreshed_token is not None:
        kwargs["claim_token"] = refreshed_token

    claims = kwargs.get("intent_claims")
    if not isinstance(claims, list):
        return
    refreshed_claims: list[object] = []
    for item in claims:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            refreshed_claims.append(item)
            continue
        refreshed = _refresh_claim(item[0], item[1])
        refreshed_claims.append(
            (item[0], refreshed if refreshed is not None else item[1])
        )
    kwargs["intent_claims"] = refreshed_claims


def _prune_terminal_state(
    huey: SqliteHuey,
    lease_store: TaskLeaseStore,
) -> None:
    """Reap terminal coordination rows and their Huey result data.

    Only rows that claim()/reclaim_expired()/claim() (work_intents) never
    read again are eligible for deletion — see prune_terminal() on each
    store for the recovery-safety argument. The Huey kv result for a task
    is only ever peeked (never popped) anywhere in this codebase, so
    deleting it once its lease is terminal and past retention cannot drop a
    result a caller still needs to consume.
    """
    intent_store = WorkIntentStore(
        cast(Any, huey.storage).filename,
        claim_timeout_seconds=LEASE_TIMEOUT_SECONDS,
    )
    for task_id in lease_store.prune_terminal():
        huey.storage.delete_data(task_id)
    intent_store.prune_terminal()


def _promote_due_tasks(huey: SqliteHuey) -> None:
    """Atomically transfer a bounded batch of due tasks to the queue."""
    storage = cast(Any, huey.storage)
    timestamp = (
        datetime.now(timezone.utc).replace(tzinfo=None)
        if huey.utc
        else datetime.now()
    )
    promoted_ids: list[int] = []
    try:
        with storage.db(commit=True) as cursor:
            cursor.execute(
                """
                SELECT id, data
                FROM schedule
                WHERE queue = ? AND timestamp <= ?
                ORDER BY timestamp, id
                LIMIT ?
                """,
                (storage.name, timestamp.timestamp(), SCHEDULE_PROMOTION_BATCH_SIZE),
            )
            rows = cursor.fetchall()
            for schedule_id, raw_data in rows:
                data = bytes(raw_data)
                try:
                    task = huey.deserialize_task(data)
                except Exception:
                    logger.exception(
                        "Unable to deserialize scheduled task %s", schedule_id
                    )
                    continue

                cursor.execute(
                    "INSERT INTO task (queue, data, priority) VALUES (?, ?, ?)",
                    (storage.name, data, task.priority or 0),
                )
                promoted_ids.append(int(schedule_id))

            if promoted_ids:
                placeholders = ", ".join("?" for _ in promoted_ids)
                cursor.execute(
                    f"DELETE FROM schedule WHERE queue = ? AND id IN ({placeholders})",
                    (storage.name, *promoted_ids),
                )
    except Exception:
        logger.exception("Unable to promote due Huey tasks")
        return

    logger.debug("Promoted %d due Huey task(s)", len(promoted_ids))


class HueyWorker:
    """Manages a Huey consumer thread for processing background tasks.

    Runs in the same process as the main server, in a daemon thread.
    Only started when the lifecycle state is READY_PRIMARY.
    """

    def __init__(self, huey: SqliteHuey, workers: int = 2) -> None:
        self._huey = huey
        self._workers = workers
        self._consumers: list[_HueyConsumerThread] = []
        self._started = False

    @property
    def is_running(self) -> bool:
        return (
            self._started
            and bool(self._consumers)
            and any(consumer.is_alive() for consumer in self._consumers)
        )

    def start(self) -> None:
        """Start the consumer thread pool."""
        if self._started:
            logger.warning("HueyWorker already started")
            return

        lease_store = TaskLeaseStore(
            cast(Any, self._huey.storage).filename,
            timeout_seconds=LEASE_TIMEOUT_SECONDS,
        )
        _requeue_expired_leases(self._huey, lease_store)

        self._consumers = [
            _HueyConsumerThread(
                self._huey,
                lease_store=lease_store,
                is_reclaimer=(index == 0),
            )
            for index in range(self._workers)
        ]
        for consumer in self._consumers:
            consumer.start()
        self._started = True
        logger.info("Huey worker started with %d workers", self._workers)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the consumer thread pool."""
        if not self._started or not self._consumers:
            return

        for consumer in self._consumers:
            consumer.request_stop()
        for consumer in self._consumers:
            consumer.join(timeout=timeout)

        still_alive = [consumer for consumer in self._consumers if consumer.is_alive()]
        if still_alive:
            logger.warning(
                "%d Huey consumer thread(s) did not stop within %.1fs",
                len(still_alive),
                timeout,
            )
        else:
            logger.info("Huey worker stopped")

        self._consumers = []
        self._started = False


class _HueyConsumerThread(threading.Thread):
    """Thread running the Huey consumer loop."""

    def __init__(
        self,
        huey: SqliteHuey,
        *,
        lease_store: TaskLeaseStore,
        is_reclaimer: bool = True,
    ) -> None:
        super().__init__(name="huey-consumer", daemon=True)
        self._huey = huey
        self._lease_store = lease_store
        self._is_reclaimer = is_reclaimer
        self._owner_token = uuid.uuid4().hex
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Run the consumer loop, processing tasks from the queue."""
        logger.info("Huey consumer thread started")
        try:
            next_reclaim_at = 0.0
            next_prune_at = 0.0
            while not self._stop_event.is_set():
                now = time.monotonic()
                if self._is_reclaimer and now >= next_reclaim_at:
                    _promote_due_tasks(self._huey)
                    _requeue_expired_leases(self._huey, self._lease_store)
                    next_reclaim_at = now + LEASE_RECLAIM_INTERVAL_SECONDS
                if self._is_reclaimer and now >= next_prune_at:
                    _prune_terminal_state(self._huey, self._lease_store)
                    next_prune_at = now + PRUNE_INTERVAL_SECONDS
                # Dequeue and execute one task at a time
                task = self._huey.dequeue()
                if task is not None:
                    self._execute_task(task)
                else:
                    # No task available, wait briefly
                    self._stop_event.wait(timeout=0.5)
        except Exception:
            logger.exception("Huey consumer thread error")
        finally:
            logger.info("Huey consumer thread exiting")

    def _execute_task(self, task: Any) -> None:
        task_id = str(task.id)
        task_name = getattr(task, "name", None)
        claimed = self._lease_store.claim(
            task_id,
            task_name=task_name if isinstance(task_name, str) else None,
            owner_token=self._owner_token,
            payload=self._huey.serialize_task(task),
        )
        if not claimed:
            logger.warning("Skipping task with an active lease: %s", task_id)
            return

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            args=(task_id, heartbeat_stop),
            name=f"huey-heartbeat-{task_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self._huey.execute(task)
        except Exception as exc:
            self._lease_store.fail(
                task_id,
                owner_token=self._owner_token,
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("Task execution failed")
        else:
            execution_error = self._read_execution_error(task)
            if execution_error is None:
                self._lease_store.complete(
                    task_id,
                    owner_token=self._owner_token,
                )
            else:
                self._lease_store.fail(
                    task_id,
                    owner_token=self._owner_token,
                    error=execution_error,
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=LEASE_HEARTBEAT_INTERVAL_SECONDS)

    def _heartbeat(self, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(LEASE_HEARTBEAT_INTERVAL_SECONDS):
            if not self._lease_store.heartbeat(
                task_id,
                owner_token=self._owner_token,
            ):
                logger.warning("Lost task lease ownership: %s", task_id)
                return

    def _read_execution_error(self, task: Any) -> str | None:
        raw_item = cast(Any, self._huey.storage).peek_data(task.id)
        if raw_item is None or raw_item is EmptyData:
            return None
        payload = self._huey.serializer.deserialize(raw_item)
        if not isinstance(payload, Error):
            return None
        metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
        return str(metadata.get("error", "task execution failed"))
