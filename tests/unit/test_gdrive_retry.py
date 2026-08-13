"""Tests for durable Google Drive retry work."""

from pathlib import Path

from mcp_markdown_ragdocs.coordination.work_intents import WorkIntentStore
from mcp_markdown_ragdocs.gdrive.retry import DriveRetryWorkStore


class _ApiError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider status {status}")
        self.resp = type("Response", (), {"status": status})()


def _store(tmp_path: Path) -> DriveRetryWorkStore:
    return DriveRetryWorkStore(WorkIntentStore(tmp_path / "queue.db"))


def test_retryable_provider_failure_creates_durable_work(tmp_path: Path) -> None:
    """
    Persist a retry intent with provider diagnostics for later recovery.
    """
    store = _store(tmp_path)

    work = store.schedule_failure(
        scope_identity="shared-with-me",
        source_id="file-1",
        operation="materialize",
        payload={"mime_type": "text/plain"},
        error=_ApiError(503),
        now=10,
    )

    assert work is not None
    assert work.state == "pending"
    assert work.payload["provider_status"] == 503
    assert work.payload["provider_message"] == "provider status 503"


def test_retryable_attempt_can_be_reclaimed_and_completed(tmp_path: Path) -> None:
    """
    Return transiently failed work to pending state before a fresh owner claims it.
    """
    store = _store(tmp_path)
    work = store.schedule(
        scope_identity="shared-with-me",
        source_id="file-1",
        operation="materialize",
        payload={},
        now=10,
    )

    first = store.claim(work.intent_id, now=11)
    assert first is not None
    assert store.retry(work.intent_id, first[1], TimeoutError("temporary timeout"), now=12)

    second = store.claim(work.intent_id, now=13)
    assert second is not None
    assert second[0].attempt == 2
    assert store.complete(work.intent_id, second[1], now=14)


def test_definitive_attempt_stays_failed(tmp_path: Path) -> None:
    """
    Do not requeue a retry intent after a definitive provider response.
    """
    store = _store(tmp_path)
    work = store.schedule(
        scope_identity="shared-with-me",
        source_id="file-1",
        operation="materialize",
        payload={},
        now=10,
    )
    claimed = store.claim(work.intent_id, now=11)
    assert claimed is not None

    assert store.retry(work.intent_id, claimed[1], _ApiError(404), now=12)
    assert store.claim(work.intent_id, now=13) is None

