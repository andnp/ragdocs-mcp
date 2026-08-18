"""Durable retry work for recoverable Google Drive failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_markdown_ragdocs.coordination.work_intents import WorkIntent, WorkIntentPort
from mcp_markdown_ragdocs.gdrive.errors import classify_provider_error

DRIVE_RETRY_OPERATION = "gdrive_retry"


@dataclass(frozen=True, slots=True)
class DriveRetryWork:
    """A retryable Drive operation with a stable deduplication key."""

    intent_id: str
    scope_identity: str
    source_id: str
    operation: str
    payload: dict[str, Any]
    state: str
    attempt: int
    failure_count: int
    error: str | None


class DriveRetryWorkStore:
    """Adapt generic work intents to Drive retry semantics."""

    def __init__(self, intents: WorkIntentPort) -> None:
        self._intents = intents

    def schedule(
        self,
        *,
        scope_identity: str,
        source_id: str,
        operation: str,
        payload: dict[str, Any],
        now: float | None = None,
    ) -> DriveRetryWork:
        key = f"{scope_identity}:{source_id}:{operation}"
        intent = self._intents.submit(
            DRIVE_RETRY_OPERATION,
            key,
            {
                "scope_identity": scope_identity,
                "source_id": source_id,
                "operation": operation,
                "payload": payload,
            },
            now=now,
        )
        return _work(intent)

    def schedule_failure(
        self,
        *,
        scope_identity: str,
        source_id: str,
        operation: str,
        payload: dict[str, Any],
        error: BaseException,
        now: float | None = None,
    ) -> DriveRetryWork | None:
        """Persist only failures that the provider classifies as recoverable."""
        info = classify_provider_error(error)
        if not info.retryable:
            return None
        return self.schedule(
            scope_identity=scope_identity,
            source_id=source_id,
            operation=operation,
            payload={
                **payload,
                "provider_status": info.status_code,
                "provider_reason": info.reason,
                "provider_message": info.message,
            },
            now=now,
        )

    def claim(
        self,
        intent_id: str,
        *,
        now: float | None = None,
    ) -> tuple[DriveRetryWork, str] | None:
        claimed = self._intents.claim(intent_id, now=now)
        return None if claimed is None else (_work(claimed[0]), claimed[1])

    def retry(
        self,
        intent_id: str,
        claim_token: str,
        error: BaseException,
        *,
        now: float | None = None,
    ) -> bool:
        """Return recoverable work to pending state after a failed attempt."""
        if not classify_provider_error(error).retryable:
            return self._intents.fail(intent_id, claim_token, str(error), now=now)
        return self._intents.release(intent_id, claim_token, now=now)

    def complete(self, intent_id: str, claim_token: str, *, now: float | None = None) -> bool:
        return self._intents.succeed(intent_id, claim_token, now=now)


def _work(intent: WorkIntent) -> DriveRetryWork:
    payload = intent.payload
    operation_payload = payload.get("payload")
    return DriveRetryWork(
        intent_id=intent.intent_id,
        scope_identity=str(payload["scope_identity"]),
        source_id=str(payload["source_id"]),
        operation=str(payload["operation"]),
        payload=dict(operation_payload) if isinstance(operation_payload, dict) else {},
        state=intent.state,
        attempt=intent.attempt,
        failure_count=intent.failure_count,
        error=intent.error,
    )


__all__ = ["DRIVE_RETRY_OPERATION", "DriveRetryWork", "DriveRetryWorkStore"]
