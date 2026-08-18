"""Work-intent coordination for indexing tasks."""

from __future__ import annotations

import functools
import logging
from typing import Protocol, cast

from mcp_markdown_ragdocs.coordination.work_intents import (
    CLAIMED,
    PENDING,
    RUNNING,
    WorkIntent,
    WorkIntentPort,
)

logger = logging.getLogger(__name__)


class WorkIntentPortProvider(Protocol):
    def __call__(self) -> WorkIntentPort | None: ...


def _intent_claim(
    store_provider: WorkIntentPortProvider,
    operation: str,
    canonical_key: str,
    payload: dict[str, object],
    *,
    force_reopen: bool = False,
) -> tuple[WorkIntent, str] | None:
    store = store_provider()
    if store is None:
        return None
    intent = store.submit(
        operation,
        canonical_key,
        payload,
        force_reopen=force_reopen,
    )
    if intent.state != PENDING:
        if (
            force_reopen
            and intent.state in {CLAIMED, RUNNING}
            and intent.claim_token is not None
        ):
            return intent, intent.claim_token
        return None
    return store.claim(intent.intent_id)


def _intent_claim_batch(
    store_provider: WorkIntentPortProvider,
    operation: str,
    items: list[tuple[str, dict[str, object]]],
    *,
    force_reopen: bool = False,
) -> tuple[list[tuple[str, tuple[str, str]]], int]:
    store = store_provider()
    if store is None:
        return [], 0
    claims: list[tuple[str, tuple[str, str]]] = []
    skipped = 0
    for canonical_key, payload in items:
        claim = _intent_claim(
            store_provider,
            operation,
            canonical_key,
            payload,
            force_reopen=force_reopen,
        )
        if claim is None:
            skipped += 1
        else:
            claims.append((canonical_key, (claim[0].intent_id, claim[1])))
    return claims, skipped


def _release_intent(
    store_provider: WorkIntentPortProvider,
    intent_id: str,
    claim_token: str,
) -> None:
    store = store_provider()
    if store is not None:
        store.release(intent_id, claim_token)


def _intent_task(store_provider: WorkIntentPortProvider, operation: str):
    def _result_outcomes(result: object, count: int) -> list[bool] | None:
        if not isinstance(result, dict):
            return None
        raw_outcomes = result.get("outcomes")
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) != count:
            return None
        if not all(isinstance(outcome, bool) for outcome in raw_outcomes):
            return None
        return cast(list[bool], raw_outcomes)

    def _decorate(function):
        @functools.wraps(function)
        def _wrapped(*args, **kwargs):
            intent_id = kwargs.pop("intent_id", None)
            claim_token = kwargs.pop("claim_token", None)
            claims = kwargs.pop("intent_claims", None)
            claim_pairs = (
                list(claims)
                if isinstance(claims, list)
                else (
                    [(intent_id, claim_token)]
                    if isinstance(intent_id, str) and isinstance(claim_token, str)
                    else []
                )
            )
            store = store_provider()
            if store is not None and claim_pairs:
                started: list[tuple[str, str]] = []
                for item in claim_pairs:
                    if (
                        not isinstance(item, (list, tuple))
                        or len(item) != 2
                        or not store.start(str(item[0]), str(item[1]))
                    ):
                        for started_id, started_token in started:
                            store.fail(
                                started_id,
                                started_token,
                                "batch claim became stale before execution",
                            )
                        return False
                    started.append((str(item[0]), str(item[1])))
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                if store is not None:
                    for item in claim_pairs:
                        store.fail(str(item[0]), str(item[1]), f"{type(exc).__name__}: {exc}")
                raise
            success = result is True or (
                isinstance(result, dict)
                and result.get("status") in {"ok", "succeeded"}
            )
            if store is not None:
                outcomes = _result_outcomes(result, len(claim_pairs))
                if outcomes is None:
                    outcomes = [success] * len(claim_pairs)
                error = (
                    str(result["error"])
                    if isinstance(result, dict) and isinstance(result.get("error"), str)
                    else "task execution failed"
                )
                for item, item_succeeded in zip(claim_pairs, outcomes, strict=True):
                    if item_succeeded:
                        store.succeed(str(item[0]), str(item[1]))
                    else:
                        store.fail(str(item[0]), str(item[1]), error)
                        logger.warning(
                            "Intent task failed: operation=%s result_status=%s outcome=%s error=%s",
                            operation,
                            result.get("status") if isinstance(result, dict) else result,
                            item_succeeded,
                            error,
                        )
            return result

        return _wrapped

    return _decorate
