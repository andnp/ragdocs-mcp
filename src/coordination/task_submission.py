"""Shared helpers for task queue inspection, coalescing, and submission."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huey import SqliteHuey

logger = logging.getLogger(__name__)


type TaskSubmitter = Callable[..., object]


def get_pending_task_count(huey: SqliteHuey | None) -> int:
    if huey is None:
        return 0
    return int(huey.pending_count())


def is_backpressured(
    huey: SqliteHuey | None,
    limit: int,
    *,
    item: str,
    warning_message: str,
) -> bool:
    pending_count = get_pending_task_count(huey)
    if pending_count < limit:
        return False

    logger.warning(warning_message, item, pending_count, limit)
    return True


def get_pending_task_first_args(
    huey: SqliteHuey | None,
    task_name: str,
    *,
    inspection_failure_log_message: str,
    deserialize_failure_log_message: str,
) -> set[str]:
    if huey is None:
        return set()

    try:
        pending_messages = huey.storage.enqueued_items()
    except Exception:
        logger.warning(inspection_failure_log_message, exc_info=True)
        return set()

    pending_args: set[str] = set()
    for message in pending_messages:
        try:
            task = huey.deserialize_task(message)
        except Exception:
            logger.warning(deserialize_failure_log_message, exc_info=True)
            continue

        if getattr(task, "name", None) != task_name:
            continue

        args = getattr(task, "args", ())
        if not args:
            continue

        first_arg = args[0]
        if isinstance(first_arg, str):
            pending_args.add(first_arg)

    return pending_args


def submit_single_task(
    task_submitter: TaskSubmitter,
    first_arg: str,
    *,
    task_kwargs: Mapping[str, object] | None = None,
    pending_first_args: set[str] | None = None,
    pending_skip_log_message: str | None = None,
) -> bool:
    if pending_first_args is not None and first_arg in pending_first_args:
        if pending_skip_log_message is not None:
            logger.info(pending_skip_log_message, first_arg)
        return False

    _submit_task(task_submitter, first_arg, task_kwargs=task_kwargs)
    return True


def submit_task_batch(
    task_submitter: TaskSubmitter,
    first_args: list[str],
    *,
    task_kwargs: Mapping[str, object] | None = None,
    pending_first_args: set[str] | None = None,
    skipped_pending_log_message: str | None = None,
) -> int:
    pending_items = pending_first_args or set()
    seen_items = set(pending_items)
    enqueued = 0
    skipped_pending = 0

    for first_arg in first_args:
        if first_arg in seen_items:
            if first_arg in pending_items:
                skipped_pending += 1
            continue

        _submit_task(task_submitter, first_arg, task_kwargs=task_kwargs)
        seen_items.add(first_arg)
        enqueued += 1

    if skipped_pending > 0 and skipped_pending_log_message is not None:
        logger.info(skipped_pending_log_message, skipped_pending)

    return enqueued


def _submit_task(
    task_submitter: TaskSubmitter,
    first_arg: str,
    *,
    task_kwargs: Mapping[str, object] | None,
) -> None:
    if task_kwargs is None:
        task_submitter(first_arg)
        return

    task_submitter(first_arg, **dict(task_kwargs))