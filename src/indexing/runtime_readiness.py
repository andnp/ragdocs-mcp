from __future__ import annotations

from typing import Literal


type IndexStatus = Literal["uninitialized", "indexing", "partial", "ready", "failed"]


def can_serve_queries(
    *,
    init_error: Exception | None,
    ready_event_set: bool,
    is_virgin_startup: bool,
    indices_queryable: bool,
) -> bool:
    if init_error is not None:
        return False
    if not indices_queryable:
        return False
    if ready_event_set:
        return True
    return not is_virgin_startup


def is_fully_ready(
    *,
    init_error: Exception | None,
    ready_event_set: bool,
    index_status: IndexStatus,
    indices_queryable: bool,
) -> bool:
    return (
        ready_event_set
        and init_error is None
        and index_status == "ready"
        and indices_queryable
    )


def can_refresh_loaded_indices(
    *,
    ready_event_set: bool,
    init_error: Exception | None,
) -> bool:
    return ready_event_set and init_error is None