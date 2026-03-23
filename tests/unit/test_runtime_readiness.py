from src.indexing.runtime_readiness import (
    can_refresh_loaded_indices,
    can_serve_queries,
    is_fully_ready,
)


def test_can_serve_queries_requires_queryable_indices():
    assert can_serve_queries(
        init_error=None,
        ready_event_set=True,
        is_virgin_startup=False,
        indices_queryable=False,
    ) is False


def test_can_serve_queries_blocks_virgin_startup_before_ready_event():
    assert can_serve_queries(
        init_error=None,
        ready_event_set=False,
        is_virgin_startup=True,
        indices_queryable=True,
    ) is False


def test_can_serve_queries_allows_loaded_nonvirgin_startup_before_ready_event():
    assert can_serve_queries(
        init_error=None,
        ready_event_set=False,
        is_virgin_startup=False,
        indices_queryable=True,
    ) is True


def test_can_serve_queries_blocks_init_error_even_if_indices_are_loaded():
    assert can_serve_queries(
        init_error=RuntimeError("boom"),
        ready_event_set=True,
        is_virgin_startup=False,
        indices_queryable=True,
    ) is False


def test_is_fully_ready_requires_ready_event_ready_status_and_queryable_indices():
    assert is_fully_ready(
        init_error=None,
        ready_event_set=True,
        index_status="ready",
        indices_queryable=True,
    ) is True
    assert is_fully_ready(
        init_error=None,
        ready_event_set=False,
        index_status="ready",
        indices_queryable=True,
    ) is False
    assert is_fully_ready(
        init_error=None,
        ready_event_set=True,
        index_status="partial",
        indices_queryable=True,
    ) is False
    assert is_fully_ready(
        init_error=None,
        ready_event_set=True,
        index_status="ready",
        indices_queryable=False,
    ) is False


def test_can_refresh_loaded_indices_requires_completed_nonfailed_startup():
    assert can_refresh_loaded_indices(
        ready_event_set=True,
        init_error=None,
    ) is True
    assert can_refresh_loaded_indices(
        ready_event_set=False,
        init_error=None,
    ) is False
    assert can_refresh_loaded_indices(
        ready_event_set=True,
        init_error=RuntimeError("boom"),
    ) is False