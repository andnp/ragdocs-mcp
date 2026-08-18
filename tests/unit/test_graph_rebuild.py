from threading import Event

from mcp_markdown_ragdocs.indexing.graph_rebuild import DebouncedGraphRebuilder


def test_graph_rebuilder_runs_the_latest_pending_snapshot() -> None:
    """A running rebuild is followed by only the newest queued snapshot."""
    first_started = Event()
    release_first = Event()
    calls: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
    first_snapshot = (("first", ("one",)),)
    latest_snapshot = (("latest", ("three",)),)

    def rebuild(snapshot: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        calls.append(snapshot)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=1)

    coordinator = DebouncedGraphRebuilder(rebuild, debounce_seconds=0)
    try:
        coordinator.request(first_snapshot)
        assert first_started.wait(timeout=1)
        coordinator.request((("middle", ("two",)),))
        coordinator.request(latest_snapshot)
        release_first.set()
        coordinator.flush()

        assert calls == [first_snapshot, latest_snapshot]
    finally:
        coordinator.close()
