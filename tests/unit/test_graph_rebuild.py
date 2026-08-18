from threading import Event

from mcp_markdown_ragdocs.indexing.graph_rebuild import DebouncedGraphRebuilder


def test_graph_rebuilder_coalesces_requests_made_while_it_runs() -> None:
    """Requests arriving during a rebuild collapse into exactly one more run."""
    first_started = Event()
    release_first = Event()
    runs = 0

    def rebuild() -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            first_started.set()
            assert release_first.wait(timeout=1)

    coordinator = DebouncedGraphRebuilder(rebuild, debounce_seconds=0)
    try:
        coordinator.request()
        assert first_started.wait(timeout=1)
        coordinator.request()
        coordinator.request()
        release_first.set()
        coordinator.flush()

        assert runs == 2
    finally:
        coordinator.close()
