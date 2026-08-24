from __future__ import annotations

from pathlib import Path

from searchkernel.api import (
    CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    BootstrapCheckpoint,
    BootstrapFileStamp,
    SearchAvailability,
    get_bootstrap_availability,
    load_bootstrap_checkpoint,
    publish_bootstrap_availability,
    save_bootstrap_checkpoint,
)

from mcp_markdown_ragdocs.indexing.progressive import (
    StagedBootstrapPort,
    run_progressive_bootstrap,
)


class _CanonicalManager:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.kernel = object()
        self.persist_calls = 0
        self.indexed: list[str] = []
        self.graph_rebuild_calls = 0
        self.update_graph_flags: list[bool] = []
        self.events: list[tuple[str, str | None]] = []
        self.checkpoint_at_persist: BootstrapCheckpoint | None = None
        self.failed_paths: set[str] = set()

    def persist(self) -> None:
        self.persist_calls += 1
        self.events.append(("persist", None))
        self.checkpoint_at_persist = load_bootstrap_checkpoint(self.index_path)

    def index_document(
        self,
        file_path: str,
        *,
        update_graph: bool = True,
    ) -> bool:
        self.indexed.append(file_path)
        self.update_graph_flags.append(update_graph)
        self.events.append(("index", file_path))
        return file_path not in self.failed_paths

    def rebuild_graph(self) -> None:
        self.graph_rebuild_calls += 1
        self.events.append(("graph", None))


def _save_pending_checkpoint(index_path: Path, paths: list[Path]) -> None:
    stamps = {
        path.name: BootstrapFileStamp(
            path.name,
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in paths
    }
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets=stamps,
            completed={},
        ),
    )


def test_canonical_manager_satisfies_staged_bootstrap_port(tmp_path: Path) -> None:
    """Require the existing manager capabilities for staged bootstrap."""
    manager = _CanonicalManager(tmp_path)

    assert isinstance(manager, StagedBootstrapPort)


def test_canonical_bootstrap_marks_checkpoint_complete(
    tmp_path: Path,
) -> None:
    """Resume pending files and publish complete search availability."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First")
    second.write_text("# Second")
    index_path = tmp_path / "index"
    index_path.mkdir()
    first_stamp = BootstrapFileStamp(
        first.name,
        first.stat().st_mtime_ns,
        first.stat().st_size,
    )
    second_stamp = BootstrapFileStamp(
        second.name,
        second.stat().st_mtime_ns,
        second.stat().st_size,
    )
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets={
                first.name: first_stamp,
                second.name: second_stamp,
            },
            completed={first.name: first_stamp},
        ),
    )

    manager = _CanonicalManager(index_path)
    receipt = run_progressive_bootstrap(
        manager,
        [str(first), str(second)],
        documents_roots=[tmp_path],
    )

    checkpoint = load_bootstrap_checkpoint(index_path)
    assert receipt.successful == 1
    assert manager.indexed == [str(second)]
    assert manager.update_graph_flags == [False]
    assert manager.graph_rebuild_calls == 1
    assert manager.persist_calls == 1
    assert manager.events == [
        ("index", str(second)),
        ("graph", None),
        ("persist", None),
    ]
    assert checkpoint is not None
    assert manager.checkpoint_at_persist is not None
    assert manager.checkpoint_at_persist.complete is False
    assert checkpoint.complete is True
    assert set(checkpoint.completed) == {"first.md", "second.md"}
    assert get_bootstrap_availability(index_path) == SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="complete",
        semantic_fine="complete",
    )


def test_progressive_bootstrap_indexes_all_pending_documents_successfully(
    tmp_path: Path,
) -> None:
    """Commit every pending document through the progressive path."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First")
    second.write_text("# Second")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _save_pending_checkpoint(index_path, [first, second])

    manager = _CanonicalManager(index_path)
    receipt = run_progressive_bootstrap(
        manager,
        [str(first), str(second)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 2
    assert manager.indexed == [str(first), str(second)]
    assert manager.update_graph_flags == [False, False]


def test_progressive_bootstrap_rebuilds_graph_once_before_persisting(
    tmp_path: Path,
) -> None:
    """Defer graph work and persist only after one successful finalization."""
    document = tmp_path / "document.md"
    document.write_text("# Document")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _save_pending_checkpoint(index_path, [document])

    manager = _CanonicalManager(index_path)
    run_progressive_bootstrap(
        manager,
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert manager.graph_rebuild_calls == 1
    assert manager.events == [
        ("index", str(document)),
        ("graph", None),
        ("persist", None),
    ]


def test_progressive_bootstrap_defers_graph_until_successful_records_finish(
    tmp_path: Path,
) -> None:
    """Build the graph once after indexing both successful and failed records."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First")
    second.write_text("# Second")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _save_pending_checkpoint(index_path, [first, second])

    manager = _CanonicalManager(index_path)
    manager.failed_paths = {str(second)}
    receipt = run_progressive_bootstrap(
        manager,
        [str(first), str(second)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 1
    assert manager.update_graph_flags == [False, False]
    assert manager.events == [
        ("index", str(first)),
        ("index", str(second)),
        ("graph", None),
        ("persist", None),
    ]


def test_progressive_bootstrap_checkpoints_only_successful_records(
    tmp_path: Path,
) -> None:
    """Retain failed records for a later bootstrap attempt."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First")
    second.write_text("# Second")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _save_pending_checkpoint(index_path, [first, second])

    manager = _CanonicalManager(index_path)
    manager.failed_paths = {str(second)}
    receipt = run_progressive_bootstrap(
        manager,
        [str(first), str(second)],
        documents_roots=[tmp_path],
    )

    checkpoint = load_bootstrap_checkpoint(index_path)
    assert receipt.successful == 1
    assert receipt.ingestion.records[1].status == "failed"
    assert checkpoint is not None
    assert set(checkpoint.completed) == {"first.md"}
    assert checkpoint.complete is False


def test_progressive_bootstrap_preserves_queryable_partial_readiness(
    tmp_path: Path,
) -> None:
    """Keep partial search available while unfinished files remain."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First")
    second.write_text("# Second")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _save_pending_checkpoint(index_path, [first, second])
    partial = SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="available",
        semantic_fine="backfilling",
    )
    publish_bootstrap_availability(index_path, partial)

    manager = _CanonicalManager(index_path)
    run_progressive_bootstrap(
        manager,
        [str(first)],
        documents_roots=[tmp_path],
    )

    assert get_bootstrap_availability(index_path) == partial
