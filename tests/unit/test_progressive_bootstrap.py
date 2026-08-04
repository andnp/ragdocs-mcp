from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import pytest
from searchkernel.api import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    SearchAvailability,
    get_bootstrap_availability,
    get_semantic_completion_status,
    load_bootstrap_checkpoint,
    save_bootstrap_checkpoint,
)

from mcp_markdown_ragdocs.indexing.progressive import run_progressive_bootstrap


@dataclass
class _Fingerprint:
    namespace: str = "encoder-v1"
    model: str = "test"


class _Cache:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.get_requests: list[int] = []
        self.put_requests: list[int] = []

    def get_many(self, _hashes):
        hashes = list(_hashes)
        self.get_requests.append(len(hashes))
        return {
            content_hash: self.vectors[content_hash]
            for content_hash in hashes
            if content_hash in self.vectors
        }

    def put_many(self, vectors) -> None:
        self.put_requests.append(len(vectors))
        self.vectors.update(vectors)
        return


class _Vector:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.added: list[str] = []
        self.embedding_texts: list[str] = []
        self.fail_after = fail_after

    def get_text_embedding(self, text: str) -> list[float]:
        if self.fail_after is not None and len(self.embedding_texts) >= self.fail_after:
            raise RuntimeError("semantic work interrupted")
        self.embedding_texts.append(text)
        return [1.0]

    def add_chunk(self, chunk) -> None:
        self.added.append(chunk.chunk_id)


@dataclass
class _Chunk:
    chunk_id: str
    header_path: str
    content: str
    modified_time: datetime
    metadata: dict[str, object]
    file_path: str


class _Manager:
    def __init__(
        self,
        index_path: Path,
        events: list[str],
        *,
        with_chunk: bool = False,
        chunk_count: int = 1,
        same_content: bool = False,
        unique_chunk_ids: bool = False,
        cache: _Cache | None = None,
        fail_after: int | None = None,
        fail_persist_on: int | None = None,
    ) -> None:
        self.index_path = index_path
        self._encoder_fingerprint = _Fingerprint()
        self._embedding_cache = cache or _Cache()
        self.vector = _Vector(fail_after=fail_after)
        self.events = events
        self.with_chunk = with_chunk
        self.chunk_count = chunk_count
        self.same_content = same_content
        self.unique_chunk_ids = unique_chunk_ids
        self.fail_persist_on = fail_persist_on
        self.persist_calls = 0

    def prepare_progressive_document(self, file_path: str):
        self.events.append(f"prepare:{Path(file_path).name}")
        chunks = ()
        if self.with_chunk:
            chunks = tuple(
                _Chunk(
                    chunk_id=(
                        "chunk-1"
                        if self.chunk_count == 1 and not self.unique_chunk_ids
                        else f"{Path(file_path).stem}-chunk-{index}"
                    ),
                    header_path="Guide",
                    content=(
                        "Body"
                        if self.same_content
                        else f"Body {Path(file_path).stem} {index}"
                    ),
                    modified_time=datetime.now(UTC),
                    metadata={},
                    file_path=file_path,
                )
                for index in range(self.chunk_count)
            )
        return type(
            "Prepared",
            (),
            {"file_path": file_path, "chunks": chunks},
        )()

    def apply_progressive_lexical_graph(self, prepared_documents) -> None:
        _ = prepared_documents
        self.events.append("lexical-graph")

    def finalize_progressive_documents(self, prepared_documents) -> None:
        _ = prepared_documents
        self.events.append("finalize")

    def persist(self) -> None:
        self.persist_calls += 1
        self.events.append("persist")
        if self.fail_persist_on == self.persist_calls:
            raise RuntimeError("checkpoint persistence failed")


class _CanonicalManager(_Manager):
    def __init__(self, index_path: Path) -> None:
        super().__init__(index_path, [])
        self.kernel = object()
        self.indexed: list[str] = []
        self.graph_rebuild_calls = 0
        self.update_graph_flags: list[bool] = []

    def index_document(
        self,
        file_path: str,
        *,
        update_graph: bool = True,
    ) -> bool:
        self.indexed.append(file_path)
        self.update_graph_flags.append(update_graph)
        return True

    def rebuild_graph(self) -> None:
        self.graph_rebuild_calls += 1


def _seed_checkpoint(
    index_path: Path,
    document: Path,
    *,
    semantic_encoder_namespace: str | None = None,
    semantic_completed: dict[str, bool] | None = None,
    availability: SearchAvailability | None = None,
) -> None:
    stamp = BootstrapFileStamp(
        relative_path=document.name,
        mtime_ns=document.stat().st_mtime_ns,
        size=document.stat().st_size,
    )
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets={document.name: stamp},
            completed={},
            semantic_encoder_namespace=semantic_encoder_namespace,
            semantic_completed=semantic_completed or {},
            availability=availability,
        ),
    )


def test_canonical_bootstrap_marks_checkpoint_complete(
    tmp_path: Path,
) -> None:
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
    assert checkpoint is not None
    assert checkpoint.complete is True
    assert set(checkpoint.completed) == {"first.md", "second.md"}
    assert get_bootstrap_availability(index_path) == SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="complete",
        semantic_fine="complete",
    )


def test_progressive_bootstrap_commits_lexical_before_semantic(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    stamp = BootstrapFileStamp(
        relative_path="guide.md",
        mtime_ns=document.stat().st_mtime_ns,
        size=document.stat().st_size,
    )
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets={"guide.md": stamp},
            completed={},
        ),
    )

    events: list[str] = []
    receipt = run_progressive_bootstrap(
        _Manager(index_path, events),
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 1
    assert events.index("lexical-graph") < events.index("persist", events.index("lexical-graph"))
    assert events.index("persist", events.index("lexical-graph")) < events.index(
        "finalize"
    )
    assert get_semantic_completion_status(index_path, "encoder-v1") == {
        "guide.md": True
    }
    assert get_bootstrap_availability(index_path) == SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="complete",
        semantic_fine="complete",
    )

    events.clear()
    resumed = run_progressive_bootstrap(
        _Manager(index_path, events),
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert resumed.successful == 0
    assert events == []


def test_progressive_bootstrap_materializes_shared_semantic_vectors(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    stamp = BootstrapFileStamp(
        relative_path="guide.md",
        mtime_ns=document.stat().st_mtime_ns,
        size=document.stat().st_size,
    )
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets={"guide.md": stamp},
            completed={},
        ),
    )
    manager = _Manager(index_path, [], with_chunk=True)

    receipt = run_progressive_bootstrap(
        manager,
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 1
    assert manager.vector.added == ["chunk-1"]


def test_progressive_bootstrap_bounds_batches_and_publishes_progress(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _seed_checkpoint(index_path, document)
    cache = _Cache()
    events: list[str] = []
    manager = _Manager(
        index_path,
        events,
        with_chunk=True,
        chunk_count=130,
        cache=cache,
    )

    receipt = run_progressive_bootstrap(
        manager,
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 130
    assert cache.get_requests == [64, 64, 2]
    assert sum(cache.put_requests) == 260
    assert max(cache.put_requests) <= 32
    assert [
        (progress.batch_index, progress.stage)
        for progress in receipt.progress
    ] == [
        (0, "records"),
        (0, "semantic"),
        (1, "records"),
        (1, "semantic"),
        (2, "records"),
        (2, "semantic"),
        (2, "checkpoint"),
    ]
    assert get_bootstrap_availability(index_path) == SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="complete",
        semantic_fine="complete",
    )


def test_interrupted_semantic_batch_resumes_from_cached_work(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _seed_checkpoint(index_path, document)
    cache = _Cache()
    first_events: list[str] = []
    first = _Manager(
        index_path,
        first_events,
        with_chunk=True,
        chunk_count=66,
        cache=cache,
        fail_after=64,
    )

    with pytest.raises(Exception):
        run_progressive_bootstrap(
            first,
            [str(document)],
            documents_roots=[tmp_path],
        )

    checkpoint_after_failure = load_bootstrap_checkpoint(index_path)
    assert checkpoint_after_failure is not None
    assert checkpoint_after_failure.completed == {}
    assert checkpoint_after_failure.semantic_completed == {}
    assert len(cache.vectors) == 64
    assert sum(cache.put_requests) == 128
    assert max(cache.put_requests) <= 32

    second_events: list[str] = []
    second = _Manager(
        index_path,
        second_events,
        with_chunk=True,
        chunk_count=66,
        cache=cache,
    )
    receipt = run_progressive_bootstrap(
        second,
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 66
    assert second.vector.embedding_texts
    assert len(second.vector.embedding_texts) == 2
    assert receipt.semantic_progress[0].cache_hits == 64
    assert receipt.semantic_progress[0].cache_misses == 0
    assert receipt.semantic_progress[1].cache_misses == 2
    assert get_semantic_completion_status(index_path, "encoder-v1") == {
        "guide.md": True
    }


def test_duplicate_semantic_inputs_reuse_cache_across_files_and_restart(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# Guide")
    second.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    cache = _Cache()
    first_stamp = BootstrapFileStamp(
        "first.md",
        mtime_ns=first.stat().st_mtime_ns,
        size=first.stat().st_size,
    )
    second_stamp = BootstrapFileStamp(
        "second.md",
        mtime_ns=second.stat().st_mtime_ns,
        size=second.stat().st_size,
    )
    save_bootstrap_checkpoint(
        index_path,
        BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="generation",
            complete=False,
            targets={"first.md": first_stamp, "second.md": second_stamp},
            completed={},
        ),
    )

    manager = _Manager(
        index_path,
        [],
        with_chunk=True,
        same_content=True,
        unique_chunk_ids=True,
        cache=cache,
    )
    receipt = run_progressive_bootstrap(
        manager,
        [str(first), str(second)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 2
    assert len(manager.vector.embedding_texts) == 1
    assert receipt.semantic_progress[0].cache_misses == 1
    assert receipt.semantic_progress[1].cache_hits == 1

    second_index_path = tmp_path / "second-index"
    second_index_path.mkdir()
    _seed_checkpoint(second_index_path, first)
    restarted = _Manager(
        second_index_path,
        [],
        with_chunk=True,
        same_content=True,
        unique_chunk_ids=True,
        cache=cache,
    )
    resumed = run_progressive_bootstrap(
        restarted,
        [str(first)],
        documents_roots=[tmp_path],
    )

    assert resumed.semantic_progress[0].cache_hits == 1
    assert restarted.vector.embedding_texts == []


def test_failed_persistence_preserves_checkpoint_for_resume(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _seed_checkpoint(index_path, document)
    cache = _Cache()
    failed = _Manager(
        index_path,
        [],
        with_chunk=True,
        cache=cache,
        fail_persist_on=2,
    )

    with pytest.raises(Exception):
        run_progressive_bootstrap(
            failed,
            [str(document)],
            documents_roots=[tmp_path],
        )

    checkpoint = load_bootstrap_checkpoint(index_path)
    assert checkpoint is not None
    assert checkpoint.completed == {}
    assert checkpoint.semantic_completed == {}
    assert get_bootstrap_availability(index_path) == SearchAvailability(
        lexical="available",
        graph="available",
        semantic_coarse="backfilling",
        semantic_fine="backfilling",
    )

    resumed = _Manager(index_path, [], with_chunk=True, cache=cache)
    receipt = run_progressive_bootstrap(
        resumed,
        [str(document)],
        documents_roots=[tmp_path],
    )
    assert receipt.successful == 1
    assert get_semantic_completion_status(index_path, "encoder-v1") == {
        "guide.md": True
    }


def test_encoder_namespace_change_invalidates_previous_semantic_work(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Guide")
    index_path = tmp_path / "index"
    index_path.mkdir()
    _seed_checkpoint(
        index_path,
        document,
        semantic_encoder_namespace="old-encoder",
        semantic_completed={"guide.md": True},
    )
    manager = _Manager(index_path, [], with_chunk=True)
    manager._encoder_fingerprint.namespace = "new-encoder"

    receipt = run_progressive_bootstrap(
        manager,
        [str(document)],
        documents_roots=[tmp_path],
    )

    assert receipt.successful == 1
    assert get_semantic_completion_status(index_path, "old-encoder") == {}
    assert get_semantic_completion_status(index_path, "new-encoder") == {
        "guide.md": True
    }
