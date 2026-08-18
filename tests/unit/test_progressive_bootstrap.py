from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.api import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    SearchAvailability,
    get_bootstrap_availability,
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
