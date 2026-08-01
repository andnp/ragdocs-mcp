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
    get_semantic_completion_status,
    save_bootstrap_checkpoint,
)

from mcp_markdown_ragdocs.indexing.progressive import run_progressive_bootstrap


@dataclass
class _Fingerprint:
    namespace: str = "encoder-v1"
    model: str = "test"


class _Cache:
    def get_many(self, _hashes):
        return {}

    def put_many(self, _vectors) -> None:
        return


class _Vector:
    def __init__(self) -> None:
        self.added: list[str] = []

    def get_text_embedding(self, _text: str) -> list[float]:
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
    ) -> None:
        self.index_path = index_path
        self._encoder_fingerprint = _Fingerprint()
        self._embedding_cache = _Cache()
        self.vector = _Vector()
        self.events = events
        self.with_chunk = with_chunk

    def prepare_progressive_document(self, file_path: str):
        self.events.append(f"prepare:{Path(file_path).name}")
        chunks = (
            _Chunk(
                chunk_id="chunk-1",
                header_path="Guide",
                content="Body",
                modified_time=datetime.now(UTC),
                metadata={},
                file_path=file_path,
            ),
        ) if self.with_chunk else ()
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
        self.events.append("persist")


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
