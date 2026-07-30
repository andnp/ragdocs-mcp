"""Unit tests for indexing stages and batch preparation."""

from datetime import UTC, datetime

from searchkernel.domain import Chunk, Record, RecordStatus
from searchkernel.indexing.stages import (
    GraphStage,
    KeywordStage,
    PreparedIndexBatch,
    PreparedIndexDocument,
    SemanticStage,
    iter_prepared_index_batches,
)


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk



def make_test_chunk(chunk_id: str = "chunk-1", doc_id: str = "doc-1") -> Chunk:
    """Create a test chunk with minimal required fields."""
    return _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content="content", metadata={ "header_path": "", "start_pos": 0, "end_pos": 7, "file_path": "/test.md", "modified_time": datetime.now(UTC)}, chunk_index=0))


def make_test_record(doc_id: str = "doc-1", file_path: str = "/test.md") -> Record:
    """Create a test record with minimal required fields."""
    now = datetime.now(UTC)
    return Record(
        source_kind="note",
        source_id=doc_id,
        title=doc_id,
        body="test content",
        created_at=now,
        updated_at=now,
        metadata={"links": [], "tags": [], "file_path": file_path},
        uri=f"file://{file_path}",
        status=RecordStatus.ACTIVE,
    )


class FakeChunkWriter:
    """Mock chunk index writer for testing."""

    def __init__(self) -> None:
        self.added_chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.added_chunks.extend(chunks)


class FakeGraphWriter:
    """Mock graph writer for testing."""

    def __init__(self) -> None:
        self.added_nodes: list[tuple[str, dict]] = []
        self.added_edges: list[tuple[str, str, str, str]] = []

    def add_nodes(self, nodes: list[tuple[str, dict]]) -> None:
        self.added_nodes.extend(nodes)

    def add_edges(self, edges: list[tuple[str, str, str, str]]) -> None:
        self.added_edges.extend(edges)


class TestPreparedIndexBatch:
    """Test batch preparation."""

    def test_prepared_batch_from_documents(self) -> None:
        record = make_test_record(doc_id="doc-1")
        chunk = make_test_chunk(chunk_id="chunk-1", doc_id="doc-1")

        prepared = PreparedIndexDocument(
            file_path="/test.md",
            parser=object(),
            record=record,
            chunks=[chunk],
            graph_metadata={"tags": []},
        )

        batch = PreparedIndexBatch.from_documents([prepared], encoder_namespace="ns-1")

        assert len(batch.documents) == 1
        assert len(batch.chunks) == 1
        assert len(batch.semantic_inputs) == 1
        assert batch.semantic_inputs[0].source_id == "chunk-1"

    def test_prepared_batch_from_multiple_documents(self) -> None:
        prepared_docs = []
        for i in range(3):
            record = make_test_record(doc_id=f"doc-{i}", file_path=f"/test-{i}.md")
            chunks = [
                make_test_chunk(
                    chunk_id=f"chunk-{i}-{j}",
                    doc_id=f"doc-{i}",
                )
                for j in range(2)
            ]

            prepared_docs.append(
                PreparedIndexDocument(
                    file_path=f"/test-{i}.md",
                    parser=object(),
                    record=record,
                    chunks=chunks,
                    graph_metadata={},
                )
            )

        batch = PreparedIndexBatch.from_documents(prepared_docs)

        assert len(batch.documents) == 3
        assert len(batch.chunks) == 6
        assert len(batch.semantic_inputs) == 6


class TestIterPreparedIndexBatches:
    """Test batch iteration with bounds."""

    def test_batch_iterator_respects_document_limit(self) -> None:
        prepared_docs = []
        for i in range(5):
            record = make_test_record(doc_id=f"doc-{i}", file_path=f"/test-{i}.md")
            chunk = make_test_chunk(chunk_id=f"chunk-{i}", doc_id=f"doc-{i}")

            prepared_docs.append(
                PreparedIndexDocument(
                    file_path=f"/test-{i}.md",
                    parser=object(),
                    record=record,
                    chunks=[chunk],
                    graph_metadata={},
                )
            )

        batches = list(iter_prepared_index_batches(prepared_docs, max_documents=2, max_chunks=100))

        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0].documents) == 2
        assert len(batches[1].documents) == 2
        assert len(batches[2].documents) == 1

    def test_batch_iterator_respects_chunk_limit(self) -> None:
        prepared_docs = []
        for i in range(3):
            record = make_test_record(doc_id=f"doc-{i}", file_path=f"/test-{i}.md")
            # Each document has 3 chunks
            chunks = [
                make_test_chunk(
                    chunk_id=f"chunk-{i}-{j}",
                    doc_id=f"doc-{i}",
                )
                for j in range(3)
            ]

            prepared_docs.append(
                PreparedIndexDocument(
                    file_path=f"/test-{i}.md",
                    parser=object(),
                    record=record,
                    chunks=chunks,
                    graph_metadata={},
                )
            )

        # Max 5 chunks per batch: 3 docs * 3 chunks each = 9 total
        # Batch 1: doc-0 (3 chunks)
        # Batch 2: doc-1 (3 chunks)  (doc-2 would exceed limit)
        # Batch 3: doc-2 (3 chunks)
        batches = list(iter_prepared_index_batches(prepared_docs, max_documents=10, max_chunks=5))

        assert len(batches) == 3
        assert len(batches[0].chunks) == 3  # 1 doc * 3 chunks
        assert len(batches[1].chunks) == 3  # 1 doc * 3 chunks
        assert len(batches[2].chunks) == 3  # 1 doc * 3 chunks

    def test_batch_iterator_raises_on_zero_bounds(self) -> None:
        with_pytest_raises = True
        try:
            import pytest
            import_pytest_error = False
        except ImportError:
            import_pytest_error = True

        if import_pytest_error:
            with_pytest_raises = False

        prepared_docs = []

        if with_pytest_raises:
            with pytest.raises(ValueError):
                list(iter_prepared_index_batches(prepared_docs, max_documents=0, max_chunks=10))
        else:
            try:
                list(iter_prepared_index_batches(prepared_docs, max_documents=0, max_chunks=10))
                assert False, "Should have raised ValueError"
            except ValueError:
                pass


class TestKeywordStage:
    """Test keyword indexing stage."""

    def test_keyword_stage_adds_chunks(self) -> None:
        writer = FakeChunkWriter()
        stage = KeywordStage(writer)

        chunk = make_test_chunk()
        batch = PreparedIndexBatch(
            documents=[],
            chunks=[chunk],
        )

        result = stage.apply(batch)

        assert result.stage == "keyword"
        assert result.counters.chunks == 1
        assert len(writer.added_chunks) == 1

    def test_keyword_stage_empty_batch(self) -> None:
        writer = FakeChunkWriter()
        stage = KeywordStage(writer)

        batch = PreparedIndexBatch(
            documents=[],
            chunks=[],
        )

        result = stage.apply(batch)

        assert result.stage == "keyword"
        assert result.counters.chunks == 0
        assert len(writer.added_chunks) == 0


class TestGraphStage:
    """Test graph indexing stage."""

    def test_graph_stage_adds_nodes_and_edges(self) -> None:
        writer = FakeGraphWriter()
        stage = GraphStage(writer)

        nodes = [
            ("doc-1", {"title": "Doc 1"}),
            ("chunk-1", {"type": "chunk"}),
        ]
        edges = [
            ("doc-1", "doc-2", "links_to", ""),
        ]
        batch = PreparedIndexBatch(
            documents=[],
            chunks=[],
            graph_nodes=nodes,
            graph_edges=edges,
        )

        result = stage.apply(batch)

        assert result.stage == "graph"
        assert result.counters.nodes == 2
        assert result.counters.edges == 1
        assert len(writer.added_nodes) == 2
        assert len(writer.added_edges) == 1

    def test_graph_stage_empty_batch(self) -> None:
        writer = FakeGraphWriter()
        stage = GraphStage(writer)

        batch = PreparedIndexBatch(
            documents=[],
            chunks=[],
            graph_nodes=[],
            graph_edges=[],
        )

        result = stage.apply(batch)

        assert result.stage == "graph"
        assert result.counters.nodes == 0
        assert result.counters.edges == 0
        assert len(writer.added_nodes) == 0
        assert len(writer.added_edges) == 0


class TestSemanticStage:
    """Test semantic stage."""

    def test_semantic_stage_adds_chunks(self) -> None:
        writer = FakeChunkWriter()
        stage = SemanticStage(writer)

        chunk = make_test_chunk()
        batch = PreparedIndexBatch(
            documents=[],
            chunks=[chunk],
        )

        result = stage.apply(batch)

        assert result.stage == "semantic"
        assert result.counters.chunks == 1
        assert len(writer.added_chunks) == 1

    def test_semantic_stage_empty_batch(self) -> None:
        writer = FakeChunkWriter()
        stage = SemanticStage(writer)

        batch = PreparedIndexBatch(
            documents=[],
            chunks=[],
        )

        result = stage.apply(batch)

        assert result.stage == "semantic"
        assert result.counters.chunks == 0
        assert len(writer.added_chunks) == 0
