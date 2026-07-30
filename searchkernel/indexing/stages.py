"""Small, behavior-preserving stages for bulk index construction.

The stage objects intentionally only wrap the existing index APIs.  They are
an architectural seam for later progressive indexing work, not a new
readiness or scheduling system.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from searchkernel.indexing.semantic import SemanticInput, semantic_input_for_chunk
from searchkernel.models import Chunk, Document
from searchkernel.search.edge_types import infer_edge_type


@dataclass
class PreparedIndexDocument:
    file_path: str
    parser: object
    document: Document
    chunks: list[Chunk]
    graph_metadata: dict


@dataclass
class PreparedIndexBatch:
    """Prepared documents and the bounded payloads consumed by each stage."""

    documents: list[PreparedIndexDocument]
    chunks: list[Chunk] = field(default_factory=list)
    graph_nodes: list[tuple[str, dict]] = field(default_factory=list)
    graph_edges: list[tuple[str, str, str, str]] = field(default_factory=list)
    semantic_inputs: list[SemanticInput] = field(default_factory=list)

    @classmethod
    def from_documents(
        cls,
        documents: list[PreparedIndexDocument],
        *,
        encoder_namespace: str = "",
    ) -> "PreparedIndexBatch":
        chunks = [chunk for document in documents for chunk in document.chunks]
        graph_nodes, graph_edges = build_graph_payload(documents)
        return cls(
            documents=documents,
            chunks=chunks,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            semantic_inputs=[
                semantic_input_for_chunk(
                    chunk,
                    encoder_namespace=encoder_namespace,
                )
                for chunk in chunks
            ],
        )


def build_graph_payload(
    documents: list[PreparedIndexDocument],
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, str]]]:
    """Shape document, chunk, and link data for the bulk graph APIs."""
    from searchkernel.parsers.markdown import MarkdownParser

    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, str, str]] = []
    for prepared in documents:
        nodes.append((prepared.document.id, prepared.graph_metadata))
        nodes.extend((chunk.chunk_id, chunk.metadata) for chunk in prepared.chunks)
        if isinstance(prepared.parser, MarkdownParser):
            links = prepared.parser.extract_links_with_context(prepared.file_path)
            edges.extend(
                (
                    prepared.document.id,
                    link.target,
                    infer_edge_type(link.header_context, link.target).value,
                    link.header_context,
                )
                for link in links
            )
        else:
            edges.extend(
                (prepared.document.id, link, "links_to", "")
                for link in prepared.document.links
            )
    return nodes, edges


def iter_prepared_index_batches(
    documents: list[PreparedIndexDocument],
    *,
    max_documents: int,
    max_chunks: int,
) -> Iterator[PreparedIndexBatch]:
    """Yield non-empty batches bounded by document and chunk counts."""
    if max_documents <= 0 or max_chunks <= 0:
        raise ValueError("batch bounds must be positive")

    current: list[PreparedIndexDocument] = []
    chunk_count = 0
    for document in documents:
        document_chunks = len(document.chunks)
        if current and (
            len(current) >= max_documents or chunk_count + document_chunks > max_chunks
        ):
            yield PreparedIndexBatch.from_documents(current)
            current = []
            chunk_count = 0
        current.append(document)
        chunk_count += document_chunks
    if current:
        yield PreparedIndexBatch.from_documents(current)


@dataclass(frozen=True)
class StageCounters:
    documents: int = 0
    chunks: int = 0
    nodes: int = 0
    edges: int = 0


@dataclass(frozen=True)
class StageResult:
    stage: str
    counters: StageCounters


class IndexStage(Protocol):
    name: str

    def apply(self, batch: PreparedIndexBatch) -> StageResult: ...


class ChunkIndexWriter(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...


class GraphIndexWriter(Protocol):
    def add_nodes(self, nodes: list[tuple[str, dict]]) -> None: ...

    def add_edges(self, edges: list[tuple[str, str, str, str]]) -> None: ...


class KeywordStage:
    name = "keyword"

    def __init__(self, keyword: ChunkIndexWriter) -> None:
        self._keyword = keyword

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.chunks:
            self._keyword.add_chunks(batch.chunks)
        return StageResult(
            self.name,
            StageCounters(documents=len(batch.documents), chunks=len(batch.chunks)),
        )


class GraphStage:
    name = "graph"

    def __init__(self, graph: GraphIndexWriter) -> None:
        self._graph = graph

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.graph_nodes:
            self._graph.add_nodes(batch.graph_nodes)
        if batch.graph_edges:
            self._graph.add_edges(batch.graph_edges)
        return StageResult(
            self.name,
            StageCounters(
                documents=len(batch.documents),
                chunks=len(batch.chunks),
                nodes=len(batch.graph_nodes),
                edges=len(batch.graph_edges),
            ),
        )


class SemanticStage:
    name = "semantic"

    def __init__(self, vector: ChunkIndexWriter) -> None:
        self._vector = vector

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.chunks:
            self._vector.add_chunks(batch.chunks)
        return StageResult(
            self.name,
            StageCounters(documents=len(batch.documents), chunks=len(batch.chunks)),
        )
