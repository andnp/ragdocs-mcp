"""Integration coverage for the canonical record-backed index stack."""

import asyncio
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest
from searchkernel.api import GraphEdge, RecordSearchPolicy

from mcp_markdown_ragdocs.config import Config, IndexingConfig
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter
from tests.integration._canonical import make_record, make_record_index_manager


def _manager(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index")))
    )


def _resolver_supported() -> bool:
    return any(
        field.name == "graph_target_resolver"
        for field in fields(RecordSearchPolicy)
    )


@pytest.mark.asyncio
async def test_natural_language_graph_query_resolves_target_scope_and_provenance(
    tmp_path: Path,
) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    docs = tmp_path / "docs"
    docs.mkdir()
    manager = make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index"))),
        documents_roots=[docs],
    )
    target_a = replace(
        make_record(
            "target-a",
            "Hybrid Search Strategy",
            workspace_id="project-a",
            metadata={"file_path": str(docs / "target-a.md")},
        ),
        title="Hybrid Search Strategy",
    )
    target_b = replace(
        make_record(
            "target-b",
            "Hybrid Search Strategy",
            workspace_id="project-b",
            metadata={"file_path": str(docs / "target-b.md")},
        ),
        title="Hybrid Search Strategy",
    )
    neighbor_a = make_record(
        "neighbor-a",
        "Authentication notes",
        workspace_id="project-a",
        metadata={"file_path": str(docs / "neighbor-a.md")},
    )
    neighbor_b = make_record(
        "neighbor-b",
        "Authentication notes",
        workspace_id="project-b",
        metadata={"file_path": str(docs / "neighbor-b.md")},
    )
    assert manager.index_records([target_a, target_b, neighbor_a, neighbor_b])
    manager.graph.upsert_edges(
        [
            GraphEdge(target_a.identity, neighbor_a.identity, "links_to", 1.0),
            GraphEdge(target_b.identity, neighbor_b.identity, "links_to", 1.0),
        ]
    )
    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "What documents are neighbors of Hybrid Search Strategy?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )
    assert [result.file_path for result in results] == [
        str(docs / "neighbor-a.md"),
        str(docs / "target-a.md"),
    ]
    assert [result.metadata["source_id"] for result in results] == [
        "neighbor-a",
        "target-a",
    ]
    assert [result.project_id for result in results] == ["project-a", "project-a"]
    assert strategy.graph_count == 1
    assert results[0].provenance is not None
    assert "graph" in results[0].provenance.strategies


@pytest.mark.asyncio
async def test_compound_graph_query_resolves_target_with_project_scope(
    tmp_path: Path,
) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    target = make_record(
        "hybrid-search",
        "This explains Hybrid Search Strategy.",
        workspace_id="project-a",
        metadata={"file_path": str(docs / "hybrid.md")},
    )
    neighbor = make_record(
        "neighbor",
        "Neighbor explanation",
        workspace_id="project-a",
        metadata={"file_path": str(docs / "neighbor.md")},
    )
    other_neighbor = make_record(
        "other-neighbor",
        "Other project explanation",
        workspace_id="project-b",
        metadata={"file_path": str(docs / "other.md")},
    )
    assert manager.index_records([target, neighbor, other_neighbor])
    manager.graph.upsert_edges(
        [
            GraphEdge(neighbor.identity, target.identity, "links_to", 1.0),
            GraphEdge(other_neighbor.identity, target.identity, "links_to", 1.0),
        ]
    )

    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "Which documents are linked from the hybrid search strategy and what do "
        "their neighbors explain?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )

    assert {result.metadata["source_id"] for result in results} == {
        "neighbor",
        "hybrid-search",
    }
    assert all(result.project_id == "project-a" for result in results)
    assert strategy.graph_count == 1
    neighbor_result = next(
        result for result in results if result.metadata["source_id"] == "neighbor"
    )
    assert neighbor_result.provenance is not None
    assert "graph" in neighbor_result.provenance.strategies


@pytest.mark.asyncio
async def test_neighbor_query_reads_incoming_chunk_parent_neighbors(
    tmp_path: Path,
) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    docs = tmp_path / "docs"
    docs.mkdir()
    manager = make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index"))),
        documents_roots=[docs],
    )
    target_parent = make_record(
        "target_parent_0",
        "Target context",
        workspace_id="project-a",
        metadata={"doc_id": "target", "file_path": str(docs / "target.md")},
    )
    target_chunk = make_record(
        "target_chunk_26",
        "Hybrid Search Strategy",
        workspace_id="project-a",
        metadata={
            "doc_id": "target",
            "parent_chunk_id": "target_parent_0",
            "file_path": str(docs / "target.md"),
        },
    )
    source = make_record(
        "source_parent_0",
        "Authentication notes",
        workspace_id="project-a",
        metadata={"doc_id": "source", "file_path": str(docs / "source.md")},
    )
    assert manager.index_records([target_parent, target_chunk, source])
    manager.graph.upsert_edges(
        [GraphEdge(source.identity, target_parent.identity, "links_to", 1.0)]
    )

    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "What documents are neighbors of Hybrid Search Strategy?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )

    source_result = next(result for result in results if result.doc_id == "source")
    assert source_result.file_path == str(docs / "source.md")
    assert source_result.metadata["source_id"] == "source_parent_0"
    assert source_result.provenance is not None
    assert "graph" in source_result.provenance.strategies
    assert strategy.graph_count is not None
    assert strategy.graph_count >= 1


@pytest.mark.asyncio
async def test_chunk_target_resolves_parent_for_document_graph_neighbors(
    tmp_path: Path,
) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    docs = tmp_path / "docs"
    docs.mkdir()
    manager = make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index"))),
        documents_roots=[docs],
    )
    target_parent_a = make_record(
        "target-a_parent_0",
        "Target context",
        workspace_id="project-a",
        metadata={"doc_id": "target-a", "file_path": str(docs / "target-a.md")},
    )
    target_chunk_a = make_record(
        "target-a_chunk_26",
        "Hybrid Search Strategy",
        workspace_id="project-a",
        metadata={
            "doc_id": "target-a",
            "parent_chunk_id": "target-a_parent_0",
            "file_path": str(docs / "target-a.md"),
        },
    )
    target_parent_b = make_record(
        "target-b_parent_0",
        "Target context",
        workspace_id="project-b",
        metadata={"doc_id": "target-b", "file_path": str(docs / "target-b.md")},
    )
    target_chunk_b = make_record(
        "target-b_chunk_26",
        "Hybrid Search Strategy",
        workspace_id="project-b",
        metadata={
            "doc_id": "target-b",
            "parent_chunk_id": "target-b_parent_0",
            "file_path": str(docs / "target-b.md"),
        },
    )
    neighbor_a = make_record(
        "neighbor-a_parent_0",
        "Authentication notes",
        workspace_id="project-a",
        metadata={"doc_id": "neighbor-a", "file_path": str(docs / "neighbor-a.md")},
    )
    neighbor_b = make_record(
        "neighbor-b_parent_0",
        "Authentication notes",
        workspace_id="project-b",
        metadata={"doc_id": "neighbor-b", "file_path": str(docs / "neighbor-b.md")},
    )
    assert manager.index_records(
        [
            target_parent_a,
            target_chunk_a,
            target_parent_b,
            target_chunk_b,
            neighbor_a,
            neighbor_b,
        ]
    )
    manager.graph.upsert_edges(
        [
            GraphEdge(target_parent_a.identity, neighbor_a.identity, "links_to", 1.0),
            GraphEdge(target_parent_b.identity, neighbor_b.identity, "links_to", 1.0),
        ]
    )

    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "What pages does Hybrid Search Strategy link to?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )

    assert [result.doc_id for result in results] == ["target-a", "neighbor-a"]
    assert [result.metadata["source_id"] for result in results] == [
        "target-a_chunk_26",
        "neighbor-a_parent_0",
    ]
    assert [result.project_id for result in results] == ["project-a", "project-a"]
    assert strategy.graph_count == 1
    assert results[1].provenance is not None
    assert "graph" in results[1].provenance.strategies


@pytest.mark.asyncio
async def test_outbound_query_uses_links_from_child_chunks(tmp_path: Path) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    docs = tmp_path / "docs"
    docs.mkdir()
    manager = make_record_index_manager(
        Config(indexing=IndexingConfig(documents_path=str(docs), index_path=str(tmp_path / "index"))),
        documents_roots=[docs],
    )
    target_parent = make_record(
        "target_parent_0",
        "Target context",
        workspace_id="project-a",
        metadata={"doc_id": "target", "file_path": str(docs / "target.md")},
    )
    target_chunk = make_record(
        "target_chunk_26",
        "Hybrid Search Strategy",
        workspace_id="project-a",
        metadata={
            "doc_id": "target",
            "parent_chunk_id": "target_parent_0",
            "file_path": str(docs / "target.md"),
            "links": ["neighbor-a.md"],
        },
    )
    neighbor = make_record(
        "neighbor-a",
        "Authentication notes",
        workspace_id="project-a",
        metadata={"file_path": str(docs / "neighbor-a.md")},
    )
    assert manager.index_records([target_parent, target_chunk, neighbor])
    manager.rebuild_graph()

    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "What pages does Hybrid Search Strategy link to?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )

    neighbor_result = next(result for result in results if result.doc_id == "neighbor-a")
    assert neighbor_result.file_path == str(docs / "neighbor-a.md")
    assert neighbor_result.provenance is not None
    assert "graph" in neighbor_result.provenance.strategies
    assert strategy.graph_count is not None
    assert strategy.graph_count >= 1


@pytest.mark.asyncio
async def test_natural_language_graph_query_has_empty_neighbor_lane(tmp_path: Path) -> None:
    if not _resolver_supported():
        pytest.skip("requires Searchkernel graph_target_resolver support")

    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    isolated_parent = make_record(
        "isolated_parent_0",
        "Isolated context",
        workspace_id="project-a",
        metadata={"doc_id": "isolated", "file_path": str(docs / "isolated.md")},
    )
    isolated_chunk = make_record(
        "isolated_chunk_26",
        "Isolated Target",
        workspace_id="project-a",
        metadata={
            "doc_id": "isolated",
            "parent_chunk_id": "isolated_parent_0",
            "file_path": str(docs / "isolated.md"),
        },
    )
    assert manager.index_records([isolated_parent, isolated_chunk])

    results, _, strategy = await CanonicalSearchAdapter(manager).query(
        "What pages does Isolated Target link to?",
        top_k=20,
        top_n=5,
        project_filter=["project-a"],
    )

    assert [result.file_path for result in results] == [str(docs / "isolated.md")]
    assert results[0].metadata["source_id"] == "isolated_chunk_26"
    assert strategy.graph_count == 0
    assert results[0].provenance is not None
    assert "graph" not in results[0].provenance.strategies


def test_reverse_graph_scan_batches_large_identity_sets(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    records = [
        make_record(
            f"record-{index}",
            f"Indexed record {index}",
            workspace_id="project-a",
            metadata={"file_path": str(tmp_path / "docs" / f"{index}.md")},
        )
        for index in range(1_005)
    ]
    source = records[0]
    target = records[-1]
    assert manager.index_records(records)
    manager.graph.upsert_edges(
        [GraphEdge(source.identity, target.identity, "links_to", 1.0)]
    )

    graph_store = cast(Any, manager.kernel.pipeline._graph_store)
    graph_store._identities = lambda: pytest.fail("native reverse traversal was skipped")
    incoming = graph_store.incoming_neighbors_many(
        [target.identity],
        depth=1,
    )

    assert [neighbor.identity.source_id for neighbor in incoming[target.storage_key]] == [
        source.source_id
    ]


def test_index_document_updates_record_stores(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "sample.md"
    document.write_text("# Sample\n\nSearchable record content.")

    assert manager.index_document(str(document))
    assert manager.get_document_count() == 1
    assert manager.count_records("note") >= 1
    assert manager.keyword.search("searchable record", 5)
    assert manager.vector.search(
        manager.embedding_provider.embed(["searchable record"])[0],
        5,
        model_name=manager.embedding_provider.model_name,
        dim=manager.embedding_provider.dim,
    )


def test_index_document_preserves_heading_retrieval_fields(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "headings.md"
    document.write_text("# Guide\n\n## Authentication\n\nUse bearer tokens.")

    assert manager.index_document(str(document))

    records = [
        manager.kernel.backend.hydrate_record(key)
        for key in manager._source_records[manager.prepare_document(str(document)).document.id]
    ]
    authentication = next(
        record for record in records if record and record.metadata.get("header_path") == "Guide > Authentication"
    )

    assert authentication.title == "Guide > Authentication"
    assert authentication.indexed_text == authentication.body


def test_remove_document_removes_record_from_all_local_stores(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "remove.md"
    document.write_text("# Remove\n\nContent to remove.")
    manager.index_document(str(document))
    doc_id = str(manager.describe_documents()[0]["doc_id"])

    manager.remove_document(doc_id)

    assert manager.get_document_count() == 0
    assert manager.count_records("note") == 0
    assert manager.keyword.search("content", 5) == []


def test_links_are_written_to_canonical_graph_store(tmp_path):
    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    target = docs / "target.md"
    source = docs / "source.md"
    target.write_text("# Target\n\nTarget content.")
    source.write_text("# Source\n\nSee [target](target.md).")
    manager.index_document(str(target))
    manager.index_document(str(source))

    assert manager.graph.graph_integrity_errors() == []


def test_markdown_links_retrieve_indexed_target_chunks(tmp_path):
    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    source = docs / "source.md"
    target = docs / "target.md"
    source.write_text("# Source\n\nSee [target](target.md).")
    target.write_text(
        "# Target\n\n"
        "This target has enough content to produce a canonical chunk identity."
    )

    manager.index_documents([str(source), str(target)])

    source_record = manager.prepare_document(str(source)).records[0]
    neighbors = manager.graph.neighbors(source_record.identity)

    assert neighbors
    assert all(neighbor.identity.source_id.startswith("target_chunk_") for neighbor in neighbors)
    assert manager.graph.graph_integrity_errors() == []


def test_root_relative_markdown_links_retrieve_graph_neighbors(tmp_path):
    manager = _manager(tmp_path)
    docs = Path(manager._config.indexing.documents_path)
    source = docs / "nested" / "source.md"
    target = docs / "target.md"
    source.parent.mkdir()
    source.write_text("# Source\n\nSee [target](/target.md).")
    target.write_text("# Target\n\nRoot-relative graph target content.")

    manager.index_documents([str(source), str(target)])

    source_record = manager.prepare_document(str(source)).records[0]
    assert manager.graph.neighbors(source_record.identity)
    results, _, stats = asyncio.run(
        CanonicalSearchAdapter(manager, documents_path=docs).query(
            "what links to target",
            top_k=20,
            top_n=5,
        )
    )
    assert [result.file_path for result in results] == [str(source)]
    assert results[0].metadata["source_id"] == "nested/source_chunk_0"
    assert results[0].provenance is not None
    assert "graph" in results[0].provenance.strategies
    assert stats.graph_count == 1


def test_empty_document_does_not_break_record_manager(tmp_path):
    manager = _manager(tmp_path)
    document = Path(manager._config.indexing.documents_path) / "empty.md"
    document.write_text("")

    assert manager.index_document(str(document))
    assert manager.is_ready()
