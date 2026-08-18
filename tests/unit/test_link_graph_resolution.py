"""Link-to-document resolution behind the ``links_to`` graph edges."""

from searchkernel.domain import RecordIdentity

from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from tests.conftest import make_test_config


def _identities(manager: RecordIndexManager, doc_id: str) -> list[RecordIdentity]:
    # Plumbing only: translate a doc_id into the record identities the graph
    # is keyed by, so the assertions can talk about edges.
    return [
        RecordIdentity.from_storage_key(key)
        for key in manager._source_records[doc_id]
    ]


def _linked_doc_ids(manager: RecordIndexManager, source_doc_id: str) -> set[str]:
    found: set[str] = set()
    for identity in _identities(manager, source_doc_id):
        for neighbor in manager.graph.neighbors(identity):
            if neighbor.edge_type != "links_to":
                continue
            record = manager.storage.hydrate_record(neighbor.identity.storage_key)
            if record is not None:
                found.add(str(record.metadata.get("doc_id")))
    return found


def _manager(tmp_path, kernel, provider, roots):
    return RecordIndexManager(
        make_test_config(tmp_path),
        kernel,
        provider,
        documents_roots=roots,
    )


def test_relative_link_resolves_to_the_target_document(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    root = tmp_path / "root_a"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "target.md").write_text("# Target\n")
    (root / "source.md").write_text("# Source\n\n[go](notes/target.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents(
        [str(root / "source.md"), str(root / "notes" / "target.md")], force=True
    )
    manager.persist()

    assert "notes/target" in _linked_doc_ids(manager, "source")


def test_link_present_in_several_roots_resolves_to_the_last_root(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    """Pins existing precedence: later-configured roots win over earlier ones."""
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    for root in (root_a, root_b):
        (root / "notes").mkdir(parents=True)
        (root / "notes" / "target.md").write_text(f"# Target in {root.name}\n")
    (root_a / "source.md").write_text("# Source\n\n[go](notes/target.md)\n")

    manager = _manager(
        tmp_path,
        local_record_kernel,
        deterministic_embedding_provider,
        [root_a, root_b],
    )
    manager.index_documents(
        [
            str(root_a / "source.md"),
            str(root_a / "notes" / "target.md"),
            str(root_b / "notes" / "target.md"),
        ],
        force=True,
    )
    manager.persist()

    assert _linked_doc_ids(manager, "root_a/source") == {"root_b/notes/target"}
