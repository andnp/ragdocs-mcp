"""Incremental link-graph maintenance must match a rebuild from scratch."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from searchkernel.api import LocalRecordKernel, build_local_record_kernel

from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from tests.conftest import make_test_config
from tests.unit.test_link_graph_resolution import _linked_doc_ids, _manager

_DOC_NAMES = tuple(f"doc{index}" for index in range(8))


@dataclass
class _Doc:
    """The mutable shape of one Markdown file in the property test."""

    links: list[str] = field(default_factory=list)
    padding: int = 0


def _write_doc(root: Path, name: str, doc: _Doc) -> str:
    body = "\n\n".join(f"[link]({target}.md)" for target in doc.links)
    filler = "".join(
        f"## section {index}\n\n" + "lorem ipsum dolor sit amet " * 80 + "\n\n"
        for index in range(doc.padding)
    )
    path = root / f"{name}.md"
    path.write_text(f"# {name}\n\n{body}\n\n{filler}\n")
    return str(path)


def _fresh_manager(home: Path, root: Path, provider) -> RecordIndexManager:
    home.mkdir(parents=True, exist_ok=True)
    kernel: LocalRecordKernel = build_local_record_kernel(
        home / "records.db",
        embedding_provider=provider,
        embedding_model_name=provider.model_name,
        embedding_dim=provider.dim,
        vector_engine="exact",
    )
    return RecordIndexManager(
        make_test_config(home),
        kernel,
        provider,
        documents_roots=[root],
    )


def _link_edges(manager: RecordIndexManager) -> set[tuple[str, str]]:
    """Every stored ``links_to`` edge, as canonical storage-key pairs."""
    edges: set[tuple[str, str]] = set()
    for identity in list(manager.storage.iter_identities()):
        for neighbor in manager.graph.neighbors(identity):
            if neighbor.edge_type == "links_to":
                edges.add((identity.storage_key, neighbor.identity.storage_key))
    return edges


def test_removing_a_link_removes_its_edge(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    root = tmp_path / "root"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "target.md").write_text("# Target\n")
    src = root / "source.md"
    src.write_text("# Source\n\n[go](notes/target.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents([str(src), str(root / "notes" / "target.md")], force=True)
    manager.persist()
    assert "notes/target" in _linked_doc_ids(manager, "source")

    src.write_text("# Source\n\nno links any more\n")
    manager.index_documents([str(src)], force=True)
    manager.persist()

    assert "notes/target" not in _linked_doc_ids(manager, "source")


def test_a_link_added_later_gains_its_edge(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    root = tmp_path / "root"
    root.mkdir()
    (root / "target.md").write_text("# Target\n")
    src = root / "source.md"
    src.write_text("# Source\n\nno links yet\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents([str(src), str(root / "target.md")], force=True)
    manager.persist()
    assert _linked_doc_ids(manager, "source") == set()

    src.write_text("# Source\n\n[go](target.md)\n")
    manager.index_documents([str(src)], force=True)
    manager.persist()

    assert "target" in _linked_doc_ids(manager, "source")


def test_a_link_to_a_not_yet_indexed_document_resolves_when_it_arrives(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    """The source is untouched, so only the reverse index can find it again."""
    root = tmp_path / "root"
    root.mkdir()
    src = root / "source.md"
    src.write_text("# Source\n\n[go](target.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents([str(src)], force=True)
    manager.persist()
    assert _linked_doc_ids(manager, "source") == set()

    (root / "target.md").write_text("# Target\n")
    manager.index_documents([str(root / "target.md")], force=True)
    manager.persist()

    assert "target" in _linked_doc_ids(manager, "source")


def test_a_retargeted_link_moves_its_edge(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    root = tmp_path / "root"
    root.mkdir()
    (root / "first.md").write_text("# First\n")
    (root / "second.md").write_text("# Second\n")
    src = root / "source.md"
    src.write_text("# Source\n\n[go](first.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents(
        [str(src), str(root / "first.md"), str(root / "second.md")], force=True
    )
    manager.persist()
    assert _linked_doc_ids(manager, "source") == {"first"}

    src.write_text("# Source\n\n[go](second.md)\n")
    manager.index_documents([str(src)], force=True)
    manager.persist()

    assert _linked_doc_ids(manager, "source") == {"second"}


def test_a_rechunked_target_keeps_the_link_pointing_at_it(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    """New chunk records mean new storage keys, so the source must be redone."""
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.md"
    target.write_text("# Target\n\nshort\n")
    src = root / "source.md"
    src.write_text("# Source\n\n[go](target.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents([str(src), str(target)], force=True)
    manager.persist()
    original_keys = set(manager._source_records["target"])

    target.write_text(
        "# Target\n\n"
        + "".join(
            f"## section {index}\n\n" + "lorem ipsum dolor sit amet " * 80 + "\n\n"
            for index in range(4)
        )
    )
    manager.index_documents([str(target)], force=True)
    manager.persist()

    assert set(manager._source_records["target"]) != original_keys
    assert _linked_doc_ids(manager, "source") == {"target"}


def test_removing_a_target_document_removes_the_edges_aimed_at_it(
    tmp_path, local_record_kernel, deterministic_embedding_provider
):
    root = tmp_path / "root"
    root.mkdir()
    (root / "target.md").write_text("# Target\n")
    src = root / "source.md"
    src.write_text("# Source\n\n[go](target.md)\n")

    manager = _manager(
        tmp_path, local_record_kernel, deterministic_embedding_provider, [root]
    )
    manager.index_documents([str(src), str(root / "target.md")], force=True)
    manager.persist()
    assert _linked_doc_ids(manager, "source") == {"target"}

    manager.remove_documents(["target"], persist=True)

    assert _link_edges(manager) == set()


def test_incremental_maintenance_matches_a_rebuild_from_scratch(
    tmp_path, deterministic_embedding_provider
):
    """A randomized mutation sequence must leave both graphs edge-identical."""
    root = tmp_path / "root"
    root.mkdir()
    random_source = random.Random(20260818)
    docs: dict[str, _Doc] = {
        "doc0": _Doc(links=["doc1", "doc7"]),
        "doc1": _Doc(links=["doc2"]),
        "doc2": _Doc(),
    }
    for name, doc in docs.items():
        _write_doc(root, name, doc)

    incremental = _fresh_manager(
        tmp_path / "incremental", root, deterministic_embedding_provider
    )
    incremental.index_documents(
        [str(root / f"{name}.md") for name in docs], force=True
    )
    incremental.persist()

    for step in range(24):
        changed: set[str] = set()
        removed: set[str] = set()
        operations = ["add_doc", "add_link", "remove_link", "retarget_link", "rechunk"]
        if len(docs) > 1:
            operations.append("remove_doc")
        operation = random_source.choice(operations)
        linked = [name for name, doc in docs.items() if doc.links]

        if operation == "add_doc":
            missing = [name for name in _DOC_NAMES if name not in docs]
            if missing:
                name = random_source.choice(missing)
                docs[name] = _Doc(
                    links=random_source.sample(
                        list(_DOC_NAMES), random_source.randint(0, 2)
                    )
                )
                changed.add(name)
        elif operation == "remove_doc":
            name = random_source.choice(list(docs))
            del docs[name]
            (root / f"{name}.md").unlink()
            removed.add(name)
        elif operation == "add_link" or not linked:
            name = random_source.choice(list(docs))
            docs[name].links.append(random_source.choice(list(_DOC_NAMES)))
            changed.add(name)
        elif operation == "remove_link":
            name = random_source.choice(linked)
            docs[name].links.pop(random_source.randrange(len(docs[name].links)))
            changed.add(name)
        elif operation == "retarget_link":
            name = random_source.choice(linked)
            index = random_source.randrange(len(docs[name].links))
            docs[name].links[index] = random_source.choice(list(_DOC_NAMES))
            changed.add(name)
        else:
            name = random_source.choice(list(docs))
            docs[name].padding = (docs[name].padding + 3) % 9
            changed.add(name)

        for name in changed:
            _write_doc(root, name, docs[name])
        if removed:
            incremental.remove_documents(sorted(removed))
        if changed:
            incremental.index_documents(
                [str(root / f"{name}.md") for name in sorted(changed)], force=True
            )
        incremental.persist()

        scratch = _fresh_manager(
            tmp_path / f"scratch{step}", root, deterministic_embedding_provider
        )
        scratch.index_documents(
            [str(root / f"{name}.md") for name in sorted(docs)], force=True
        )
        scratch.rebuild_graph()
        scratch.persist()

        assert _link_edges(incremental) == _link_edges(scratch), (
            f"step {step} ({operation}) diverged"
        )
