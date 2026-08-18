from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from mcp_markdown_ragdocs.config import Config, IndexingConfig, LLMConfig, SearchConfig
from mcp_markdown_ragdocs.daemon.admin_payloads import _build_per_root_index_rows
from tests.integration._canonical import make_record, make_record_index_manager


def test_index_stats_keeps_global_records_unattributed(tmp_path: Path) -> None:
    documents_root = tmp_path / "docs"
    documents_root.mkdir()
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(documents_root),
            index_path=str(tmp_path / "index"),
        ),
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="local"),
    )
    manager = make_record_index_manager(config)
    assert manager.index_records(
        [
            make_record(
                "inside",
                "Indexed documentation",
                metadata={
                    "doc_id": "inside",
                    "file_path": str(documents_root / "inside.md"),
                },
            ),
            make_record("global", "Global record", metadata={"doc_id": "global"}),
            make_record(
                "outside",
                "External record",
                metadata={
                    "doc_id": "outside",
                    "file_path": str(tmp_path / "outside.md"),
                },
            ),
        ]
    )

    rows, unattributed_documents, unattributed_chunks = _build_per_root_index_rows(
        cast(Any, SimpleNamespace(documents_roots=[documents_root], index_manager=manager)),
        discovered_files=[],
        common_root=documents_root,
        include_indexed_estimates=True,
    )

    assert rows[0]["indexed_documents_estimate"] == 1
    assert rows[0]["indexed_chunks_estimate"] == 1
    assert unattributed_documents == 2
    assert unattributed_chunks == 2
    assert all(
        record.workspace_id is None
        for record in manager.storage.iter_records()
    )
