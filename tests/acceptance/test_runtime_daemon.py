from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any

import pytest
from searchkernel.api import Record, build_local_record_kernel

from mcp_markdown_ragdocs.config import (
    Config,
    IndexingConfig,
    ProjectConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.daemon.request_router import (
    DaemonRequestRouterDependencies,
    build_daemon_request_handler,
)
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from mcp_markdown_ragdocs.lifecycle import LifecycleState
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

pytestmark = pytest.mark.e2e


@dataclass
class _Runtime:
    manager: RecordIndexManager
    handler: Callable[[str, dict[str, object]], Coroutine[Any, Any, dict[str, object]]]
    documents_roots: list[Path]


class _Coordinator:
    state = LifecycleState.READY

    def request_shutdown(self) -> None:
        self.state = LifecycleState.SHUTTING_DOWN

    async def wait_ready(self, timeout: float = 60.0) -> None:
        del timeout


def _runtime(
    tmp_path: Path,
    provider,
    roots: list[Path],
    *,
    index_path: Path | None = None,
) -> _Runtime:
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    index_path = index_path or tmp_path / "index"
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(roots[0]),
            index_path=str(index_path),
        ),
        search=SearchConfig(),
        projects=[
            ProjectConfig(name=f"root-{index}", path=str(root))
            for index, root in enumerate(roots)
        ],
    )
    kernel = build_local_record_kernel(
        index_path / "index.db",
        embedding_provider=provider,
        embedding_model_name=provider.model_name,
        embedding_dim=provider.dim,
        vector_engine="exact",
    )
    manager = RecordIndexManager(
        config,
        kernel,
        provider,
        documents_roots=roots,
    )
    adapter = CanonicalSearchAdapter(manager)
    async def ensure_fresh_indices() -> None:
        return None

    context = SimpleNamespace(
        config=config,
        index_path=index_path,
        index_manager=manager,
        orchestrator=adapter,
        search_use_case=adapter.search_use_case,
        documents_roots=roots,
        git_indexing_enabled=False,
        is_ready=lambda: True,
        get_index_state=lambda: SimpleNamespace(
            status="ready",
            indexed_count=manager.get_document_count(),
            total_count=manager.get_document_count(),
            last_error=None,
            to_dict=lambda: {
                "status": "ready",
                "indexed_count": manager.get_document_count(),
                "total_count": manager.get_document_count(),
                "last_error": None,
            },
        ),
        get_total_git_commits_indexed=lambda: manager.count_records("git_commit"),
        ensure_fresh_indices=ensure_fresh_indices,
        schedule_freshness_refresh=lambda: True,
    )
    dependencies = DaemonRequestRouterDependencies(
        ctx=context,
        coordinator=_Coordinator(),
        runtime_root=tmp_path / "runtime",
        queue_db_path=tmp_path / "runtime" / "queue.db",
        socket_path=tmp_path / "runtime" / "daemon.sock",
        index_db_path=index_path / "index.db",
        get_worker_running=lambda: True,
        get_worker_pid=lambda: 1,
        build_admin_overview_payload=lambda *_args: {},
        build_index_stats_payload=lambda *_args: {},
        build_queue_status_payload=lambda **_kwargs: {},
    )
    return _Runtime(manager, build_daemon_request_handler(dependencies), roots)


def _record(source_id: str, body: str, file_path: Path) -> Record:
    now = datetime(2026, 8, 2, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=now,
        updated_at=now,
        metadata={
            "doc_id": source_id,
            "chunk_id": f"{source_id}:chunk",
            "file_path": str(file_path),
        },
    )


async def _query(runtime: _Runtime, query: str) -> dict[str, object]:
    response = await runtime.handler("/api/search/query", {"query": query, "top_n": 5})
    assert response["query"] == query
    return response


def _result_doc_id(payload: dict[str, object]) -> str:
    results = payload.get("results")
    assert isinstance(results, list) and results
    result = results[0]
    assert isinstance(result, dict)
    doc_id = result.get("doc_id")
    assert isinstance(doc_id, str)
    return doc_id


@pytest.mark.asyncio
async def test_ingest_reaches_http_and_mcp_query_paths(
    tmp_path: Path,
    deterministic_embedding_provider,
) -> None:
    docs = tmp_path / "docs"
    runtime = _runtime(tmp_path, deterministic_embedding_provider, [docs])
    payload = await runtime.handler(
        "/api/index/records",
        {
            "records": [
                _record(
                    "ingested-auth",
                    "Bearer tokens protect authenticated API endpoints.",
                    docs / "auth.md",
                ).to_dict()
            ]
        },
    )
    assert payload == {"status": "ok", "indexed_count": 1}

    http_payload = await _query(runtime, "authenticated API endpoints")
    assert _result_doc_id(http_payload) == "ingested-auth"

    mcp_payload = await runtime.handler(
        "/api/mcp/tool",
        {"name": "query_documents", "arguments": {"query": "Bearer tokens"}},
    )
    contents = mcp_payload.get("contents")
    assert isinstance(contents, list) and contents
    first_content = contents[0]
    assert isinstance(first_content, dict)
    text = first_content.get("text")
    assert isinstance(text, str)
    assert json.loads(text)["results"][0]["doc_id"] == "ingested-auth"


@pytest.mark.asyncio
async def test_persisted_records_survive_worker_restart(
    tmp_path: Path,
    deterministic_embedding_provider,
) -> None:
    docs = tmp_path / "docs"
    first = _runtime(tmp_path, deterministic_embedding_provider, [docs])
    assert first.manager.index_record(
        _record("restart-me", "Worker restart persistence marker.", docs / "one.md")
    )
    first.manager.persist()

    restarted = _runtime(
        tmp_path,
        deterministic_embedding_provider,
        [docs],
        index_path=tmp_path / "index",
    )
    response = await _query(restarted, "persistence marker")
    assert _result_doc_id(response) == "restart-me"


@pytest.mark.asyncio
async def test_deletion_and_multi_root_reconciliation_remove_stale_results(
    tmp_path: Path,
    deterministic_embedding_provider,
) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    (root_a / "alpha.md").parent.mkdir(parents=True)
    (root_a / "alpha.md").write_text("# Alpha\n\nalpha root marker")
    (root_b / "beta.md").parent.mkdir(parents=True)
    (root_b / "beta.md").write_text("# Beta\n\nbeta root marker")
    runtime = _runtime(tmp_path, deterministic_embedding_provider, [root_a, root_b])
    runtime.manager.index_documents(
        [str(root_a / "alpha.md"), str(root_b / "beta.md")],
        persist=True,
    )

    alpha_results = (await _query(runtime, "alpha root marker")).get("results")
    beta_results = (await _query(runtime, "beta root marker")).get("results")
    assert isinstance(alpha_results, list) and alpha_results
    assert isinstance(beta_results, list) and beta_results

    alpha_id = runtime.manager._doc_id_for_path(str(root_a / "alpha.md"))
    runtime.manager.remove_document(alpha_id)
    alpha_results = (await _query(runtime, "alpha root marker")).get("results")
    assert isinstance(alpha_results, list)
    assert all(
        isinstance(result, dict) and result.get("doc_id") != alpha_id
        for result in alpha_results
    )

    (root_b / "beta.md").unlink()
    runtime.manager.reconcile_indices([str(root_a / "alpha.md")], root_a)
    beta_results = (await _query(runtime, "beta root marker")).get("results")
    assert isinstance(beta_results, list)
    assert all(
        isinstance(result, dict)
        and result.get("file_path") != str(root_b / "beta.md")
        for result in beta_results
    )
