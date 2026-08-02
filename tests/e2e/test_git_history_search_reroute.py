"""E2E proof that search_git_history is rerouted through SearchOrchestrator.

W2b acceptance: the MCP search_git_history tool must answer from the unified
kernel index (SearchOrchestrator.query with source_filter=["git_commit"]),
not the deleted commit_indexer/commit_search stack.
"""

import subprocess
from pathlib import Path
from typing import Any

import pytest
from searchkernel.api import build_local_record_kernel
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from mcp_markdown_ragdocs.indexing.git_ingestion import iter_git_ingestion_receipts
from mcp_markdown_ragdocs.indexing.record_manager import (
    RecordIndexManager,
    build_embedding_provider,
)
from mcp_markdown_ragdocs.lifecycle import LifecycleState
from mcp_markdown_ragdocs.mcp.handlers import HandlerContext
from mcp_markdown_ragdocs.mcp.tools.document_tools import handle_search_git_history


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _commit(repo_path: Path, file_name: str, content: str, message: str) -> None:
    file_path = repo_path / file_name
    file_path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message], cwd=repo_path, check=True, capture_output=True
    )


@pytest.fixture
def repo(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)
    _commit(
        repo_path,
        "auth.py",
        "def login(): ...",
        "Fix a subtle authentication token refresh bug",
    )
    return repo_path


class _ReadyContext:
    def __init__(self, orchestrator, git_indexing_enabled: bool, total_commits: int) -> None:
        self.orchestrator = orchestrator
        self.git_indexing_enabled = git_indexing_enabled
        self.documents_roots = [Path("/docs")]
        self._total_commits = total_commits

    def is_ready(self) -> bool:
        return True

    def get_total_git_commits_indexed(self) -> int:
        return self._total_commits

    def get_index_state(self) -> Any:
        return None

    def get_nonblocking_search_payload(self, *, query, include_git_metadata=False):
        return None


class _FakeCoordinator:
    state = LifecycleState.READY

    async def wait_ready(self, timeout: float = 60.0) -> None:
        return None


def _create_record_manager(config: Config) -> RecordIndexManager:
    embedding_provider = build_embedding_provider(
        config, config.llm.resolved_embedding_model
    )
    local_kernel = build_local_record_kernel(
        Path(config.indexing.index_path) / "index.db",
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_provider.model_name,
        embedding_dim=embedding_provider.dim,
        vector_engine="exact",
    )
    return RecordIndexManager(config, local_kernel, embedding_provider)


@pytest.mark.asyncio
async def test_search_git_history_tool_routes_through_orchestrator(repo):
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(repo), index_path=str(repo.parent / ".index_data")
        ),
        search=SearchConfig(semantic_weight=1.0, keyword_weight=1.0),
        llm=LLMConfig(embedding_model="local"),
        chunking=ChunkingConfig(
            strategy="header_based",
            min_chunk_chars=50,
            max_chunk_chars=1500,
            overlap_chars=50,
        ),
    )
    manager = _create_record_manager(config)
    orchestrator = CanonicalSearchAdapter(manager)

    source = GitContentSource(repo / ".git")
    receipts = [
        receipt
        async for receipt in iter_git_ingestion_receipts(
            manager, source, since=None, batch_size=100
        )
    ]
    assert len(receipts) == 1
    assert receipts[0].committed == 1

    ctx = _ReadyContext(orchestrator, git_indexing_enabled=True, total_commits=1)
    hctx = HandlerContext(lambda: ctx, _FakeCoordinator())

    contents = await handle_search_git_history(
        hctx, {"query": "authentication token refresh bug"}
    )

    assert len(contents) == 1
    text = contents[0].text
    assert "# Git History Search Results" in text
    assert "**Total Commits Indexed:** 1" in text
    assert "**Results Returned:** 1" in text
    assert "**Author:**" in text


@pytest.mark.asyncio
async def test_search_git_history_tool_reports_unavailable_when_disabled(tmp_path):
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path), index_path=str(tmp_path / "indices")
        ),
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="local"),
    )
    manager = _create_record_manager(config)
    orchestrator = CanonicalSearchAdapter(manager)

    ctx = _ReadyContext(orchestrator, git_indexing_enabled=False, total_commits=0)
    hctx = HandlerContext(lambda: ctx, _FakeCoordinator())

    contents = await handle_search_git_history(hctx, {"query": "anything"})

    assert len(contents) == 1
    assert "not available" in contents[0].text
