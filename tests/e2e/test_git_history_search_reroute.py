"""E2E proof that search_git_history is rerouted through SearchOrchestrator.

W2b acceptance: the MCP search_git_history tool must answer from the unified
kernel index (SearchOrchestrator.query with source_filter=["git_commit"]),
not the deleted commit_indexer/commit_search stack.
"""

import subprocess
from pathlib import Path

import pytest
from searchkernel.indexing.git_ingestion import ingest_git_source
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.search.orchestrator import SearchOrchestrator

from ragdocs.adapters.sources.git import GitContentSource
from ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from ragdocs.indexing.manager import IndexManager
from ragdocs.mcp.handlers import HandlerContext
from ragdocs.mcp.tools.document_tools import handle_search_git_history


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
        self._total_commits = total_commits

    def is_ready(self) -> bool:
        return True

    def get_total_git_commits_indexed(self) -> int:
        return self._total_commits

    def get_nonblocking_search_payload(self, *, query, include_git_metadata=False):
        return None


class _FakeCoordinator:
    async def wait_ready(self, timeout: float = 60.0) -> None:
        return None


@pytest.mark.asyncio
async def test_search_git_history_tool_routes_through_orchestrator(repo, shared_embedding_model):
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
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

    source = GitContentSource(repo / ".git")
    ingested = ingest_git_source(manager, source)
    assert ingested == 1

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
async def test_search_git_history_tool_reports_unavailable_when_disabled(shared_embedding_model):
    config = Config(
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="local"),
    )
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

    ctx = _ReadyContext(orchestrator, git_indexing_enabled=False, total_commits=0)
    hctx = HandlerContext(lambda: ctx, _FakeCoordinator())

    contents = await handle_search_git_history(hctx, {"query": "anything"})

    assert len(contents) == 1
    assert "not available" in contents[0].text
