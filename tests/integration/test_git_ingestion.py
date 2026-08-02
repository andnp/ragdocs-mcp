"""End-to-end test: a real git commit becomes discoverable through
SearchOrchestrator via the git_ingestion wiring, scoped by source_filter.

This is the W2b port proof: GitContentSource -> IndexManager.index_record ->
the shared vector/keyword store -> CanonicalSearchAdapter.query(source_filter=
["git_commit"]), with zero core search changes beyond source_filter itself.
"""

import subprocess
from pathlib import Path

import pytest
from mcp_markdown_ragdocs.search import CanonicalSearchAdapter

from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
from mcp_markdown_ragdocs.config import (
    ChunkingConfig,
    Config,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
)
from tests.integration._canonical import make_record_index_manager


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


@pytest.fixture
def kernel(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        indexing=IndexingConfig(
            documents_path=str(docs_dir), index_path=str(tmp_path / ".index_data")
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

    manager = make_record_index_manager(config)
    orchestrator = CanonicalSearchAdapter(manager)
    return manager, orchestrator


@pytest.mark.asyncio
async def test_commit_appears_in_search_via_source_filter(repo, kernel):
    manager, orchestrator = kernel

    source = GitContentSource(repo / ".git")
    records = list(source.iter_records())
    assert manager.index_records(records)
    assert manager.count_records("git_commit") == 1

    results, _, _ = await orchestrator.query(
        "authentication token refresh bug",
        top_k=10,
        top_n=5,
        source_filter=["git_commit"],
    )

    assert results
    assert all(result.doc_id.startswith("git:") for result in results)
    assert any("authentication" in result.content.lower() for result in results)
    assert all(result.metadata.get("source_kind") == "git_commit" for result in results)
    assert all(result.metadata.get("author") for result in results)


@pytest.mark.asyncio
async def test_commit_absent_when_source_filter_excludes_git(repo, kernel):
    manager, orchestrator = kernel

    source = GitContentSource(repo / ".git")
    assert manager.index_records(list(source.iter_records()))

    results, _, _ = await orchestrator.query(
        "authentication token refresh bug",
        top_k=10,
        top_n=5,
        source_filter=["note"],
    )

    assert results == []
