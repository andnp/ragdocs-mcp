import subprocess
import time
from pathlib import Path

import pytest

from src.git.commit_indexer import CommitIndexer
from src.git.commit_parser import parse_commit, build_commit_document
from src.git.parallel_indexer import (
    ParallelIndexingConfig,
    add_commits_batch,
    batch_embed_texts,
    index_commits_parallel_sync,
    parse_commits_parallel,
)
from src.git.repository import get_commits_after_timestamp


def _init_git_repo(path: Path):
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _create_commit(
    repo_path: Path,
    file_name: str,
    content: str,
    message: str,
):
    file_path = repo_path / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_parse_commits_parallel(tmp_path):
    """
    Verify that parallel parsing produces correct commit data.
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    commit_hashes = []
    for i in range(5):
        h = _create_commit(
            repo_path,
            f"file_{i}.py",
            f"def func_{i}(): pass",
            f"Add feature {i}",
        )
        commit_hashes.append(h)

    git_dir = repo_path / ".git"

    commits = parse_commits_parallel(
        git_dir,
        commit_hashes,
        max_delta_lines=200,
        max_workers=2,
    )

    assert len(commits) == 5

    for commit in commits:
        assert commit.hash in commit_hashes
        assert "Add feature" in commit.title


def test_batch_embed_texts(shared_embedding_model, tmp_path):
    """
    Verify that batch embedding produces correct number of embeddings.
    """
    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    texts = [f"Document content {i}" for i in range(10)]

    embeddings = batch_embed_texts(indexer, texts, batch_size=3)

    assert len(embeddings) == 10
    for emb in embeddings:
        assert len(emb) > 0


def test_add_commits_batch(tmp_path, shared_embedding_model):
    """
    Verify that bulk insert correctly stores commits.
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    h = _create_commit(repo_path, "file.py", "content", "Test commit")
    git_dir = repo_path / ".git"

    commit_data = parse_commit(git_dir, h, max_delta_lines=200)
    doc = build_commit_document(commit_data)

    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    embeddings = batch_embed_texts(indexer, [doc], batch_size=32)

    pairs = [(commit_data, embeddings[0])]
    inserted = add_commits_batch(indexer, pairs, str(repo_path))

    assert inserted == 1
    assert indexer.get_total_commits() == 1


def test_index_commits_parallel_sync(tmp_path, shared_embedding_model):
    """
    Full integration test for parallel sync indexing.
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    for i in range(20):
        _create_commit(
            repo_path,
            f"file_{i}.py",
            f"def func_{i}(): pass",
            f"Feature {i}",
        )

    git_dir = repo_path / ".git"
    commit_hashes = get_commits_after_timestamp(git_dir, after_timestamp=None)
    assert len(commit_hashes) == 20

    db_path = tmp_path / "commits.db"
    indexer = CommitIndexer(db_path=db_path, embedding_model=shared_embedding_model)

    config = ParallelIndexingConfig(
        max_workers=2,
        batch_size=10,
        embed_batch_size=5,
    )

    indexed = index_commits_parallel_sync(
        commit_hashes,
        git_dir,
        indexer,
        config,
        max_delta_lines=200,
    )

    assert indexed == 20
    assert indexer.get_total_commits() == 20


@pytest.mark.serial  # Performance benchmark with timing, needs isolated CPU
def test_parallel_indexing_performance_vs_serial(tmp_path, shared_embedding_model):
    """
    Compare parallel vs serial indexing performance.

    Parallel should be faster than serial for larger commit counts.
    """
    repo_path = tmp_path / "perf_repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)

    num_commits = 30
    for i in range(num_commits):
        _create_commit(
            repo_path,
            f"file_{i}.py",
            f"def process_{i}():\n    return {i}\n",
            f"Add feature {i}",
        )

    git_dir = repo_path / ".git"
    commit_hashes = get_commits_after_timestamp(git_dir, after_timestamp=None)

    db_path_serial = tmp_path / "serial.db"
    indexer_serial = CommitIndexer(
        db_path=db_path_serial, embedding_model=shared_embedding_model
    )

    start_serial = time.time()
    for commit_hash in commit_hashes:
        commit_data = parse_commit(git_dir, commit_hash, max_delta_lines=200)
        doc = build_commit_document(commit_data)
        indexer_serial.add_commit(
            hash=commit_data.hash,
            timestamp=commit_data.timestamp,
            author=commit_data.author,
            committer=commit_data.committer,
            title=commit_data.title,
            message=commit_data.message,
            files_changed=commit_data.files_changed,
            delta_truncated=commit_data.delta_truncated,
            commit_document=doc,
            repo_path=str(git_dir),
        )
    serial_time = time.time() - start_serial

    db_path_parallel = tmp_path / "parallel.db"
    indexer_parallel = CommitIndexer(
        db_path=db_path_parallel, embedding_model=shared_embedding_model
    )

    config = ParallelIndexingConfig(
        max_workers=4,
        batch_size=15,
        embed_batch_size=10,
    )

    start_parallel = time.time()
    indexed = index_commits_parallel_sync(
        commit_hashes,
        git_dir,
        indexer_parallel,
        config,
        max_delta_lines=200,
    )
    parallel_time = time.time() - start_parallel

    assert indexed == num_commits
    assert indexer_serial.get_total_commits() == num_commits
    assert indexer_parallel.get_total_commits() == num_commits

    print(f"\nSerial time: {serial_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
