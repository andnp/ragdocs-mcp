import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.git.commit_indexer import CommitIndexer
from src.git.commit_parser import CommitData, build_commit_document, parse_commit

logger = logging.getLogger(__name__)


@dataclass
class ParallelIndexingConfig:
    max_workers: int = 4
    batch_size: int = 100
    embed_batch_size: int = 32


def _parse_commit_safe(
    git_dir: Path,
    commit_hash: str,
    max_delta_lines: int,
) -> CommitData | None:
    try:
        return parse_commit(git_dir, commit_hash, max_delta_lines)
    except Exception:
        logger.warning(f"Failed to parse commit {commit_hash}", exc_info=True)
        return None


def parse_commits_parallel(
    git_dir: Path,
    commit_hashes: list[str],
    max_delta_lines: int,
    max_workers: int = 4,
) -> list[CommitData]:
    results: list[CommitData] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_commit_safe, git_dir, h, max_delta_lines): h
            for h in commit_hashes
        }

        for future in as_completed(futures):
            commit_data = future.result()
            if commit_data is not None:
                results.append(commit_data)

    return results


def batch_embed_texts(
    indexer: CommitIndexer,
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = [indexer._embedding_model.get_text_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
    return embeddings


def add_commits_batch(
    indexer: CommitIndexer,
    commits_with_embeddings: list[tuple[CommitData, list[float]]],
    repo_path: str,
) -> int:
    if not commits_with_embeddings:
        return 0

    conn = indexer._get_connection()
    indexed_at = int(time.time())
    normalized_path = indexer._normalize_repo_path(repo_path)

    rows = []
    for commit, embedding in commits_with_embeddings:
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        rows.append((
            commit.hash,
            commit.timestamp,
            commit.author,
            commit.committer,
            commit.title,
            commit.message,
            json.dumps(commit.files_changed),
            commit.delta_truncated,
            embedding_bytes,
            indexed_at,
            normalized_path,
        ))

    conn.executemany(
        """
        INSERT OR REPLACE INTO git_commits
        (hash, timestamp, author, committer, title, message,
         files_changed, delta_truncated, embedding, indexed_at, repo_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

    logger.debug(f"Bulk inserted {len(rows)} commits for {normalized_path}")
    return len(rows)


def _process_batch(
    batch_hashes: list[str],
    git_dir: Path,
    commit_indexer: CommitIndexer,
    config: ParallelIndexingConfig,
    max_delta_lines: int,
    repo_path: str,
) -> int:
    commits = parse_commits_parallel(
        git_dir,
        batch_hashes,
        max_delta_lines,
        config.max_workers,
    )

    if not commits:
        return 0

    documents = [build_commit_document(c) for c in commits]
    embeddings = batch_embed_texts(commit_indexer, documents, config.embed_batch_size)
    pairs = list(zip(commits, embeddings, strict=True))

    indexed = add_commits_batch(commit_indexer, pairs, repo_path)
    logger.debug(f"Indexed batch of {indexed} commits from {git_dir}")
    return indexed


async def index_commits_parallel(
    commit_hashes: list[str],
    git_dir: Path,
    commit_indexer: CommitIndexer,
    config: ParallelIndexingConfig,
    max_delta_lines: int = 200,
) -> int:
    if not commit_hashes:
        return 0

    total_indexed = 0
    repo_path = str(git_dir.parent)

    for i in range(0, len(commit_hashes), config.batch_size):
        batch_hashes = commit_hashes[i : i + config.batch_size]
        indexed = await asyncio.to_thread(
            _process_batch,
            batch_hashes,
            git_dir,
            commit_indexer,
            config,
            max_delta_lines,
            repo_path,
        )
        total_indexed += indexed

    return total_indexed


def index_commits_parallel_sync(
    commit_hashes: list[str],
    git_dir: Path,
    commit_indexer: CommitIndexer,
    config: ParallelIndexingConfig,
    max_delta_lines: int = 200,
) -> int:
    if not commit_hashes:
        return 0

    total_indexed = 0
    repo_path = str(git_dir.parent)

    for i in range(0, len(commit_hashes), config.batch_size):
        batch_hashes = commit_hashes[i : i + config.batch_size]
        indexed = _process_batch(
            batch_hashes,
            git_dir,
            commit_indexer,
            config,
            max_delta_lines,
            repo_path,
        )
        total_indexed += indexed

    return total_indexed
