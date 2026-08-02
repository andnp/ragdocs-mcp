"""Git commit content source adapter.

Implements ContentSource for git repositories, yielding commits as Records
suitable for indexing into the kernel's vector/keyword stores.
"""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import ChangeSignal, Cursor, Record, RecordStatus

from mcp_markdown_ragdocs.git.commit_chunker import chunk_commit
from mcp_markdown_ragdocs.git.commit_parser import (
    CommitData,
    parse_commit,
    parse_commits,
)
from mcp_markdown_ragdocs.git.repository import get_commits_after_timestamp

logger = logging.getLogger(__name__)


class GitContentSource:
    """Ingestible content source for git commits.

    Yields records representing commits from a git repository, with full
    commit metadata and diff summary suitable for embedding and indexing.
    """

    source_kind: str = "git_commit"

    def __init__(self, git_dir: Path):
        """
        Initialize a git content source.

        Args:
            git_dir: Path to the .git directory of the repository.

        Raises:
            ValueError: If git_dir does not exist or is not a .git directory.
        """
        self.git_dir = Path(git_dir).resolve()

        if not self.git_dir.is_dir():
            raise ValueError(f"Git directory does not exist: {self.git_dir}")

        if self.git_dir.name != ".git":
            raise ValueError(
                f"Expected .git directory, got: {self.git_dir.name}. "
                f"Did you mean {self.git_dir / '.git'}?"
            )

        self.repo_path = self.git_dir.parent

    def iter_records(self, since: Cursor | None = None) -> Iterable[Record]:
        """
        Iterate over commits to ingest, optionally since a cursor.

        If since is provided, it should be a Unix timestamp (as a string).
        Only commits after this timestamp will be yielded.

        Args:
            since: Optional watermark (Unix timestamp as string, or None for all commits).

        Yields:
            Records representing commits, ready for chunking and indexing.
        """
        # Parse since as an optional timestamp
        after_timestamp: int | None = None
        if since is not None:
            try:
                after_timestamp = int(since)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid cursor format: {since}. Expected Unix timestamp. "
                    "Falling back to all commits."
                )
                after_timestamp = None

        # Get commit hashes (newest first)
        commit_hashes = get_commits_after_timestamp(
            self.git_dir,
            after_timestamp=after_timestamp,
        )

        logger.debug(f"Found {len(commit_hashes)} commits in {self.repo_path.name}")

        commit_data = parse_commits(self.git_dir, commit_hashes, max_delta_lines=200)
        for data in commit_data:
            try:
                yield from self._commit_data_to_records(data)
            except Exception:
                logger.exception(f"Failed to convert commit {data.hash[:8]} to Record")
                continue

    def change_signal(self) -> ChangeSignal:
        """
        Return change-detection signal for this source.

        Git repositories are polled for new commits on a fixed interval.

        Returns:
            A dict indicating polling-based change detection.
        """
        return {"poll_interval": 30}

    def _commit_to_record(self, commit_hash: str) -> Record | None:
        """
        Convert a single commit to a Record.

        Args:
            commit_hash: Full git commit SHA.

        Returns:
            A Record representing the commit, or None if conversion fails.
        """
        commit_data = parse_commit(self.git_dir, commit_hash, max_delta_lines=200)
        records = self._commit_data_to_records(commit_data)
        return records[0] if records else None

    def _commit_data_to_records(self, commit_data: CommitData) -> tuple[Record, ...]:
        """Convert one parsed commit into structure-aware records."""

        if not commit_data.hash:
            logger.warning("Failed to parse commit with empty hash")
            return ()

        commit_id = f"git:{commit_data.hash}"
        chunks = chunk_commit(commit_data)

        # Convert unix timestamp to datetime
        commit_dt = datetime.fromtimestamp(commit_data.timestamp, tz=UTC)

        # Build metadata dict
        metadata = {
            "author": commit_data.author,
            "committer": commit_data.committer,
            "timestamp": commit_data.timestamp,
            "files_changed": commit_data.files_changed,
            "title": commit_data.title or "(no commit message)",
            "doc_id": commit_id,
            "commit_id": commit_id,
        }

        return tuple(
            Record(
                source_kind=self.source_kind,
                source_id=f"{commit_id}:{chunk.section}:{chunk.section_index}",
                title=commit_data.title or "(no commit message)",
                body=chunk.text,
                created_at=commit_dt,
                updated_at=commit_dt,
                metadata={
                    **metadata,
                    "chunk_section": chunk.section,
                    "chunk_index": chunk.section_index,
                    "chunk_count": len(chunks),
                    "estimated_tokens": chunk.estimated_tokens,
                },
                uri=None,
                status=RecordStatus.ACTIVE,
                embedding=None,
                embedding_model=None,
            )
            for chunk in chunks
        )
