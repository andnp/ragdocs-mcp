"""Git commit content source adapter.

Implements ContentSource for git repositories, yielding commits as Records
suitable for indexing into the kernel's vector/keyword stores.
"""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import ChangeSignal, Cursor, Record, RecordStatus

from mcp_markdown_ragdocs.git.commit_chunker import chunk_commit, is_diff_chunk_eligible
from mcp_markdown_ragdocs.git.commit_parser import (
    CommitData,
    MAX_FILES_CHANGED,
    iter_commits,
    parse_commit,
)
from mcp_markdown_ragdocs.git.repository import iter_commit_hashes_after_timestamp

logger = logging.getLogger(__name__)


class GitContentSource:
    """Ingestible content source for git commits.

    Yields records representing commits from a git repository, with full
    commit metadata and diff summary suitable for embedding and indexing.
    """

    source_kind: str = "git_commit"

    def __init__(
        self,
        git_dir: Path,
        workspace_id: str | None = None,
        *,
        git_diff_embedding_days: int = 0,
        reference_time: datetime | None = None,
    ):
        """
        Initialize a git content source.

        Args:
            git_dir: Path to the .git directory of the repository.
            git_diff_embedding_days: Skip diff-section chunking for commits
                older than this many days; 0 disables the gate.
            reference_time: "Now" used to evaluate commit age; defaults to
                the current time, computed once per instance so tests can
                pin it instead of depending on wall-clock timing.

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
        self.workspace_id = workspace_id
        self.git_diff_embedding_days = git_diff_embedding_days
        self.reference_time = reference_time or datetime.now(UTC)

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

        # Stream commit hashes in newest-first order.
        commit_hashes = iter_commit_hashes_after_timestamp(
            self.git_dir,
            after_timestamp=after_timestamp,
        )

        for data in iter_commits(self.git_dir, commit_hashes, max_delta_lines=200):
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
        return {"poll_interval": 300}

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

        # Convert unix timestamp to datetime
        commit_dt = datetime.fromtimestamp(commit_data.timestamp, tz=UTC)

        include_diff = is_diff_chunk_eligible(
            commit_data.timestamp,
            self.git_diff_embedding_days,
            self.reference_time,
        )
        chunks = chunk_commit(commit_data, include_diff=include_diff)
        files_changed = commit_data.files_changed[:MAX_FILES_CHANGED]
        files_changed_total = max(commit_data.files_changed_total, len(commit_data.files_changed))

        # Build metadata dict
        metadata = {
            "author": commit_data.author,
            "committer": commit_data.committer,
            "timestamp": commit_data.timestamp,
            "files_changed_total": files_changed_total,
            "title": commit_data.title or "(no commit message)",
            "doc_id": commit_id,
            "commit_id": commit_id,
        }
        if self.workspace_id is not None:
            metadata["project_id"] = self.workspace_id

        return tuple(
            Record(
                workspace_id=self.workspace_id,
                source_kind=self.source_kind,
                source_id=f"{commit_id}:{chunk.section}:{chunk_position}",
                title=commit_data.title or "(no commit message)",
                body=chunk.text,
                created_at=commit_dt,
                updated_at=commit_dt,
                metadata={
                    **metadata,
                    **(
                        {"files_changed": files_changed}
                        if chunk.section == "summary"
                        else {}
                    ),
                    "chunk_section": chunk.section,
                    "chunk_index": chunk_position,
                    "section_index": chunk.section_index,
                    "chunk_count": len(chunks),
                    "estimated_tokens": chunk.estimated_tokens,
                },
                uri=None,
                status=RecordStatus.ACTIVE,
                embedding=None,
                embedding_model=None,
            )
            for chunk_position, chunk in enumerate(chunks)
        )
