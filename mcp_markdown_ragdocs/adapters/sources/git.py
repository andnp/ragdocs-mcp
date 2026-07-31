"""Git commit content source adapter.

Implements ContentSource for git repositories, yielding commits as Records
suitable for indexing into the kernel's vector/keyword stores.
"""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import ChangeSignal, Cursor, Record, RecordStatus
from mcp_markdown_ragdocs.git.commit_parser import build_commit_document, parse_commit
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

        for commit_hash in commit_hashes:
            try:
                record = self._commit_to_record(commit_hash)
                if record is not None:
                    yield record
            except Exception:
                logger.exception(
                    f"Failed to convert commit {commit_hash[:8]} to Record"
                )
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

        if not commit_data.hash:
            logger.warning(f"Failed to parse commit {commit_hash[:8]}")
            return None

        # Build the searchable body text (message + diff summary)
        body = build_commit_document(commit_data)

        # Convert unix timestamp to datetime
        commit_dt = datetime.fromtimestamp(commit_data.timestamp, tz=UTC)

        # Build metadata dict
        metadata = {
            "author": commit_data.author,
            "committer": commit_data.committer,
            "timestamp": commit_data.timestamp,
            "files_changed": commit_data.files_changed,
        }

        # Create and return the Record
        record = Record(
            source_kind=self.source_kind,
            source_id=f"git:{commit_data.hash}",
            title=commit_data.title or "(no commit message)",
            body=body,
            created_at=commit_dt,
            updated_at=commit_dt,
            metadata=metadata,
            uri=None,  # Could be set to a commit link if repo URL is known
            status=RecordStatus.ACTIVE,
            embedding=None,  # Kernel will compute
            embedding_model=None,
        )

        return record
