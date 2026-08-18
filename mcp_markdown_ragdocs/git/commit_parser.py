"""Commit metadata extraction and delta truncation."""

import logging
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from searchkernel.api import truncate_delta

logger = logging.getLogger(__name__)
COMMIT_BATCH_SIZE = 32
MAX_FILES_CHANGED = 50
_BULK_RECORD_SEPARATOR = "\x1e"
_BULK_FIELD_SEPARATOR = "\x1f"
_BULK_METADATA_SEPARATOR = "\x00"


@dataclass
class CommitData:
    hash: str
    timestamp: int
    author: str
    committer: str
    title: str
    message: str
    files_changed: list[str]
    delta_truncated: str
    files_changed_total: int = 0


def _cap_files_changed(files: list[str]) -> tuple[list[str], int]:
    """Cap a raw file list, preserving the true count separately."""
    return files[:MAX_FILES_CHANGED], len(files)


def parse_commits(
    git_dir: Path,
    commit_hashes: list[str],
    max_delta_lines: int = 200,
) -> list[CommitData]:
    """Extract all commits while preserving the list-based parser API."""
    return list(iter_commits(git_dir, commit_hashes, max_delta_lines))


def iter_commits(
    git_dir: Path,
    commit_hashes: Iterable[str],
    max_delta_lines: int = 200,
) -> Iterator[CommitData]:
    """Yield commits in bounded batches, falling back per batch on failure."""
    commit_hash_iterator = iter(commit_hashes)
    while batch := list(islice(commit_hash_iterator, COMMIT_BATCH_SIZE)):
        try:
            yield from _parse_commit_batch(git_dir, batch, max_delta_lines)
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ) as e:
            logger.warning(
                "Bulk parsing failed for commit batch starting at %s: %s",
                batch[0][:8],
                e,
            )
            for commit_hash in batch:
                yield parse_commit(git_dir, commit_hash, max_delta_lines)


def _parse_commit_batch(
    git_dir: Path,
    commit_hashes: list[str],
    max_delta_lines: int,
) -> list[CommitData]:
    """Parse one bounded batch with a single Git subprocess."""
    if not commit_hashes:
        return []

    format_string = (
        f"{_BULK_RECORD_SEPARATOR}%H{_BULK_FIELD_SEPARATOR}%ct"
        f"{_BULK_FIELD_SEPARATOR}%an <%ae>{_BULK_FIELD_SEPARATOR}%cn <%ce>"
        f"{_BULK_FIELD_SEPARATOR}%s{_BULK_FIELD_SEPARATOR}%b"
        "%x00"
    )
    result = subprocess.run(
        [
            "git",
            "show",
            "--format=" + format_string,
            "--raw",
            "--patch",
            "--no-color",
            "--no-renames",
            "--no-ext-diff",
            *commit_hashes,
        ],
        cwd=git_dir.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )

    parsed: dict[str, CommitData] = {}
    for raw_record in result.stdout.split(_BULK_RECORD_SEPARATOR):
        if not raw_record.strip():
            continue
        metadata, body = raw_record.split(_BULK_METADATA_SEPARATOR, 1)
        fields = metadata.split(_BULK_FIELD_SEPARATOR, 5)
        if len(fields) != 6:
            raise ValueError("Bulk Git output contained incomplete metadata")

        hash_val, timestamp_text, author, committer, title, message_text = fields
        hash_val = hash_val.strip()
        if hash_val not in commit_hashes:
            raise ValueError(f"Unexpected commit in bulk Git output: {hash_val}")

        try:
            timestamp = int(timestamp_text.strip())
        except ValueError as e:
            raise ValueError(f"Invalid timestamp for commit {hash_val}") from e

        diff_start = re.search(r"(?m)^diff --(?:git|cc) ", body)
        if diff_start is None:
            files_text = body
            delta = ""
        else:
            files_text = body[: diff_start.start()]
            delta = body[diff_start.start() :]

        files_changed = [
            line.split("\t", 1)[1]
            for line in files_text.splitlines()
            if line.startswith(":") and "\t" in line
        ]
        capped_files, files_total = _cap_files_changed(files_changed)
        parsed[hash_val] = CommitData(
            hash=hash_val,
            timestamp=timestamp,
            author=author.strip(),
            committer=committer.strip(),
            title=title.strip(),
            message=_clean_message(message_text),
            files_changed=capped_files,
            delta_truncated=truncate_delta(delta, max_delta_lines),
            files_changed_total=files_total,
        )

    if set(parsed) != set(commit_hashes):
        missing = set(commit_hashes) - set(parsed)
        raise ValueError(f"Bulk Git output omitted commits: {sorted(missing)}")
    return [parsed[commit_hash] for commit_hash in commit_hashes]


def _clean_message(message_text: str) -> str:
    """Match the existing parser's treatment of commit message bodies."""
    message_lines = []
    started = False
    for line in message_text.splitlines():
        if line.strip() or started:
            started = True
            message_lines.append(line.rstrip())
    return "\n".join(message_lines).rstrip()


def parse_commit(
    git_dir: Path, commit_hash: str, max_delta_lines: int = 200
) -> CommitData:
    """
    Extract commit metadata and truncated delta.

    Args:
        git_dir: Path to .git directory
        commit_hash: Full commit SHA
        max_delta_lines: Maximum diff lines to keep

    Returns:
        CommitData with all fields populated
    """
    repo_path = git_dir.parent

    # Get commit metadata
    format_string = "%H%n%ct%n%an <%ae>%n%cn <%ce>%n%s%n%b"

    try:
        result = subprocess.run(
            ["git", "show", "--format=" + format_string, "--no-patch", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        lines = result.stdout.splitlines()

        # Parse metadata
        hash_val = lines[0].strip() if len(lines) > 0 else commit_hash
        timestamp = int(lines[1].strip()) if len(lines) > 1 else 0
        author = lines[2].strip() if len(lines) > 2 else ""
        committer = lines[3].strip() if len(lines) > 3 else ""
        title = lines[4].strip() if len(lines) > 4 else ""

        # Message body (everything after title, excluding empty lines at start)
        message = _clean_message("\n".join(lines[5:])) if len(lines) > 5 else ""

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
    ) as e:
        logger.error(f"Failed to parse commit metadata {commit_hash}: {e}")
        # Return minimal commit data
        return CommitData(
            hash=commit_hash,
            timestamp=0,
            author="",
            committer="",
            title="",
            message="",
            files_changed=[],
            delta_truncated="",
            files_changed_total=0,
        )

    # Get changed files
    files_changed = _get_changed_files(repo_path, commit_hash)
    capped_files, files_total = _cap_files_changed(files_changed)

    # Get delta
    delta = _get_delta(repo_path, commit_hash)
    delta_truncated = truncate_delta(delta, max_delta_lines)

    return CommitData(
        hash=hash_val,
        timestamp=timestamp,
        author=author,
        committer=committer,
        title=title,
        message=message,
        files_changed=capped_files,
        delta_truncated=delta_truncated,
        files_changed_total=files_total,
    )


def _get_changed_files(repo_path: Path, commit_hash: str) -> list[str]:
    """Get list of files changed in commit."""
    try:
        result = subprocess.run(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "--root",
                commit_hash,
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return files

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to get changed files for {commit_hash}: {e}")
        return []


def _get_delta(repo_path: Path, commit_hash: str) -> str:
    """Get diff delta for commit."""
    try:
        result = subprocess.run(
            ["git", "show", "--format=", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        return result.stdout

    except subprocess.CalledProcessError:
        # Try with encoding fallback
        try:
            result = subprocess.run(
                ["git", "show", "--format=", commit_hash],
                cwd=repo_path,
                capture_output=True,
                check=True,
                timeout=10,
            )
            # Try decoding with fallback
            try:
                return result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                return result.stdout.decode("latin-1", errors="replace")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to get delta for {commit_hash}: {e}")
            return ""
    except subprocess.TimeoutExpired:
        logger.warning(f"Delta fetch timeout for {commit_hash}")
        return ""


def build_commit_document(commit: CommitData) -> str:
    """
    Build searchable text document from commit data.

    Format:
    {title}

    {message}

    Author: {author}
    Committer: {committer}

    Files changed:
    {file_1}
    {file_2}

    {delta_truncated}

    Returns:
        Formatted text for embedding
    """
    parts = []

    if commit.title:
        parts.append(commit.title)
        parts.append("")

    if commit.message:
        parts.append(commit.message)
        parts.append("")

    if commit.author or commit.committer:
        if commit.author:
            parts.append(f"Author: {commit.author}")
        if commit.committer:
            parts.append(f"Committer: {commit.committer}")
        parts.append("")

    if commit.files_changed:
        parts.append("Files changed:")
        parts.extend(commit.files_changed)
        parts.append("")

    if commit.delta_truncated:
        parts.append(commit.delta_truncated)

    return "\n".join(parts)
