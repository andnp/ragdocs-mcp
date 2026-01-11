"""Commit metadata extraction and delta truncation."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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


def parse_commit(git_dir: Path, commit_hash: str, max_delta_lines: int = 200) -> CommitData:
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
        message_lines = []
        if len(lines) > 5:
            started = False
            for line in lines[5:]:
                if line.strip() or started:
                    started = True
                    message_lines.append(line.rstrip())
        message = "\n".join(message_lines).rstrip()
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
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
        )
    
    # Get changed files
    files_changed = _get_changed_files(repo_path, commit_hash)
    
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
        files_changed=files_changed,
        delta_truncated=delta_truncated,
    )


def _get_changed_files(repo_path: Path, commit_hash: str) -> list[str]:
    """Get list of files changed in commit."""
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
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
        for file_path in commit.files_changed:
            parts.append(file_path)
        parts.append("")
    
    if commit.delta_truncated:
        parts.append(commit.delta_truncated)
    
    return "\n".join(parts)


def truncate_delta(diff_output: str, max_lines: int = 200) -> str:
    """Truncate diff to max_lines with indicator if truncated."""
    lines = diff_output.splitlines()
    if len(lines) <= max_lines:
        return diff_output
    
    truncated = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return f"{truncated}\n\n... ({remaining} lines omitted)"
