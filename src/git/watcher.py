"""Git directory file watcher for automatic commit indexing."""

import asyncio
import logging
import queue
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from src.config import Config
    from src.git.commit_indexer import CommitIndexer

logger = logging.getLogger(__name__)


class GitWatcher:
    """Watches .git directories for changes and triggers incremental indexing."""
    
    def __init__(
        self,
        git_repos: list[Path],
        commit_indexer: CommitIndexer,
        config: Config,
        cooldown: float = 5.0,
    ):
        """
        Initialize git watcher.
        
        Args:
            git_repos: List of .git directory paths to watch
            commit_indexer: CommitIndexer instance
            config: Configuration object
            cooldown: Debounce cooldown in seconds
        """
        self._git_repos = git_repos
        self._commit_indexer = commit_indexer
        self._config = config
        self._cooldown = cooldown
        self._observers: list = []
        self._event_queue = queue.Queue()
        self._running = False
        self._task = None
    
    def start(self) -> None:
        """Start watching git directories."""
        if self._running:
            return
        
        self._running = True
        
        for git_dir in self._git_repos:
            # Watch specific paths: HEAD and refs/
            watch_paths = [
                git_dir / "HEAD",
                git_dir / "refs",
            ]
            
            for watch_path in watch_paths:
                if watch_path.exists():
                    event_handler = _GitEventHandler(self._event_queue, git_dir)
                    observer = Observer()
                    observer.schedule(
                        event_handler,
                        str(watch_path),
                        recursive=(watch_path.name == "refs"),
                    )
                    observer.start()
                    self._observers.append(observer)
        
        self._task = asyncio.create_task(self._process_events())
        logger.info(f"Git watcher started for {len(self._git_repos)} repositories")
    
    async def stop(self) -> None:
        """Stop watching git directories."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all observers
        for observer in self._observers:
            observer.stop()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(observer.join, timeout=1.0),
                    timeout=1.5,
                )
            except TimeoutError:
                logger.warning("Observer thread did not stop within timeout")
        
        self._observers.clear()
        
        # Cancel processing task
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None
        
        logger.info("Git watcher stopped")
    
    async def _process_events(self) -> None:
        """Process queued git directory changes with debouncing."""
        pending_repos: set[Path] = set()
        
        while self._running:
            try:
                try:
                    git_dir = await asyncio.to_thread(
                        self._event_queue.get, timeout=0.5
                    )
                    pending_repos.add(git_dir)
                except queue.Empty:
                    if pending_repos:
                        await asyncio.sleep(self._cooldown)
                        await self._batch_process(pending_repos)
                        pending_repos.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in git event processing: {e}")
        
        # Process remaining
        if pending_repos:
            try:
                await asyncio.wait_for(
                    self._batch_process(pending_repos),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Failed to process final git events: {e}")
    
    async def _batch_process(self, git_dirs: set[Path]) -> None:
        """Incrementally index commits for changed repositories."""
        from src.git.repository import get_commits_after_timestamp
        from src.git.commit_parser import parse_commit, build_commit_document
        
        for git_dir in git_dirs:
            try:
                # Get last indexed timestamp
                last_timestamp = await asyncio.to_thread(
                    self._commit_indexer.get_last_indexed_timestamp,
                    str(git_dir),
                )
                
                # Get new commits
                commit_hashes = await asyncio.to_thread(
                    get_commits_after_timestamp,
                    git_dir,
                    last_timestamp,
                )
                
                if not commit_hashes:
                    logger.debug(f"No new commits in {git_dir.parent}")
                    continue
                
                logger.info(f"Indexing {len(commit_hashes)} new commits from {git_dir.parent}")
                
                # Index new commits
                for hash in commit_hashes:
                    try:
                        commit = await asyncio.to_thread(
                            parse_commit,
                            git_dir,
                            hash,
                            self._config.git_indexing.delta_max_lines,
                        )
                        doc = build_commit_document(commit)
                        
                        await asyncio.to_thread(
                            self._commit_indexer.add_commit,
                            hash=commit.hash,
                            timestamp=commit.timestamp,
                            author=commit.author,
                            committer=commit.committer,
                            title=commit.title,
                            message=commit.message,
                            files_changed=commit.files_changed,
                            delta_truncated=commit.delta_truncated,
                            commit_document=doc,
                            repo_path=str(git_dir),
                        )
                    except Exception as e:
                        logger.error(f"Failed to index commit {hash}: {e}")
                
                logger.info(f"Updated commit index for {git_dir.parent.name}")
            
            except Exception as e:
                logger.error(f"Failed to update commits for {git_dir}: {e}")


class _GitEventHandler(FileSystemEventHandler):
    """Event handler for git directory changes."""
    
    def __init__(self, queue: queue.Queue[Path], git_dir: Path):
        super().__init__()
        self._queue = queue
        self._git_dir = git_dir
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Detect commits via refs/ or HEAD changes."""
        if event.is_directory:
            return
        
        path = Path(str(event.src_path))
        
        # Trigger on HEAD or refs/* changes
        if path.name == "HEAD" or "refs" in path.parts:
            try:
                self._queue.put_nowait(self._git_dir)
            except queue.Full:
                pass
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Detect new branches/tags."""
        if not event.is_directory and "refs" in Path(str(event.src_path)).parts:
            try:
                self._queue.put_nowait(self._git_dir)
            except queue.Full:
                pass
