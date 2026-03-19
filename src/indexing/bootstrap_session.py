from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from src.indexing.bootstrap_checkpoint import (
    BootstrapFileStamp,
    build_file_stamps,
    compute_bootstrap_generation,
    load_bootstrap_checkpoint,
    prepare_bootstrap_checkpoint,
)
from src.indexing.bootstrap_snapshot import (
    PublicIndexStateSnapshot,
    compute_bootstrap_completed_paths,
    derive_bootstrap_readiness_snapshot,
    derive_loaded_index_state_snapshot,
)
from src.indexing.manifest import IndexManifest, load_manifest
from src.indexing import tasks as indexing_tasks

logger = logging.getLogger(__name__)


type DiscoverFilesFn = Callable[[], list[str]]
type DiscoverGitRepositoriesFn = Callable[[], list[Path]]
type GetBootstrapManifestFn = Callable[[], IndexManifest]
type AsyncActionFn = Callable[[], Awaitable[None]]
type AsyncFloatFn = Callable[[], Awaitable[float]]
type GetFloatFn = Callable[[], float]
type GetIntFn = Callable[[], int]
type GetBoolFn = Callable[[], bool]
type PublishPublicStateFn = Callable[[PublicIndexStateSnapshot], None]
type MarkReadyFn = Callable[[], None]
type ScheduleWarmupFn = Callable[[], bool]
type ScheduleFollowUpFn = Callable[[], None]
type ReportFailureFn = Callable[[Exception, int, int], None]


@dataclass
class BootstrapSession:
    """Concrete coordinator for task-backed startup bootstrap.

    The session owns discovery, checkpoint preparation, remaining-work
    calculation, task enqueue, persisted-progress monitoring, and reporting
    public readiness state back to the runtime owner.
    """

    index_path: Path
    documents_roots: list[Path]
    git_refresh_enabled: bool
    discover_files: DiscoverFilesFn
    discover_git_repositories: DiscoverGitRepositoriesFn
    get_bootstrap_manifest: GetBootstrapManifestFn
    load_persisted_indices: AsyncActionFn
    persist_indices: AsyncActionFn
    compute_index_state_version: AsyncFloatFn
    get_loaded_index_state_version: GetFloatFn
    get_loaded_document_count: GetIntFn
    is_queryable: GetBoolFn
    publish_public_state: PublishPublicStateFn
    mark_ready: MarkReadyFn
    schedule_embedding_warmup: ScheduleWarmupFn
    schedule_initial_vocabulary_build: ScheduleFollowUpFn
    report_failure: ReportFailureFn

    async def preload_persisted_state(self, *, rebuild_pending: bool) -> bool:
        try:
            await self.load_persisted_indices()
        except Exception:
            logger.warning(
                "Failed to load persisted indices before background startup bootstrap",
                exc_info=True,
            )
            return False

        files_to_index = await asyncio.to_thread(self.discover_files)
        target_stamps = await asyncio.to_thread(
            build_file_stamps,
            files_to_index,
            self.documents_roots,
        )
        checkpoint = await asyncio.to_thread(load_bootstrap_checkpoint, self.index_path)
        saved_manifest = await asyncio.to_thread(load_manifest, self.index_path)

        preloaded_snapshot = derive_bootstrap_readiness_snapshot(
            checkpoint,
            saved_manifest,
            target_stamps,
            loaded_indexed_count=self.get_loaded_document_count(),
            queryable=self.is_queryable(),
            rebuild_pending=rebuild_pending,
        )
        if preloaded_snapshot is None:
            return False

        self.publish_public_state(preloaded_snapshot.public_state)
        if preloaded_snapshot.queryable:
            self.mark_ready()
        self.schedule_embedding_warmup()
        return True

    async def run(self) -> None:
        files_to_index = await asyncio.to_thread(self.discover_files)
        target_stamps = await asyncio.to_thread(
            build_file_stamps,
            files_to_index,
            self.documents_roots,
        )
        bootstrap_manifest = self.get_bootstrap_manifest()
        generation = compute_bootstrap_generation(bootstrap_manifest, target_stamps)
        checkpoint = await asyncio.to_thread(
            prepare_bootstrap_checkpoint,
            self.index_path,
            generation,
            target_stamps,
        )
        saved_manifest = await asyncio.to_thread(load_manifest, self.index_path)
        completed_paths = compute_bootstrap_completed_paths(
            checkpoint,
            saved_manifest,
            target_stamps,
        )
        remaining_files = [
            file_path
            for file_path in files_to_index
            if self._relative_path_for_file(file_path) not in completed_paths
        ]

        self.publish_public_state(
            PublicIndexStateSnapshot(
                status="indexing",
                indexed_count=max(self.get_loaded_document_count(), len(completed_paths)),
                total_count=len(files_to_index),
            )
        )

        if self.git_refresh_enabled:
            await self._enqueue_startup_git_refresh()

        if not files_to_index:
            await self.persist_indices()
            self.publish_public_state(
                derive_loaded_index_state_snapshot(
                    total_targets=0,
                    loaded_indexed_count=self.get_loaded_document_count(),
                )
            )
            self.mark_ready()
            self.schedule_embedding_warmup()
            return

        if not remaining_files:
            await self._load_ready_state_or_fail(
                total_targets=len(files_to_index),
                durably_completed_targets=len(completed_paths),
            )
            return

        submission = indexing_tasks.submit_index_batch(remaining_files)
        logger.info(
            "Enqueued %d startup indexing task(s) for %d remaining documents (%d already durably complete, %d already pending)",
            submission.enqueued_count,
            len(remaining_files),
            len(completed_paths),
            submission.already_pending_count,
        )

        if submission.enqueued_count == 0:
            if submission.all_represented:
                logger.info(
                    "Startup bootstrap found %d remaining document(s) already pending in queue; continuing to monitor persisted progress",
                    submission.already_pending_count,
                )
            else:
                self.report_failure(
                    RuntimeError("Task queue unavailable during startup bootstrap"),
                    len(completed_paths),
                    len(files_to_index),
                )
                return

        await self._monitor_persisted_progress(
            total_targets=len(files_to_index),
            target_stamps=target_stamps,
        )

    async def _enqueue_startup_git_refresh(self) -> None:
        repos = await asyncio.to_thread(self.discover_git_repositories)
        if not repos:
            logger.info("No git repositories found for task-driven startup refresh")
            return

        submission = indexing_tasks.submit_refresh_git_batch([str(repo) for repo in repos])
        logger.info(
            "Enqueued %d startup git refresh task(s) for %d repositories (%d already pending)",
            submission.enqueued_count,
            len(repos),
            submission.already_pending_count,
        )

    async def _load_ready_state_or_fail(
        self,
        *,
        total_targets: int,
        durably_completed_targets: int,
    ) -> None:
        try:
            await self.load_persisted_indices()
        except Exception as exc:
            self.report_failure(exc, durably_completed_targets, total_targets)
            return

        self._publish_loaded_progress(
            total_targets=total_targets,
            durably_completed_targets=durably_completed_targets,
        )
        self.mark_ready()
        self.schedule_embedding_warmup()
        self.schedule_initial_vocabulary_build()

    async def _monitor_persisted_progress(
        self,
        *,
        total_targets: int,
        target_stamps: dict[str, BootstrapFileStamp],
    ) -> None:
        while True:
            current_version = await self.compute_index_state_version()
            if current_version > self.get_loaded_index_state_version():
                try:
                    await self.load_persisted_indices()
                except Exception:
                    logger.debug(
                        "Startup bootstrap monitor could not load persisted state yet",
                        exc_info=True,
                    )
                else:
                    checkpoint = await asyncio.to_thread(
                        load_bootstrap_checkpoint,
                        self.index_path,
                    )
                    saved_manifest = await asyncio.to_thread(
                        load_manifest,
                        self.index_path,
                    )
                    completed_paths = compute_bootstrap_completed_paths(
                        checkpoint,
                        saved_manifest,
                        target_stamps,
                    )
                    self._publish_loaded_progress(
                        total_targets=total_targets,
                        durably_completed_targets=len(completed_paths),
                    )
                    if self.is_queryable():
                        self.mark_ready()
                        self.schedule_embedding_warmup()
                    if len(completed_paths) >= total_targets:
                        self.schedule_initial_vocabulary_build()
                        return

            await asyncio.sleep(0.2)

    def _publish_loaded_progress(
        self,
        *,
        total_targets: int,
        durably_completed_targets: int,
    ) -> None:
        loaded_snapshot = derive_loaded_index_state_snapshot(
            total_targets=total_targets,
            loaded_indexed_count=self.get_loaded_document_count(),
        )
        indexed_count = max(loaded_snapshot.indexed_count, durably_completed_targets)
        if total_targets > 0:
            indexed_count = min(indexed_count, total_targets)

        self.publish_public_state(
            PublicIndexStateSnapshot(
                status=loaded_snapshot.status,
                indexed_count=indexed_count,
                total_count=loaded_snapshot.total_count,
            )
        )

    def _relative_path_for_file(self, file_path: str) -> str | None:
        current_stamps = build_file_stamps([file_path], self.documents_roots)
        if not current_stamps:
            return None
        return next(iter(current_stamps.keys()))