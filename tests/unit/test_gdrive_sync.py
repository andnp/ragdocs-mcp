"""Tests for resumable Google Drive synchronization."""

from pathlib import Path
from typing import Any, cast

import pytest
from searchkernel.api import Record
from searchkernel.domain import RecordStatus

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.checkpoints import (
    GDriveSyncCheckpointStore,
    checkpoint_namespace,
)
from mcp_markdown_ragdocs.gdrive.extraction import (
    DEFAULT_EXTRACTION_LIMITS,
    ExtractionLimits,
    ExtractionProfile,
    ExtractionResult,
    ExtractionStatus,
)
from mcp_markdown_ragdocs.gdrive.models import (
    DriveChange,
    DriveChangePage,
    DriveFile,
    DriveFilePage,
    DriveStartPageToken,
)
from mcp_markdown_ragdocs.gdrive.state import GDriveScopeIdentity, GDriveStateRepository
from mcp_markdown_ragdocs.gdrive.sync import GoogleDriveSync


def _file(file_id: str) -> DriveFile:
    return DriveFile(
        file_id,
        f"{file_id}.txt",
        "text/plain",
        modified_time="2026-01-01T00:00:00Z",
    )


def _extractor(
    payload: bytes,
    mime_type: str,
    *,
    profile: ExtractionProfile | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    del payload, mime_type, limits
    if profile is None:
        raise ValueError("profile is required")
    return ExtractionResult(
        ExtractionStatus.INDEXED,
        f"body:{profile.name}",
        profile.name,
        profile.version,
    )


class _Client:
    def __init__(self) -> None:
        self.start_calls = 0
        self.invalid_change_token: str | None = None
        self.page_tokens: list[str | None] = []
        self.change_tokens: list[str] = []
        self.pages = {
            None: DriveFilePage((_file("first"),), next_page_token="page-2"),
            "page-2": DriveFilePage((_file("second"),)),
        }
        self.change_pages = {
            "start-token": DriveChangePage(
                (
                    DriveChange("updated", False, _file("updated")),
                    DriveChange("gone", True),
                ),
                next_page_token="change-2",
            ),
            "change-2": DriveChangePage(
                (
                    DriveChange(
                        "trashed",
                        False,
                        DriveFile(
                            "trashed",
                            "trashed.txt",
                            "text/plain",
                            trashed=True,
                        ),
                    ),
                ),
                new_start_page_token="new-start-token",
            ),
        }

    async def get_start_page_token(self, scope: object) -> DriveStartPageToken:
        del scope
        self.start_calls += 1
        token = "start-token" if self.start_calls == 1 else "reset-start-token"
        return DriveStartPageToken(token)

    async def list_files_page(
        self,
        scope: object,
        *,
        page_token: str | None = None,
        page_size: int = 1000,
    ) -> DriveFilePage:
        del scope, page_size
        self.page_tokens.append(page_token)
        return self.pages[page_token]

    async def download_file(self, file_id: str) -> bytes:
        return f"body:{file_id}".encode()

    async def list_changes_page(
        self,
        scope: object,
        page_token: str,
        *,
        page_size: int = 1000,
    ) -> DriveChangePage:
        del scope, page_size
        self.change_tokens.append(page_token)
        if page_token == self.invalid_change_token:
            raise _ExpiredTokenError()
        return self.change_pages[page_token]


class _ExpiredTokenError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("page token expired")
        self.resp = type("Response", (), {"status": 410})()


class _Writer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.batches: list[tuple[Record, ...]] = []

    def index_records(self, records: tuple[Record, ...]) -> bool:
        self.events.append("index")
        self.batches.append(records)
        return True

    def persist(self) -> None:
        self.events.append("persist")


class _CheckpointStore(GDriveSyncCheckpointStore):
    def __init__(self, index_root: Path, events: list[str]) -> None:
        super().__init__(index_root)
        self.events = events

    def begin_inventory(self, namespace: str, start_token: str):
        self.events.append("begin")
        return super().begin_inventory(namespace, start_token)

    def persist_inventory_batch_after_index(
        self,
        namespace: str,
        *,
        page_token: str | None,
        batch: int,
    ):
        self.events.append("checkpoint")
        return super().persist_inventory_batch_after_index(
            namespace,
            page_token=page_token,
            batch=batch,
        )

    def persist_changes_after_index(self, namespace: str, changes_token: str):
        self.events.append("checkpoint")
        return super().persist_changes_after_index(namespace, changes_token)


@pytest.mark.asyncio
async def test_inventory_resume_reuses_page_checkpoint_after_durable_write(
    tmp_path: Path,
) -> None:
    """
    Resume the next Drive page without recapturing the start token.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    events: list[str] = []
    store = _CheckpointStore(tmp_path, events)
    writer = _Writer(events)
    sync = GoogleDriveSync(
        source,
        store,
        cast(Any, writer),
        scope_generation="generation",
        max_pages=1,
        max_seconds=60,
    )

    first = await sync.sync_inventory(source.scopes[0])
    namespace = checkpoint_namespace("generation-shared-with-me")
    checkpoint = store.load(namespace)

    assert first.complete is False
    assert checkpoint is not None
    assert checkpoint.inventory_start_token == "start-token"
    assert checkpoint.inventory_page_token == "page-2"
    assert checkpoint.inventory_batch == 1
    assert events == ["begin", "index", "persist", "checkpoint"]

    sync.max_pages = 2
    second = await sync.sync_inventory(source.scopes[0])

    assert second.complete is True
    assert client.start_calls == 1
    assert client.page_tokens == [None, "page-2"]
    assert [record.source_id for batch in writer.batches for record in batch] == [
        "first",
        "second",
    ]
    checkpoint = store.load(namespace)
    assert checkpoint is not None
    assert checkpoint.inventory_page_token is None


@pytest.mark.asyncio
async def test_inventory_persists_each_configured_record_batch(tmp_path: Path) -> None:
    """
    Split one Drive page into bounded durable index writes.
    """
    client = _Client()
    client.pages = {None: DriveFilePage((_file("first"), _file("second")))}
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    events: list[str] = []
    writer = _Writer(events)
    sync = GoogleDriveSync(
        source,
        _CheckpointStore(tmp_path, events),
        cast(Any, writer),
        scope_generation="generation",
        batch_size=1,
        max_seconds=60,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is True
    assert [tuple(record.source_id for record in batch) for batch in writer.batches] == [
        ("first",),
        ("second",),
    ]
    assert events == ["begin", "index", "persist", "index", "persist", "checkpoint"]


@pytest.mark.asyncio
async def test_change_replay_advances_feed_cursor_after_each_ordered_page(
    tmp_path: Path,
) -> None:
    """
    Replay updates and tombstones in order before storing the feed cursor.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    events: list[str] = []
    store = _CheckpointStore(tmp_path, events)
    writer = _Writer(events)
    sync = GoogleDriveSync(
        source,
        store,
        cast(Any, writer),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    )

    await sync.sync_inventory(source.scopes[0])
    events.clear()
    writer.batches.clear()

    progress = await sync.sync_changes(source.scopes[0])
    namespace = checkpoint_namespace("generation-shared-with-me")
    checkpoint = store.load(namespace)
    records = [record for batch in writer.batches for record in batch]

    assert progress.complete is True
    assert client.change_tokens == ["start-token", "change-2"]
    assert checkpoint is not None
    assert checkpoint.changes_token == "new-start-token"
    assert [record.source_id for record in records] == ["updated", "gone", "trashed"]
    assert [record.status for record in records] == [
        RecordStatus.ACTIVE,
        RecordStatus.ARCHIVED,
        RecordStatus.ARCHIVED,
    ]
    assert events == [
        "index",
        "persist",
        "checkpoint",
        "index",
        "persist",
        "checkpoint",
    ]


@pytest.mark.asyncio
async def test_restart_recovers_inventory_from_persisted_page_checkpoint(
    tmp_path: Path,
) -> None:
    """
    Resume inventory in a fresh sync instance after a bounded interruption.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    store = GDriveSyncCheckpointStore(tmp_path)
    first_writer = _Writer([])
    first_sync = GoogleDriveSync(
        source,
        store,
        cast(Any, first_writer),
        scope_generation="generation",
        max_pages=1,
        max_seconds=60,
    )

    interrupted = await first_sync.sync_inventory(source.scopes[0])

    restarted_source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    resumed_writer = _Writer([])
    resumed = await GoogleDriveSync(
        restarted_source,
        store,
        cast(Any, resumed_writer),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    ).sync_inventory(restarted_source.scopes[0])

    assert interrupted.complete is False
    assert resumed.complete is True
    assert client.start_calls == 1
    assert client.page_tokens == [None, "page-2"]
    assert [record.source_id for batch in resumed_writer.batches for record in batch] == [
        "second"
    ]


@pytest.mark.asyncio
async def test_restart_recovers_changes_from_persisted_feed_cursor(
    tmp_path: Path,
) -> None:
    """
    Resume change replay in a fresh sync instance after one page commits.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    store = GDriveSyncCheckpointStore(tmp_path)
    inventory_writer = _Writer([])
    inventory_sync = GoogleDriveSync(
        source,
        store,
        cast(Any, inventory_writer),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    )
    await inventory_sync.sync_inventory(source.scopes[0])

    first_change_writer = _Writer([])
    first_change = await GoogleDriveSync(
        source,
        store,
        cast(Any, first_change_writer),
        scope_generation="generation",
        max_pages=1,
        max_seconds=60,
    ).sync_changes(source.scopes[0])

    restarted_source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    resumed_writer = _Writer([])
    resumed = await GoogleDriveSync(
        restarted_source,
        store,
        cast(Any, resumed_writer),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    ).sync_changes(restarted_source.scopes[0])

    checkpoint = store.load(checkpoint_namespace("generation-shared-with-me"))
    assert first_change.complete is False
    assert resumed.complete is True
    assert client.change_tokens == ["start-token", "change-2"]
    assert checkpoint is not None
    assert checkpoint.changes_token == "new-start-token"
    assert [record.source_id for batch in resumed_writer.batches for record in batch] == [
        "trashed"
    ]
    assert resumed_writer.batches[0][0].status is RecordStatus.ARCHIVED


@pytest.mark.asyncio
async def test_invalid_change_token_starts_bounded_full_resync(tmp_path: Path) -> None:
    """
    Replace an expired feed cursor with a fresh inventory start token.
    """
    client = _Client()
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    store = GDriveSyncCheckpointStore(tmp_path)
    writer = _Writer([])
    sync = GoogleDriveSync(
        source,
        store,
        cast(Any, writer),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    )

    await sync.sync_inventory(source.scopes[0])
    client.invalid_change_token = "start-token"

    progress = await sync.sync_changes(source.scopes[0])
    checkpoint = store.load(checkpoint_namespace("generation-shared-with-me"))

    assert progress.complete is True
    assert progress.token_reset is True
    assert progress.start_token == "reset-start-token"
    assert client.start_calls == 2
    assert client.change_tokens == ["start-token"]
    assert checkpoint is not None
    assert checkpoint.inventory_start_token == "reset-start-token"
    assert checkpoint.changes_token is None


@pytest.mark.asyncio
async def test_complete_inventory_replaces_stale_scope_memberships(
    tmp_path: Path,
) -> None:
    """
    Remove stale scope visibility only after the final inventory page commits.
    """
    client = _Client()
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    identity = GDriveScopeIdentity("gdrive", "workspace", "shared-with-me")
    repository.add_membership(identity, "stale")
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        state_repository=repository,
    )
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, _Writer([])),
        scope_generation="generation",
        max_pages=2,
        max_seconds=60,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is True
    assert repository.load_scope_memberships(identity).source_ids == (
        "first",
        "second",
    )


class _FakeClock:
    """A controllable monotonic clock advanced explicitly by the test."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


class _IdempotentWriter:
    """Tracks the set of ever-indexed source ids, collapsing repeat writes."""

    def __init__(self) -> None:
        self.indexed_ids: set[str] = set()
        self.batches: list[tuple[Record, ...]] = []

    def index_records(self, records: tuple[Record, ...]) -> bool:
        self.batches.append(records)
        self.indexed_ids.update(record.source_id for record in records)
        return True

    def persist(self) -> None:
        pass


@pytest.mark.asyncio
async def test_deadline_hit_mid_page_returns_early_and_incomplete(tmp_path: Path) -> None:
    """
    A page whose per-file work exceeds max_seconds stops before finishing the page.
    """
    clock = _FakeClock()

    def slow_extractor(
        payload: bytes, mime_type: str, *, profile=None, limits=None
    ) -> ExtractionResult:
        del payload, mime_type, limits
        if profile is None:
            raise ValueError("profile is required")
        clock.advance(100.0)
        return ExtractionResult(
            ExtractionStatus.INDEXED, f"body:{profile.name}", profile.name, profile.version
        )

    client = _Client()
    client.pages = {
        None: DriveFilePage((_file("first"), _file("second"), _file("third")))
    }
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, slow_extractor),
    )
    writer = _IdempotentWriter()
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, writer),
        scope_generation="generation",
        batch_size=1,
        max_seconds=10.0,
        clock=clock,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is False
    # Only the first file's work fit inside the deadline; the page did not finish.
    assert writer.indexed_ids == {"first"}
    namespace = checkpoint_namespace("generation-shared-with-me")
    checkpoint = sync.checkpoint_store.load(namespace)
    assert checkpoint is not None
    # The checkpoint cannot represent an offset within a page, so it stays
    # at the page's start: the next invocation refetches this same page.
    assert checkpoint.inventory_page_token is None
    assert checkpoint.inventory_batch == 0


@pytest.mark.asyncio
async def test_deadline_interruption_eventually_covers_every_file(tmp_path: Path) -> None:
    """
    Repeated bounded invocations lose no file: every file is eventually indexed.
    """
    clock = _FakeClock()
    step = [100.0]

    def variable_extractor(
        payload: bytes, mime_type: str, *, profile=None, limits=None
    ) -> ExtractionResult:
        del payload, mime_type, limits
        if profile is None:
            raise ValueError("profile is required")
        clock.advance(step[0])
        return ExtractionResult(
            ExtractionStatus.INDEXED, f"body:{profile.name}", profile.name, profile.version
        )

    client = _Client()
    client.pages = {
        None: DriveFilePage((_file("first"), _file("second"), _file("third")))
    }
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, variable_extractor),
    )
    writer = _IdempotentWriter()
    store = GDriveSyncCheckpointStore(tmp_path)
    sync = GoogleDriveSync(
        source,
        store,
        cast(Any, writer),
        scope_generation="generation",
        batch_size=1,
        max_seconds=10.0,
        clock=clock,
    )

    # First pass: each file costs 100s against a 10s budget, so only the
    # first file in the page fits before the deadline breaks the loop.
    first = await sync.sync_inventory(source.scopes[0])
    assert first.complete is False

    # Now the remaining files are cheap enough to finish within budget.
    step[0] = 1.0
    progress = first
    guard = 0
    while not progress.complete:
        guard += 1
        assert guard < 10, "inventory never completed"
        progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is True
    assert writer.indexed_ids == {"first", "second", "third"}


@pytest.mark.asyncio
async def test_genuine_completion_still_reports_complete(tmp_path: Path) -> None:
    """
    An inventory that finishes within budget reports complete=True in one call.
    """
    clock = _FakeClock()
    client = _Client()
    client.pages = {
        None: DriveFilePage((_file("first"), _file("second"), _file("third")))
    }
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
    )
    writer = _IdempotentWriter()
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, writer),
        scope_generation="generation",
        batch_size=1,
        max_seconds=10.0,
        clock=clock,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is True
    assert writer.indexed_ids == {"first", "second", "third"}


@pytest.mark.asyncio
async def test_incomplete_inventory_preserves_stale_scope_memberships(
    tmp_path: Path,
) -> None:
    """
    Retain the prior scope snapshot while inventory remains bounded and incomplete.
    """
    client = _Client()
    repository = GDriveStateRepository(tmp_path / "gdrive-state.db")
    identity = GDriveScopeIdentity("gdrive", "workspace", "shared-with-me")
    repository.add_membership(identity, "stale")
    source = GoogleDriveContentSource(
        cast(Any, client),
        workspace_id="workspace",
        extractor=cast(Any, _extractor),
        state_repository=repository,
    )
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, _Writer([])),
        scope_generation="generation",
        max_pages=1,
        max_seconds=60,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is False
    assert repository.load_scope_memberships(identity).source_ids == (
        "first",
        "stale",
    )
