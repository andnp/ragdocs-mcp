"""Tests for resumable Google Drive synchronization."""

import asyncio
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
        complete: bool = False,
    ):
        self.events.append("checkpoint")
        return super().persist_inventory_batch_after_index(
            namespace,
            page_token=page_token,
            batch=batch,
            complete=complete,
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


@pytest.mark.asyncio
async def test_concurrent_materialization_preserves_input_order(tmp_path: Path) -> None:
    """
    Keep the page's file order even though later files finish first.
    """

    class _ReorderingClient(_Client):
        async def download_file(self, file_id: str) -> bytes:
            # Files earlier in the page take longer, so a naive unordered
            # gather would return "slow-2" and "slow-0" ahead of "fast-1".
            if file_id.startswith("slow"):
                await asyncio.sleep(0.02)
            return f"body:{file_id}".encode()

    client = _ReorderingClient()
    client.pages = {
        None: DriveFilePage((_file("slow-0"), _file("fast-1"), _file("slow-2")))
    }
    source = GoogleDriveContentSource(
        cast(Any, client), workspace_id="workspace", extractor=cast(Any, _extractor)
    )
    writer = _Writer([])
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, writer),
        scope_generation="generation",
        max_seconds=60,
        max_concurrent_materializations=3,
    )

    await sync.sync_inventory(source.scopes[0])

    assert [record.source_id for batch in writer.batches for record in batch] == [
        "slow-0",
        "fast-1",
        "slow-2",
    ]


@pytest.mark.asyncio
async def test_one_failing_materialization_does_not_lose_page(tmp_path: Path) -> None:
    """
    Exclude only the failing file; the rest of the page still indexes.
    """

    class _FlakySource(GoogleDriveContentSource):
        async def materialize_record(self, file, *, scope=None, known_change_keys=None):
            if file.id == "boom":
                raise RuntimeError("simulated bug")
            return await super().materialize_record(
                file, scope=scope, known_change_keys=known_change_keys
            )

    client = _Client()
    client.pages = {
        None: DriveFilePage((_file("first"), _file("boom"), _file("last")))
    }
    source = _FlakySource(
        cast(Any, client), workspace_id="workspace", extractor=cast(Any, _extractor)
    )
    writer = _Writer([])
    sync = GoogleDriveSync(
        source,
        GDriveSyncCheckpointStore(tmp_path),
        cast(Any, writer),
        scope_generation="generation",
        max_seconds=60,
    )

    progress = await sync.sync_inventory(source.scopes[0])

    assert progress.complete is True
    assert [record.source_id for batch in writer.batches for record in batch] == [
        "first",
        "last",
    ]


@pytest.mark.asyncio
async def test_mid_page_deadline_eventually_indexes_every_file(tmp_path: Path) -> None:
    """
    A deadline hit mid-page must never permanently skip the unprocessed suffix.

    Runs sync_inventory repeatedly against one 10-file page whose deadline is
    tied to the number of downloads performed so far (so it reliably fires
    mid-page), and proves every file is indexed by some run and none is
    downloaded twice thanks to the change-key skip.
    """

    class _OnePageClient:
        def __init__(self) -> None:
            self.downloads: list[str] = []

        async def get_start_page_token(self, scope: object) -> DriveStartPageToken:
            del scope
            return DriveStartPageToken("start-token")

        async def list_files_page(
            self, scope: object, *, page_token: str | None = None, page_size: int = 1000
        ) -> DriveFilePage:
            del scope, page_token, page_size
            return DriveFilePage(tuple(_file(f"f{i}") for i in range(10)))

        async def download_file(self, file_id: str) -> bytes:
            self.downloads.append(file_id)
            return f"body:{file_id}".encode()

    client = _OnePageClient()
    source = GoogleDriveContentSource(
        cast(Any, client), workspace_id="workspace", extractor=cast(Any, _extractor)
    )
    store = GDriveSyncCheckpointStore(tmp_path)

    def clock() -> float:
        return float(len(client.downloads))

    def make_sync(writer: "_Writer") -> GoogleDriveSync:
        return GoogleDriveSync(
            source,
            store,
            cast(Any, writer),
            scope_generation="generation",
            max_seconds=3.0,
            max_concurrent_materializations=4,
            clock=clock,
        )

    all_indexed: list[str] = []
    complete = False
    runs = 0
    while not complete and runs < 10:
        writer = _Writer([])
        progress = await make_sync(writer).sync_inventory(source.scopes[0])
        all_indexed.extend(
            record.source_id for batch in writer.batches for record in batch
        )
        complete = progress.complete
        runs += 1

    assert complete is True
    assert runs > 1, "test setup should force at least one mid-page truncation"
    assert set(all_indexed) == {f"f{i}" for i in range(10)}
    assert sorted(client.downloads) == sorted(f"f{i}" for i in range(10))
