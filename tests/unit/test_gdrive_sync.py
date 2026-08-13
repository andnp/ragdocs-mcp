"""Tests for resumable Google Drive synchronization."""

from pathlib import Path
from typing import Any, cast

import pytest
from searchkernel.api import Record

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.checkpoints import (
    GDriveSyncCheckpointStore,
    checkpoint_namespace,
)
from mcp_markdown_ragdocs.gdrive.extraction import ExtractionResult, ExtractionStatus
from mcp_markdown_ragdocs.gdrive.models import (
    DriveFile,
    DriveFilePage,
    DriveStartPageToken,
)
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
    profile: object,
    limits: object,
) -> ExtractionResult:
    del payload, mime_type, limits
    return ExtractionResult(
        ExtractionStatus.INDEXED,
        f"body:{profile.name}",
        profile.name,
        profile.version,
    )


class _Client:
    def __init__(self) -> None:
        self.start_calls = 0
        self.page_tokens: list[str | None] = []
        self.pages = {
            None: DriveFilePage((_file("first"),), next_page_token="page-2"),
            "page-2": DriveFilePage((_file("second"),)),
        }

    async def get_start_page_token(self, scope: object) -> DriveStartPageToken:
        del scope
        self.start_calls += 1
        return DriveStartPageToken("start-token")

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
        extractor=_extractor,
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
    assert store.load(namespace).inventory_page_token is None
