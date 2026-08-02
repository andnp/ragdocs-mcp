"""Bounded Git ingestion adapters for the searchkernel source seam."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

from mcp_markdown_ragdocs.adapters.sources.git import GitContentSource
from searchkernel.api import (
    IngestionReceipt,
    Record,
    SearchKernel,
    SourceBatch,
)


class GitBatchContentSource:
    """Expose Git records as bounded searchkernel source batches."""

    source_kind = GitContentSource.source_kind

    def __init__(
        self,
        source: GitContentSource,
        batch_size: int,
        *,
        skip_batch: Callable[[tuple[Record, ...]], bool] | None = None,
        on_skip: Callable[[tuple[Record, ...]], None] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._source = source
        self._batch_size = batch_size
        self._skip_batch = skip_batch
        self._on_skip = on_skip
        self.last_emitted_batch: tuple[Record, ...] | None = None

    async def iter_batches(
        self,
        since: str | None = None,
    ) -> AsyncIterator[SourceBatch]:
        batch: list[Record] = []
        for record in self._source.iter_records(since=since):
            batch.append(record)
            if len(batch) < self._batch_size:
                continue
            for emitted in self._emit_batch(tuple(batch)):
                yield emitted
            batch.clear()
        if batch:
            for emitted in self._emit_batch(tuple(batch)):
                yield emitted

    async def iter_records(self, since: str | None = None) -> AsyncIterator[Record]:
        for record in self._source.iter_records(since=since):
            yield record

    def change_signal(self) -> dict[str, int]:
        return self._source.change_signal()

    def _emit_batch(self, records: tuple[Record, ...]) -> Iterator[SourceBatch]:
        if self._skip_batch is not None and self._skip_batch(records):
            if self._on_skip is not None:
                self._on_skip(records)
            return
        self.last_emitted_batch = records
        yield SourceBatch(
            records=records,
            terminal_cursor=_batch_cursor(records),
        )

    @staticmethod
    def cursor_for(record: Record) -> str:
        return str(int(record.updated_at.timestamp()))


async def iter_git_ingestion_receipts(
    index_manager,
    source: GitContentSource,
    *,
    since: str | None,
    batch_size: int,
) -> AsyncIterator[IngestionReceipt]:
    """Yield one searchkernel receipt per bounded Git batch."""
    batch_source = GitBatchContentSource(source, batch_size)
    kernel = SearchKernel.build(
        ingestor=index_manager.ingestor,
        content_sources=(batch_source,),
    )
    async for receipt in kernel.iter_ingest_source(
        batch_source.source_kind,
        since=since,
        batch_size=batch_size,
        failure_mode="strict",
    ):
        yield receipt


def _batch_cursor(records: tuple[Record, ...]) -> str | None:
    if not records:
        return None
    return str(max(int(record.updated_at.timestamp()) for record in records))


__all__ = ["GitBatchContentSource", "iter_git_ingestion_receipts"]
