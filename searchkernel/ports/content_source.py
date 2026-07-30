"""ContentSource port: adapters for ingesting content from external sources.

This is the primary outbound port for the kernel. Content sources implement one
of two flavors:
  - Ingestible: the kernel stores and indexes the content
  - Searchable: the source runs its own search; kernel fuses results
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import ChangeSignal, Cursor, Record, ScoredRef


@runtime_checkable
class ContentSource(Protocol):
    """Ingestible source: kernel owns indexing and storage.

    The source yields Records; the kernel chunks, embeds (unless the record
    carries pre-computed embeddings), and indexes them.

    Attributes:
        source_kind: Stable identifier for this source type
                     (e.g., "note", "git_commit", "gmail").
    """

    source_kind: str

    def iter_records(self, since: Cursor | None = None) -> Iterable[Record]:
        """
        Iterate over records to ingest, optionally since a cursor.

        Args:
            since: Optional watermark (e.g., last processed commit SHA, timestamp).
                   If provided, only records modified after this point are returned.

        Yields:
            Records ready for chunking and indexing.
        """
        ...

    def change_signal(self) -> ChangeSignal:
        """
        Return change-detection signal for this source.

        Returns:
            A dict with one of:
              - {"watch": True}: use a file-watcher to detect changes
              - {"poll_interval": 3600}: poll for changes every N seconds
              Any other source-specific config can be included.
        """
        ...


@runtime_checkable
class SearchableSource(Protocol):
    """Federated source: source runs its own retrieval; kernel fuses results.

    The source already owns embeddings and ranking. The kernel never stores
    the source's content; it only merges ranked results from multiple sources.

    Attributes:
        source_kind: Stable identifier for this source type
                     (e.g., "memory", "jira").
    """

    source_kind: str

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        """
        Run the source's native search and return ranked references.

        Args:
            query: The search query string.
            k: Maximum number of results to return.
            filters: Optional source-specific filters (opaque to the kernel).

        Yields:
            ScoredRefs in descending score order. The kernel will use RRF
            to fuse these with results from other sources, so scores do not
            need to be normalized across sources.
        """
        ...
