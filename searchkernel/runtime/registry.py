"""Registry of named SearchableSources for federation.

Lets callers register federated/local sources once and select a subset by
source_kind at query time (search_anything(sources=["local", "memory"])),
instead of every caller wiring the full source list by hand.
"""

from searchkernel.ports.content_source import SearchableSource


class SourceRegistry:
    """Holds SearchableSources keyed by their source_kind."""

    def __init__(self) -> None:
        self._sources: dict[str, SearchableSource] = {}

    def register(self, source: SearchableSource) -> None:
        """Register a source, keyed by its source_kind (overwrites any prior)."""
        self._sources[source.source_kind] = source

    def get(self, source_kind: str) -> SearchableSource | None:
        """Look up a single registered source by source_kind."""
        return self._sources.get(source_kind)

    def select(self, source_kinds: list[str] | None = None) -> list[SearchableSource]:
        """Resolve a list of source_kinds to their registered sources.

        Unknown source_kinds are silently skipped. If source_kinds is None,
        every registered source is returned.
        """
        if source_kinds is None:
            return list(self._sources.values())
        return [
            self._sources[kind] for kind in source_kinds if kind in self._sources
        ]

    def all(self) -> list[SearchableSource]:
        """Return every registered source."""
        return list(self._sources.values())
