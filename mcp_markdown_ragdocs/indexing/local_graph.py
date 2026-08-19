"""Local SearchKernel graph capability adapter."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import islice
from typing import Protocol, runtime_checkable

from searchkernel.api import (
    GraphEdge,
    GraphNeighbor,
    LocalRecordKernel,
    RecordIdentity,
)


class LocalGraphStore(Protocol):
    """Public local graph operations consumed by the application adapter."""

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Mapping[str, Sequence[GraphNeighbor]]: ...

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def delete_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def graph_integrity_errors(self) -> list[str]: ...


@runtime_checkable
class IncomingLocalGraphStore(LocalGraphStore, Protocol):
    """Optional public inbound traversal supported by newer SearchKernel versions."""

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Mapping[str, Sequence[GraphNeighbor]]: ...


class GraphPipelineHost(Protocol):
    """Application-side view of a kernel pipeline used by the temporary hook."""

    @property
    def pipeline(self) -> object: ...


class LocalBidirectionalGraphStore:
    """Adapt the local public graph store to the application's direction port."""

    def __init__(
        self,
        graph_store: LocalGraphStore,
        identities: Callable[[], Iterable[RecordIdentity]],
    ) -> None:
        self._graph_store = graph_store
        self._identities = identities
        self._direction = contextvars.ContextVar(
            "graph_direction",
            default="outgoing",
        )

    def set_direction(self, direction: str | bool) -> None:
        normalized = (
            "incoming"
            if direction is True
            else "outgoing"
            if direction is False
            else direction
        )
        self._direction.set(normalized)

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        direction = self._direction.get()
        if direction == "incoming":
            return self.incoming_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
        if direction == "both":
            outgoing = self._outgoing_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
            incoming = self.incoming_neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
            return {
                identity.storage_key: _merge_graph_neighbors(
                    outgoing.get(identity.storage_key, ()),
                    incoming.get(identity.storage_key, ()),
                    max_neighbors,
                )
                for identity in identities
            }
        return self._outgoing_neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

    def _outgoing_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int = 1,
        max_neighbors: int | None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        return dict(
            self._graph_store.neighbors_many(
                identities,
                depth=depth,
                max_neighbors=max_neighbors,
            )
        )

    def neighbors(
        self,
        identity: RecordIdentity,
        *,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self.neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors(
        self,
        identity: RecordIdentity,
        *,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self.incoming_neighbors_many(
            [identity],
            depth=depth,
            max_neighbors=max_neighbors,
        )[identity.storage_key]

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        if isinstance(self._graph_store, IncomingLocalGraphStore):
            return dict(
                self._graph_store.incoming_neighbors_many(
                    identities,
                    depth=depth,
                    max_neighbors=max_neighbors,
                )
            )

        requested = {identity.storage_key for identity in identities}
        incoming: dict[str, list[GraphNeighbor]] = {
            identity.storage_key: [] for identity in identities
        }
        all_identities = iter(self._identities())
        outgoing: dict[str, Sequence[GraphNeighbor]] = {}
        while identity_batch := list(islice(all_identities, 100)):
            outgoing.update(
                self._graph_store.neighbors_many(
                    identity_batch,
                    depth=depth,
                    max_neighbors=None,
                )
            )
        for source_key, neighbors in outgoing.items():
            source = RecordIdentity.from_storage_key(source_key)
            for neighbor in neighbors:
                if neighbor.identity.storage_key in requested:
                    incoming[neighbor.identity.storage_key].append(
                        GraphNeighbor(source, neighbor.edge_type, neighbor.weight)
                    )
        for key, neighbors in incoming.items():
            neighbors.sort(key=lambda item: (-item.weight, item.identity.storage_key))
            if max_neighbors is not None:
                incoming[key] = neighbors[:max_neighbors]
        return {key: tuple(neighbors) for key, neighbors in incoming.items()}

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        self._graph_store.upsert_edges(edges)

    def delete_edges(self, edges: Sequence[GraphEdge]) -> None:
        self._graph_store.delete_edges(edges)

    def outgoing_edges(
        self,
        identities: Sequence[RecordIdentity],
        edge_type: str,
    ) -> list[GraphEdge]:
        """List the stored outgoing edges of one type, ignoring direction state."""
        neighbors = self._graph_store.neighbors_many(identities, depth=1)
        return [
            GraphEdge(identity, neighbor.identity, edge_type, neighbor.weight)
            for identity in identities
            for neighbor in neighbors.get(identity.storage_key, ())
            if neighbor.edge_type == edge_type
        ]

    def graph_integrity_errors(self) -> list[str]:
        return self._graph_store.graph_integrity_errors()


class SearchKernelGraphInstaller:
    """Temporary seam for SearchKernel's missing public graph replacement hook.

    Remove this adapter when SearchKernel exposes a supported pipeline graph
    replacement API; application callers should continue using the graph port.
    """

    def __init__(self, kernel: GraphPipelineHost) -> None:
        self._kernel = kernel

    def install(self, graph: object) -> None:
        """Install the application graph through the current private hook."""
        setattr(self._kernel.pipeline, "_graph_store", graph)


def _merge_graph_neighbors(
    first: Sequence[GraphNeighbor],
    second: Sequence[GraphNeighbor],
    max_neighbors: int | None,
) -> list[GraphNeighbor]:
    merged = {
        neighbor.identity.storage_key: neighbor for neighbor in (*first, *second)
    }
    neighbors = sorted(
        merged.values(),
        key=lambda item: (-item.weight, item.identity.storage_key),
    )
    return neighbors if max_neighbors is None else neighbors[:max_neighbors]


def install_bidirectional_graph_store(
    kernel: LocalRecordKernel,
    identities: Callable[[], Iterable[RecordIdentity]],
) -> LocalBidirectionalGraphStore:
    """Install and return the local graph capability adapter."""
    graph = LocalBidirectionalGraphStore(kernel.graph_store, identities)
    SearchKernelGraphInstaller(kernel).install(graph)
    return graph


__all__ = [
    "LocalBidirectionalGraphStore",
    "SearchKernelGraphInstaller",
    "install_bidirectional_graph_store",
]
