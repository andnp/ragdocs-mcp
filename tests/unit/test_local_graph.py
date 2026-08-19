from collections.abc import Iterator, Sequence
from searchkernel.domain import GraphEdge, GraphNeighbor, RecordIdentity

from mcp_markdown_ragdocs.indexing.local_graph import (
    LocalBidirectionalGraphStore,
    SearchKernelGraphInstaller,
)


def _identity(source_id: str) -> RecordIdentity:
    return RecordIdentity(None, "document", source_id)


class _GraphBase:
    def __init__(
        self,
        outgoing: dict[str, tuple[GraphNeighbor, ...]],
    ) -> None:
        self.outgoing = outgoing

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, tuple[GraphNeighbor, ...]]:
        del depth, max_neighbors
        return {
            identity.storage_key: self.outgoing.get(identity.storage_key, ())
            for identity in identities
        }

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        del edges

    def delete_edges(self, edges: Sequence[GraphEdge]) -> None:
        del edges

    def graph_integrity_errors(self) -> list[str]:
        return []


class _Graph(_GraphBase):
    def __init__(
        self,
        outgoing: dict[str, tuple[GraphNeighbor, ...]],
        incoming: dict[str, tuple[GraphNeighbor, ...]],
    ) -> None:
        super().__init__(outgoing)
        self.incoming = incoming

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, tuple[GraphNeighbor, ...]]:
        del depth, max_neighbors
        return {
            identity.storage_key: self.incoming.get(identity.storage_key, ())
            for identity in identities
        }

class _OutgoingOnlyGraph(_GraphBase):
    def __init__(self, outgoing: dict[str, tuple[GraphNeighbor, ...]]) -> None:
        super().__init__(outgoing)


def test_graph_adapter_preserves_direction_modes() -> None:
    """Graph direction modes preserve public outgoing and incoming traversal.

    Direction changes must not alter the underlying graph store.
    """
    source = _identity("source")
    target = _identity("target")
    graph = _Graph(
        {source.storage_key: (GraphNeighbor(target, "links_to", 1.0),)},
        {target.storage_key: (GraphNeighbor(source, "links_to", 1.0),)},
    )
    adapter = LocalBidirectionalGraphStore(graph, lambda: (source, target))

    assert adapter.neighbors(source)[0].identity == target
    adapter.set_direction("incoming")
    assert adapter.neighbors(target)[0].identity == source
    adapter.set_direction("both")
    assert {item.identity for item in adapter.neighbors(source)} == {target}


def test_graph_adapter_repeats_fallback_identity_scans() -> None:
    """Fallback inbound traversal uses a fresh identity iterable per call.

    This protects repeated reverse lookups when SearchKernel lacks inbound APIs.
    """
    target = _identity("target")
    sources = [_identity(f"source-{index}") for index in range(101)]
    outgoing: dict[str, tuple[GraphNeighbor, ...]] = {
        source.storage_key: (GraphNeighbor(target, "links_to", float(index)),)
        for index, source in enumerate(sources)
    }
    calls = 0

    def identities() -> Iterator[RecordIdentity]:
        nonlocal calls
        calls += 1
        return iter(sources)

    adapter = LocalBidirectionalGraphStore(_OutgoingOnlyGraph(outgoing), identities)

    first = adapter.incoming_neighbors_many([target], depth=1, max_neighbors=2)
    second = adapter.incoming_neighbors_many([target], depth=1, max_neighbors=2)

    assert [item.weight for item in first[target.storage_key]] == [100.0, 99.0]
    assert second == first
    assert calls == 2


def test_graph_installer_owns_private_pipeline_assignment() -> None:
    """The temporary installer places the application graph in the pipeline slot.

    The private assignment must remain isolated behind this temporary seam.
    """
    class Pipeline:
        _graph_store: object

    class Kernel:
        pipeline: Pipeline

    pipeline = Pipeline()
    kernel = Kernel()
    kernel.pipeline = pipeline
    graph = object()

    SearchKernelGraphInstaller(kernel).install(graph)

    assert pipeline._graph_store is graph
