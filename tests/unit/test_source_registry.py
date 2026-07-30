"""Unit tests for SourceRegistry: register/select/get named SearchableSources."""

from collections.abc import Iterable
from typing import Any

from searchkernel.domain import ScoredRef
from searchkernel.runtime.registry import SourceRegistry


class _StubSource:
    source_kind: str

    def __init__(self, source_kind: str):
        self.source_kind = source_kind

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [ScoredRef(source_id="1", score=1.0, source_kind=self.source_kind)]


def test_register_then_get_returns_the_source():
    registry = SourceRegistry()
    source = _StubSource("local")

    registry.register(source)

    assert registry.get("local") is source


def test_get_unknown_source_returns_none():
    registry = SourceRegistry()

    assert registry.get("nonexistent") is None


def test_select_none_returns_all_registered_sources():
    registry = SourceRegistry()
    local = _StubSource("local")
    memory = _StubSource("memory")
    registry.register(local)
    registry.register(memory)

    selected = registry.select(None)

    assert set(selected) == {local, memory}


def test_select_names_filters_to_requested_sources():
    registry = SourceRegistry()
    local = _StubSource("local")
    memory = _StubSource("memory")
    registry.register(local)
    registry.register(memory)

    selected = registry.select(["memory"])

    assert selected == [memory]


def test_select_skips_unknown_names():
    registry = SourceRegistry()
    local = _StubSource("local")
    registry.register(local)

    selected = registry.select(["local", "nonexistent"])

    assert selected == [local]


def test_register_overwrites_existing_source_kind():
    registry = SourceRegistry()
    first = _StubSource("local")
    second = _StubSource("local")
    registry.register(first)

    registry.register(second)

    assert registry.get("local") is second


def test_all_returns_every_registered_source():
    registry = SourceRegistry()
    local = _StubSource("local")
    memory = _StubSource("memory")
    registry.register(local)
    registry.register(memory)

    assert set(registry.all()) == {local, memory}
