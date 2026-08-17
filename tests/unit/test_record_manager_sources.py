"""Tests for logical content-source registration on the canonical manager."""

from types import SimpleNamespace


def test_record_manager_registers_sources_through_storage(record_manager, monkeypatch):
    """
    Register a source through the manager's public boundary.
    Keep source registration behind the existing storage capability.
    """
    source = SimpleNamespace(source_kind="gdrive")
    registered: list[object] = []

    monkeypatch.setattr(
        record_manager.storage,
        "register_content_source",
        registered.append,
    )

    record_manager.register_content_source(source)

    assert record_manager.get_content_source("gdrive") is source
    assert registered == [source]
