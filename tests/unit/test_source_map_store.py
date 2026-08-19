from pathlib import Path

from mcp_markdown_ragdocs.indexing.record_ports import JsonSourceMapStore


def test_json_source_map_store_round_trips_membership(tmp_path: Path) -> None:
    """Preserve source membership across the application storage boundary.

    The JSON representation remains the durable contract used by indexing.
    """
    store = JsonSourceMapStore(tmp_path / "record-sources.json")
    records = {"source-1": ("key-a", "key-b")}

    store.save(records)

    assert store.load() == {"source-1": ["key-a", "key-b"]}


def test_json_source_map_store_recovers_from_invalid_content(tmp_path: Path) -> None:
    """Treat missing or malformed source maps as empty membership."""
    path = tmp_path / "record-sources.json"
    path.write_text("not-json", encoding="utf-8")

    assert JsonSourceMapStore(path).load() == {}
