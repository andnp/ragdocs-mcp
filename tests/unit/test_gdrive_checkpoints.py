"""Tests for Google Drive synchronization checkpoints."""

import json
from pathlib import Path

import pytest

from mcp_markdown_ragdocs.gdrive.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    GDriveSyncCheckpointStore,
    checkpoint_namespace,
)


def test_start_token_is_persisted_before_inventory(tmp_path: Path) -> None:
    """
    Make the inventory reader observe its start token from durable state.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation")

    store.begin_inventory(namespace, "start-token")
    inventory_checkpoint = store.load(namespace)

    assert inventory_checkpoint is not None
    assert inventory_checkpoint.inventory_start_token == "start-token"
    assert inventory_checkpoint.inventory_page_token is None
    assert inventory_checkpoint.inventory_batch == 0
    assert inventory_checkpoint.changes_token is None

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoints": {
            namespace: inventory_checkpoint.to_payload(),
        },
    }


def test_index_mutation_precedes_inventory_cursor_persistence(tmp_path: Path) -> None:
    """
    Persist an inventory cursor only after the index mutation has completed.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation")
    store.begin_inventory(namespace, "start-token")
    events: list[str] = []

    events.append("index-mutation")
    checkpoint = store.persist_inventory_batch_after_index(
        namespace,
        page_token="page-2",
        batch=1,
    )
    events.append("cursor-persistence")

    assert events == ["index-mutation", "cursor-persistence"]
    assert store.load(namespace) == checkpoint
    assert checkpoint.inventory_start_token == "start-token"
    assert checkpoint.inventory_page_token == "page-2"
    assert checkpoint.inventory_batch == 1

    with pytest.raises(ValueError, match="advance in order"):
        store.persist_inventory_batch_after_index(
            namespace,
            page_token="page-3",
            batch=3,
        )


def test_checkpoint_namespaces_are_isolated(tmp_path: Path) -> None:
    """
    Keep independent scope generations from overwriting each other.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    first = checkpoint_namespace("scope-a")
    second = checkpoint_namespace("scope-b")

    store.begin_inventory(first, "start-a")
    store.begin_inventory(second, "start-b")

    assert store.load(first).inventory_start_token == "start-a"
    assert store.load(second).inventory_start_token == "start-b"
