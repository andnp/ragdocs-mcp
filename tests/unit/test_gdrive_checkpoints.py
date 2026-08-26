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

    first_checkpoint = store.load(first)
    second_checkpoint = store.load(second)
    assert first_checkpoint is not None
    assert second_checkpoint is not None
    assert first_checkpoint.inventory_start_token == "start-a"
    assert second_checkpoint.inventory_start_token == "start-b"


def test_checkpoint_namespace_preserves_scoped_drive_identity(tmp_path: Path) -> None:
    """
    Persist checkpoints for Drive identities that contain a scope delimiter.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation-shared-drive:drive-1")

    store.begin_inventory(namespace, "start-token")

    assert store.load(namespace) is not None


def test_inventory_failure_count_grows_and_is_cleared_on_success(tmp_path: Path) -> None:
    """
    Track consecutive inventory failures durably, and forget them once a run
    succeeds so a scope that recovers is not still treated as unhealthy.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation")
    store.begin_inventory(namespace, "start-token")

    store.record_inventory_failure(namespace)
    second = store.record_inventory_failure(namespace)

    assert second.inventory_failure_count == 2
    reloaded = store.load(namespace)
    assert reloaded is not None
    assert reloaded.inventory_failure_count == 2

    cleared = store.record_inventory_success(namespace)

    assert cleared is not None
    assert cleared.inventory_failure_count == 0
    reloaded_after_success = store.load(namespace)
    assert reloaded_after_success is not None
    assert reloaded_after_success.inventory_failure_count == 0


def test_inventory_failure_before_any_checkpoint_exists(tmp_path: Path) -> None:
    """
    Record a failure even when the first inventory call fails before a
    checkpoint (e.g. fetching the start token) has ever been persisted.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation")

    checkpoint = store.record_inventory_failure(namespace)

    assert checkpoint.inventory_failure_count == 1
    assert checkpoint.inventory_start_token is None


def test_poisoning_the_inventory_token_clears_it_but_keeps_the_start_token(tmp_path: Path) -> None:
    """
    Clear only the page token so pagination restarts from the beginning of
    the current inventory epoch rather than starting an entirely new one.
    """
    store = GDriveSyncCheckpointStore(tmp_path)
    namespace = checkpoint_namespace("scope-generation")
    store.begin_inventory(namespace, "start-token")
    store.persist_inventory_batch_after_index(namespace, page_token="bad-token", batch=1)
    store.record_inventory_failure(namespace)
    store.record_inventory_failure(namespace)
    store.record_inventory_failure(namespace)

    poisoned = store.poison_inventory_token(namespace)

    assert poisoned.inventory_page_token is None
    assert poisoned.inventory_start_token == "start-token"
    assert poisoned.inventory_failure_count == 3
