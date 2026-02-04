"""
Unit tests for IndexSyncPublisher and IndexSyncReceiver.

Tests version-based index synchronization, snapshot creation, and cleanup.
"""

import struct
from pathlib import Path

import pytest

from src.ipc.index_sync import IndexSyncPublisher, IndexSyncReceiver


@pytest.fixture
def snapshot_base(tmp_path: Path) -> Path:
    """Create a temporary directory for snapshots."""
    base = tmp_path / "snapshots"
    base.mkdir()
    return base


class TestIndexSyncPublisher:
    """Tests for IndexSyncPublisher snapshot publishing."""

    def test_publish_creates_version_file(self, snapshot_base: Path):
        """Verify publish creates version file with correct format."""
        publisher = IndexSyncPublisher(snapshot_base)

        def persist_callback(path: Path) -> None:
            (path / "test.txt").write_text("data")

        version = publisher.publish(persist_callback)

        version_file = snapshot_base / "version.bin"
        assert version_file.exists()

        # Verify binary format (little-endian unsigned int)
        data = version_file.read_bytes()
        assert len(data) == 4
        parsed_version = struct.unpack("<I", data)[0]
        assert parsed_version == version

    def test_publish_creates_snapshot_directory(self, snapshot_base: Path):
        """Verify publish creates versioned snapshot directory."""
        publisher = IndexSyncPublisher(snapshot_base)

        def persist_callback(path: Path) -> None:
            (path / "index.json").write_text("{}")

        version = publisher.publish(persist_callback)

        snapshot_dir = snapshot_base / f"v{version}"
        assert snapshot_dir.is_dir()
        assert (snapshot_dir / "index.json").exists()

    def test_publish_increments_version(self, snapshot_base: Path):
        """Verify each publish increments the version number."""
        publisher = IndexSyncPublisher(snapshot_base)

        def noop_persist(path: Path) -> None:
            pass

        v1 = publisher.publish(noop_persist)
        v2 = publisher.publish(noop_persist)
        v3 = publisher.publish(noop_persist)

        assert v2 == v1 + 1
        assert v3 == v2 + 1

    def test_publish_calls_persist_callback(self, snapshot_base: Path):
        """Verify persist_callback receives correct path."""
        publisher = IndexSyncPublisher(snapshot_base)
        received_paths: list[Path] = []

        def capture_path(path: Path) -> None:
            received_paths.append(path)

        version = publisher.publish(capture_path)

        assert len(received_paths) == 1
        expected_path = snapshot_base / f"v{version}"
        assert received_paths[0] == expected_path

    def test_cleanup_old_snapshots(self, snapshot_base: Path):
        """Verify old snapshots are cleaned up, keeping only recent ones."""
        publisher = IndexSyncPublisher(snapshot_base)

        def create_file(path: Path) -> None:
            (path / "data.txt").write_text("content")

        # Create 5 snapshots
        versions = []
        for _ in range(5):
            versions.append(publisher.publish(create_file))

        # Should keep only the last 2 (default)
        existing = sorted([d.name for d in snapshot_base.iterdir() if d.is_dir()])

        # v4, v5 should exist (v1, v2, v3 cleaned up)
        assert len(existing) == 2
        assert f"v{versions[-1]}" in existing  # Latest
        assert f"v{versions[-2]}" in existing  # Second latest


class TestIndexSyncReceiver:
    """Tests for IndexSyncReceiver snapshot loading."""

    def test_check_for_update_no_version_file(self, snapshot_base: Path):
        """Verify check_for_update returns False when no version file exists."""
        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)

        assert receiver.check_for_update() is False

    def test_check_for_update_with_version(self, snapshot_base: Path):
        """Verify check_for_update returns True when new version available."""
        # Write version file
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 1))

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)

        assert receiver.check_for_update() is True

    def test_check_for_update_same_version(self, snapshot_base: Path):
        """Verify check_for_update returns False when version unchanged."""
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 42))

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        receiver._current_version = 42  # Simulate already loaded version

        assert receiver.check_for_update() is False

    def test_reload_if_needed_returns_false_when_no_update(self, snapshot_base: Path):
        """Verify reload_if_needed returns False when no update available."""
        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)

        assert receiver.reload_if_needed() is False

    def test_reload_if_needed_calls_reload_callback(self, snapshot_base: Path):
        """Verify reload_if_needed calls reload_callback with snapshot path."""
        # Setup: create version file and snapshot directory
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 7))

        snapshot_dir = snapshot_base / "v7"
        snapshot_dir.mkdir()
        (snapshot_dir / "index.dat").write_text("data")

        loaded_info: list[tuple[Path, int]] = []

        def capture_load(path: Path, version: int) -> None:
            loaded_info.append((path, version))

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=capture_load)

        result = receiver.reload_if_needed()

        assert result is True
        assert len(loaded_info) == 1
        assert loaded_info[0][0] == snapshot_dir
        assert loaded_info[0][1] == 7

    def test_reload_updates_current_version(self, snapshot_base: Path):
        """Verify reload updates internal version tracking."""
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 99))

        snapshot_dir = snapshot_base / "v99"
        snapshot_dir.mkdir()

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        assert receiver._current_version == 0

        receiver.reload_if_needed()

        assert receiver._current_version == 99

    def test_initialize_from_loaded_version(self, snapshot_base: Path):
        """Verify initialize_from_loaded_version sets current_version.

        This is used when indices are loaded externally during startup,
        bypassing the normal reload_if_needed flow.
        """
        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        assert receiver._current_version == 0
        assert receiver.current_version == 0

        receiver.initialize_from_loaded_version(42)

        assert receiver._current_version == 42
        assert receiver.current_version == 42

    def test_initialize_from_loaded_version_prevents_redundant_reload(self, snapshot_base: Path):
        """Verify initialized version prevents reload of same version.

        If we load version 5 at startup and call initialize_from_loaded_version(5),
        subsequent check_for_update should return False for version 5.
        """
        # Setup: version file says v5 is current
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 5))

        reload_count = [0]

        def count_reloads(path: Path, version: int) -> None:
            reload_count[0] += 1

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=count_reloads)

        # Simulate: indices already loaded at startup, skip reload via initialize
        receiver.initialize_from_loaded_version(5)

        # check_for_update should return False since we're at version 5
        assert receiver.check_for_update() is False

        # reload_if_needed should not call callback
        receiver.reload_if_needed()
        assert reload_count[0] == 0


class TestPublisherReceiverIntegration:
    """Integration tests for publisher/receiver coordination."""

    def test_publisher_receiver_sync(self, snapshot_base: Path):
        """
        Verify receiver can load snapshots created by publisher.

        This tests the full publish -> check -> reload cycle.
        """
        publisher = IndexSyncPublisher(snapshot_base)

        loaded_content: list[str] = []

        def read_content(path: Path, version: int) -> None:
            loaded_content.append((path / "data.txt").read_text())

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=read_content)

        # Initial state: no updates
        assert receiver.check_for_update() is False

        # Publish first snapshot
        def write_content(path: Path) -> None:
            (path / "data.txt").write_text("version 1")

        publisher.publish(write_content)

        # Receiver should see update
        assert receiver.check_for_update() is True

        # Reload
        result = receiver.reload_if_needed()

        assert result is True
        assert loaded_content == ["version 1"]

        # No more updates available
        assert receiver.check_for_update() is False

        # Publish second snapshot
        def write_v2(path: Path) -> None:
            (path / "data.txt").write_text("version 2")

        publisher.publish(write_v2)

        # Receiver should see new update
        assert receiver.check_for_update() is True
        loaded_content.clear()
        receiver.reload_if_needed()
        assert loaded_content == ["version 2"]
