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
        """Verify persist_callback receives temp directory path (atomic pattern)."""
        publisher = IndexSyncPublisher(snapshot_base)
        received_paths: list[Path] = []

        def capture_path(path: Path) -> None:
            received_paths.append(path)

        version = publisher.publish(capture_path)

        assert len(received_paths) == 1
        # Callback receives temp path; content is moved to final path after
        expected_temp_path = snapshot_base / f"v{version}.tmp"
        assert received_paths[0] == expected_temp_path
        # Final path should exist after publish completes
        assert (snapshot_base / f"v{version}").exists()

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


class TestAtomicPublishBehavior:
    """Tests for atomic two-phase commit publish pattern."""

    def test_publish_uses_temp_directory_then_renames(self, snapshot_base: Path):
        """Verify publish creates temp dir, then renames to final location."""
        publisher = IndexSyncPublisher(snapshot_base)
        observed_paths: list[Path] = []

        def observe_path(path: Path) -> None:
            observed_paths.append(path)
            assert path.name.endswith(".tmp"), "Should write to temp dir first"
            (path / "test.txt").write_text("data")

        version = publisher.publish(observe_path)

        # After publish, temp dir should not exist
        temp_dir = snapshot_base / f"v{version}.tmp"
        assert not temp_dir.exists()

        # Final dir should exist with content
        final_dir = snapshot_base / f"v{version}"
        assert final_dir.exists()
        assert (final_dir / "test.txt").read_text() == "data"

    def test_partial_failure_persist_callback_cleans_up(self, snapshot_base: Path):
        """Verify partial failure in persist_callback doesn't leave inconsistent state."""
        publisher = IndexSyncPublisher(snapshot_base)

        def failing_persist(path: Path) -> None:
            (path / "partial.txt").write_text("partial data")
            raise RuntimeError("Simulated failure")

        with pytest.raises(RuntimeError, match="Simulated failure"):
            publisher.publish(failing_persist)

        # No snapshot directories should remain
        snapshot_dirs = [d for d in snapshot_base.iterdir() if d.is_dir()]
        assert snapshot_dirs == []

        # No temp version file should remain
        temp_version = snapshot_base / "version.tmp"
        assert not temp_version.exists()

        # Version should not have incremented
        assert publisher.version == 0

    def test_partial_failure_version_write_cleans_up(
        self, snapshot_base: Path, monkeypatch
    ):
        """
        Verify failure during version file write doesn't leave inconsistent state.
        """
        publisher = IndexSyncPublisher(snapshot_base)

        # First, let write succeed to create temp version file, then fail on rename
        write_count = 0

        original_write_bytes = Path.write_bytes

        def failing_write_bytes(self, data):
            nonlocal write_count
            write_count += 1
            if str(self).endswith("version.tmp") and write_count > 0:
                # First create the file so we test cleanup
                original_write_bytes(self, data)
                raise OSError("Simulated write failure")
            return original_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

        def noop_persist(path: Path) -> None:
            (path / "data.txt").write_text("content")

        with pytest.raises(OSError, match="Simulated write failure"):
            publisher.publish(noop_persist)

        # No temp files should remain
        temp_files = [f for f in snapshot_base.iterdir() if f.name.endswith(".tmp")]
        assert temp_files == []

        # Version should not have incremented
        assert publisher.version == 0

    def test_temp_files_cleaned_on_startup(self, snapshot_base: Path):
        """Verify orphaned temp files from crashed publishes are cleaned on startup."""
        # Simulate crashed publish by leaving temp files
        temp_dir = snapshot_base / "v5.tmp"
        temp_dir.mkdir(parents=True)
        (temp_dir / "partial.txt").write_text("orphaned")

        temp_version = snapshot_base / "version.tmp"
        temp_version.write_bytes(struct.pack("<I", 5))

        # Also leave a valid snapshot to ensure it's preserved
        valid_dir = snapshot_base / "v3"
        valid_dir.mkdir()
        (valid_dir / "data.txt").write_text("valid")

        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 3))

        # Creating publisher should clean up temp files
        publisher = IndexSyncPublisher(snapshot_base)

        # Temp files should be gone
        assert not temp_dir.exists()
        assert not temp_version.exists()

        # Valid snapshot should remain
        assert valid_dir.exists()
        assert version_file.exists()

        # Version should be loaded correctly
        assert publisher.version == 3

    def test_publish_overwrites_existing_snapshot_dir(self, snapshot_base: Path):
        """Verify publish handles case where target snapshot dir already exists."""
        publisher = IndexSyncPublisher(snapshot_base)

        # Manually create a conflicting directory (simulates race/corruption)
        conflicting_dir = snapshot_base / "v1"
        conflicting_dir.mkdir(parents=True)
        (conflicting_dir / "stale.txt").write_text("stale")

        def write_new_data(path: Path) -> None:
            (path / "fresh.txt").write_text("fresh")

        version = publisher.publish(write_new_data)

        assert version == 1
        final_dir = snapshot_base / "v1"
        # Old file should be gone, new file should exist
        assert not (final_dir / "stale.txt").exists()
        assert (final_dir / "fresh.txt").read_text() == "fresh"


class TestSnapshotFallbackResilience:
    """Tests for snapshot version mismatch fallback behavior."""

    def test_reload_falls_back_when_pointed_version_missing(self, snapshot_base: Path):
        """
        Verify reload falls back to highest available snapshot when
        version.bin points to non-existent directory.
        """
        # version.bin points to v999 (doesn't exist)
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 999))

        # But v100 and v150 exist on disk
        (snapshot_base / "v100").mkdir()
        (snapshot_base / "v100" / "data.txt").write_text("old")
        (snapshot_base / "v150").mkdir()
        (snapshot_base / "v150" / "data.txt").write_text("newer")

        loaded_info: list[tuple[Path, int]] = []

        def capture_load(path: Path, version: int) -> None:
            loaded_info.append((path, version))

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=capture_load)

        result = receiver.reload_if_needed()

        # Should succeed using fallback to v150 (highest available)
        assert result is True
        assert len(loaded_info) == 1
        assert loaded_info[0][0] == snapshot_base / "v150"
        assert loaded_info[0][1] == 150
        assert receiver._current_version == 150

    def test_reload_returns_false_when_no_snapshots_available(
        self, snapshot_base: Path
    ):
        """
        Verify reload returns False when version.bin points to missing
        snapshot and no other snapshots exist.
        """
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 42))

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)

        result = receiver.reload_if_needed()

        assert result is False
        assert receiver._current_version == 0

    def test_find_available_snapshots_returns_sorted_list(self, snapshot_base: Path):
        """Verify _find_available_snapshots returns versions sorted descending."""
        (snapshot_base / "v10").mkdir()
        (snapshot_base / "v5").mkdir()
        (snapshot_base / "v200").mkdir()
        (snapshot_base / "v50").mkdir()
        (snapshot_base / "not-a-version").mkdir()  # Should be ignored

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        available = receiver._find_available_snapshots()

        versions = [v for v, _ in available]
        assert versions == [200, 50, 10, 5]

    def test_find_available_snapshots_empty_when_no_snapshots(
        self, snapshot_base: Path
    ):
        """Verify _find_available_snapshots returns empty list when no snapshots."""
        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        available = receiver._find_available_snapshots()
        assert available == []

    def test_reload_skips_fallback_if_already_loaded_higher(self, snapshot_base: Path):
        """
        Verify fallback doesn't downgrade if current version is higher than
        available fallback snapshots.
        """
        # version.bin points to v999 (doesn't exist)
        version_file = snapshot_base / "version.bin"
        version_file.write_bytes(struct.pack("<I", 999))

        # v50 exists but is lower than already-loaded version
        (snapshot_base / "v50").mkdir()

        receiver = IndexSyncReceiver(snapshot_base, reload_callback=lambda p, v: None)
        receiver._current_version = 100  # Already loaded v100

        result = receiver.reload_if_needed()

        # Should return False since fallback (v50) < current (v100)
        assert result is False
        assert receiver._current_version == 100
