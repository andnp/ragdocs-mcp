import asyncio
import logging
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexSyncPublisher:
    def __init__(self, snapshot_base: Path):
        self._snapshot_base = snapshot_base
        self._version = 0
        self._version_file = snapshot_base / "version.bin"
        self._load_current_version()

    def _load_current_version(self) -> None:
        # Clean up any orphaned temp files from crashed publishes
        self._cleanup_orphaned_temp_files()

        if self._version_file.exists():
            try:
                data = self._version_file.read_bytes()
                if len(data) >= 4:
                    self._version = struct.unpack("<I", data[:4])[0]
            except (OSError, struct.error):
                logger.warning("Failed to load version file, starting from 0")
                self._version = 0

    def _cleanup_orphaned_temp_files(self) -> None:
        """Remove any .tmp files/directories left from crashed publishes."""
        if not self._snapshot_base.exists():
            return

        for item in self._snapshot_base.iterdir():
            if item.name.endswith(".tmp"):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                        logger.debug("Cleaned up orphaned temp directory: %s", item)
                    else:
                        item.unlink()
                        logger.debug("Cleaned up orphaned temp file: %s", item)
                except OSError as e:
                    logger.warning("Failed to clean up temp file %s: %s", item, e)

    def publish(self, persist_callback: Callable[[Path], None]) -> int:
        self._snapshot_base.mkdir(parents=True, exist_ok=True)

        new_version = self._version + 1
        snapshot_dir = self._snapshot_base / f"v{new_version}"
        temp_snapshot_dir = self._snapshot_base / f"v{new_version}.tmp"
        temp_version_file = self._version_file.with_suffix(".tmp")

        try:
            # Phase 1: Create snapshot in temp directory
            temp_snapshot_dir.mkdir(parents=True, exist_ok=True)
            persist_callback(temp_snapshot_dir)

            # Phase 2: Write version to temp file
            temp_version_file.write_bytes(struct.pack("<I", new_version))

            # Phase 3: Atomic moves (as atomic as filesystem allows)
            # On POSIX, rename is atomic if on same filesystem
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            temp_snapshot_dir.rename(snapshot_dir)
            temp_version_file.replace(self._version_file)

            # Phase 4: Write completion marker for two-phase commit validation
            # Readers verify this marker before loading to ensure consistency
            marker_file = snapshot_dir / "complete.marker"
            marker_file.write_text(str(new_version))

            self._version = new_version
            logger.info("Published index snapshot v%d to %s", new_version, snapshot_dir)

            self._cleanup_old_snapshots(keep=2)

            return new_version
        except Exception:
            # Cleanup any partial state
            logger.exception("Snapshot publish failed, cleaning up")
            if temp_snapshot_dir.exists():
                shutil.rmtree(temp_snapshot_dir)
            if temp_version_file.exists():
                temp_version_file.unlink()
            raise

    def _cleanup_old_snapshots(self, keep: int = 2) -> None:
        if not self._snapshot_base.exists():
            return

        snapshot_dirs: list[tuple[int, Path]] = []
        for item in self._snapshot_base.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    version = int(item.name[1:])
                    snapshot_dirs.append((version, item))
                except ValueError:
                    continue

        snapshot_dirs.sort(key=lambda x: x[0], reverse=True)

        for _, old_dir in snapshot_dirs[keep:]:
            try:
                shutil.rmtree(old_dir)
                logger.debug("Cleaned up old snapshot: %s", old_dir)
            except OSError as e:
                logger.warning("Failed to clean up snapshot %s: %s", old_dir, e)

    @property
    def version(self) -> int:
        return self._version


class IndexSyncReceiver:
    def __init__(self, snapshot_base: Path, reload_callback: Callable[[Path, int], None]):
        self._snapshot_base = snapshot_base
        self._version_file = snapshot_base / "version.bin"
        self._current_version = 0
        self._reload_callback = reload_callback

    def initialize_from_loaded_version(self, version: int) -> None:
        """Set the current version after indices were loaded externally.

        Use this when indices are loaded during startup (not via reload_if_needed)
        to ensure is_ready() returns True.
        """
        self._current_version = version
        logger.info("IndexSyncReceiver initialized to version %d", version)

    def _read_published_version(self) -> int | None:
        if not self._version_file.exists():
            return None
        try:
            data = self._version_file.read_bytes()
            if len(data) >= 4:
                return struct.unpack("<I", data[:4])[0]
            return None
        except (OSError, struct.error):
            return None

    def _find_available_snapshots(self) -> list[tuple[int, Path]]:
        """Return sorted list of (version, path) tuples for existing snapshots, newest first."""
        if not self._snapshot_base.exists():
            return []

        snapshots = []
        for item in self._snapshot_base.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    version = int(item.name[1:])
                    snapshots.append((version, item))
                except ValueError:
                    continue

        return sorted(snapshots, key=lambda x: x[0], reverse=True)

    def _validate_snapshot(self, snapshot_dir: Path, expected_version: int) -> bool:
        """Validate that a snapshot has a complete.marker with the expected version."""
        marker = snapshot_dir / "complete.marker"
        try:
            return marker.read_text().strip() == str(expected_version)
        except (OSError, ValueError):
            return False

    def check_for_update(self) -> bool:
        published_version = self._read_published_version()
        if published_version is None:
            return False
        return published_version > self._current_version

    def reload_if_needed(self) -> bool:
        published_version = self._read_published_version()
        target_version = published_version

        # Determine snapshot directory, with fallback if pointed version is missing or invalid
        snapshot_dir: Path | None = None
        if published_version is not None:
            snapshot_dir = self._snapshot_base / f"v{published_version}"
            # Check both existence and marker validation
            if not snapshot_dir.exists() or not self._validate_snapshot(
                snapshot_dir, published_version
            ):
                # Fallback to highest available valid snapshot
                available = self._find_available_snapshots()
                fallback_found = False
                for version, path in available:
                    if self._validate_snapshot(path, version):
                        if not snapshot_dir.exists():
                            logger.warning(
                                "version.bin points to v%d but directory missing. "
                                "Falling back to v%d.",
                                published_version,
                                version,
                            )
                        else:
                            logger.warning(
                                "Snapshot v%d has invalid marker. Falling back to v%d.",
                                published_version,
                                version,
                            )
                        snapshot_dir = path
                        target_version = version
                        fallback_found = True
                        break
                if not fallback_found:
                    logger.warning(
                        "Snapshot v%d not valid and no fallback available",
                        published_version,
                    )
                    return False

        if target_version is None or snapshot_dir is None:
            return False

        if target_version <= self._current_version:
            return False

        try:
            self._reload_callback(snapshot_dir, target_version)
            self._current_version = target_version
            logger.info("Reloaded index from snapshot v%d", target_version)
            return True
        except Exception:
            logger.exception("Failed to reload index from snapshot v%d", target_version)
            return False

    async def watch(self) -> None:
        """Watch for index updates using filesystem events."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        event_queue: asyncio.Queue[None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        class VersionHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and str(event.src_path).endswith("version.bin"):
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

        observer = Observer()
        observer.schedule(VersionHandler(), str(self._snapshot_base), recursive=False)
        observer.start()

        try:
            while True:
                await event_queue.get()
                try:
                    if self.check_for_update():
                        await asyncio.to_thread(self.reload_if_needed)
                except Exception:
                    logger.exception("Error checking for index updates")
        finally:
            observer.stop()
            observer.join(timeout=1.0)

    @property
    def current_version(self) -> int:
        return self._current_version
