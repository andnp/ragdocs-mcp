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
        if self._version_file.exists():
            try:
                data = self._version_file.read_bytes()
                if len(data) >= 4:
                    self._version = struct.unpack("<I", data[:4])[0]
            except (OSError, struct.error):
                logger.warning("Failed to load version file, starting from 0")
                self._version = 0

    def publish(self, persist_callback: Callable[[Path], None]) -> int:
        self._snapshot_base.mkdir(parents=True, exist_ok=True)

        new_version = self._version + 1
        snapshot_dir = self._snapshot_base / f"v{new_version}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        try:
            persist_callback(snapshot_dir)

            version_data = struct.pack("<I", new_version)
            temp_version_file = self._version_file.with_suffix(".tmp")
            temp_version_file.write_bytes(version_data)
            temp_version_file.replace(self._version_file)

            self._version = new_version
            logger.info("Published index snapshot v%d to %s", new_version, snapshot_dir)

            self._cleanup_old_snapshots(keep=2)

            return new_version
        except Exception:
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir, ignore_errors=True)
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

    def check_for_update(self) -> bool:
        published_version = self._read_published_version()
        if published_version is None:
            return False
        return published_version > self._current_version

    def reload_if_needed(self) -> bool:
        published_version = self._read_published_version()
        if published_version is None:
            return False

        if published_version <= self._current_version:
            return False

        snapshot_dir = self._snapshot_base / f"v{published_version}"
        if not snapshot_dir.exists():
            logger.warning("Snapshot directory for v%d not found", published_version)
            return False

        try:
            self._reload_callback(snapshot_dir, published_version)
            self._current_version = published_version
            logger.info("Reloaded index from snapshot v%d", published_version)
            return True
        except Exception:
            logger.exception("Failed to reload index from snapshot v%d", published_version)
            return False

    async def watch(self, poll_interval: float = 0.1) -> None:
        while True:
            try:
                if self.check_for_update():
                    await asyncio.to_thread(self.reload_if_needed)
            except Exception:
                logger.exception("Error checking for index updates")

            await asyncio.sleep(poll_interval)

    @property
    def current_version(self) -> int:
        return self._current_version
