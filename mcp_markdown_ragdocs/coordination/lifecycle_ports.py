from __future__ import annotations

import asyncio
import inspect
from typing import Protocol

from mcp_markdown_ragdocs.daemon import (
    DaemonMetadata,
    RuntimePaths,
    remove_daemon_metadata,
    write_daemon_metadata,
)


class LifecycleMetadataPort(Protocol):
    def write(self, metadata: DaemonMetadata) -> None: ...

    def remove(self) -> None: ...


class WorkerSupervisionPort(Protocol):
    async def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_healthy(self) -> bool: ...

    async def restart(self) -> None: ...


class DaemonMetadataAdapter:
    def __init__(self, runtime_paths: RuntimePaths) -> None:
        self._metadata_path = runtime_paths.metadata_path

    def write(self, metadata: DaemonMetadata) -> None:
        write_daemon_metadata(self._metadata_path, metadata)

    def remove(self) -> None:
        remove_daemon_metadata(self._metadata_path)


class WorkerSupervisionAdapter:
    def __init__(self, worker: object) -> None:
        self._worker = worker

    async def start(self) -> None:
        await asyncio.to_thread(self._worker.start)  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._worker.stop()  # type: ignore[attr-defined]

    def is_healthy(self) -> bool:
        return bool(self._worker.is_healthy())  # type: ignore[attr-defined]

    async def restart(self) -> None:
        result = await asyncio.to_thread(self._worker.restart)  # type: ignore[attr-defined]
        if inspect.isawaitable(result):
            await result
