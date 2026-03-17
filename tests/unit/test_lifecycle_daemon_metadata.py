from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.daemon import read_daemon_metadata
from src.lifecycle import LifecycleCoordinator, LifecycleState


@dataclass
class _FakeGitIndexingConfig:
    enabled: bool = False
    watch_enabled: bool = False


@dataclass
class _FakeConfig:
    git_indexing: _FakeGitIndexingConfig = field(
        default_factory=_FakeGitIndexingConfig
    )


@dataclass
class _FakeContext:
    config: _FakeConfig = field(default_factory=_FakeConfig)
    started: bool = False
    stopped: bool = False

    async def start(self, background_index: bool = False) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def ensure_ready(self, timeout: float = 60.0) -> None:
        return None


@pytest.mark.asyncio
async def test_lifecycle_writes_and_removes_initializing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.daemon.paths._state_home", lambda: tmp_path)

    coordinator = LifecycleCoordinator()
    ctx = _FakeContext()

    await coordinator.start(ctx, background_index=True)

    metadata_path = tmp_path / "mcp-markdown-ragdocs" / "daemon" / "daemon.json"
    metadata = read_daemon_metadata(metadata_path)

    assert metadata is not None
    assert metadata.status == LifecycleState.INITIALIZING.value

    await coordinator.shutdown()

    assert not metadata_path.exists()


@pytest.mark.asyncio
async def test_lifecycle_updates_metadata_when_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.daemon.paths._state_home", lambda: tmp_path)

    coordinator = LifecycleCoordinator()
    ctx = _FakeContext()

    await coordinator.start(ctx, background_index=False)

    metadata_path = tmp_path / "mcp-markdown-ragdocs" / "daemon" / "daemon.json"
    metadata = read_daemon_metadata(metadata_path)

    assert metadata is not None
    assert metadata.status == LifecycleState.READY.value

    await coordinator.shutdown()
    assert ctx.stopped is True