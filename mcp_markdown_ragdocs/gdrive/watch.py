"""Durable Google Drive push-channel renewal with polling fallback."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from searchkernel.api import atomic_write_json

from mcp_markdown_ragdocs.adapters.sources.gdrive import GoogleDriveContentSource
from mcp_markdown_ragdocs.gdrive.models import DriveScope, DriveWatchChannel
from mcp_markdown_ragdocs.gdrive.retry import DriveRetryWorkStore

WATCH_SCHEMA_VERSION = 1
WATCH_STATE_FILENAME = "gdrive-watch-state.json"
WatchMode = Literal["push", "poll"]


@dataclass(frozen=True, slots=True)
class GDriveWatchState:
    mode: WatchMode
    channel: DriveWatchChannel | None = None
    last_renewed_at: float | None = None
    last_error: str | None = None

    def to_payload(self) -> dict[str, object]:
        channel = None
        if self.channel is not None:
            channel = {
                "channel_id": self.channel.channel_id,
                "resource_id": self.channel.resource_id,
                "expiration": self.channel.expiration,
                "address": self.channel.address,
                "status": self.channel.status,
                "last_error": self.channel.last_error,
            }
        return {
            "mode": self.mode,
            "channel": channel,
            "last_renewed_at": self.last_renewed_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "GDriveWatchState":
        if not isinstance(payload, dict) or payload.get("mode") not in {"push", "poll"}:
            raise ValueError("invalid Google Drive watch state")
        raw_channel = payload.get("channel")
        channel = None
        if raw_channel is not None:
            if not isinstance(raw_channel, dict):
                raise ValueError("invalid Google Drive watch channel")
            channel = DriveWatchChannel(
                channel_id=str(raw_channel.get("channel_id") or ""),
                resource_id=(
                    str(raw_channel["resource_id"])
                    if raw_channel.get("resource_id")
                    else None
                ),
                expiration=int(raw_channel.get("expiration") or 0),
                address=str(raw_channel.get("address") or ""),
                status=str(raw_channel.get("status") or "active"),
                last_error=(
                    str(raw_channel["last_error"])
                    if raw_channel.get("last_error")
                    else None
                ),
            )
        renewed_at = payload.get("last_renewed_at")
        if renewed_at is not None and not isinstance(renewed_at, (float, int)):
            raise ValueError("invalid watch renewal timestamp")
        error = payload.get("last_error")
        return cls(
            payload["mode"],
            channel,
            float(renewed_at) if renewed_at is not None else None,
            str(error) if error else None,
        )


class GDriveWatchStateStore:
    """Atomically persist channel state independently for each Drive scope."""

    def __init__(self, index_root: Path) -> None:
        self.path = Path(index_root) / WATCH_STATE_FILENAME

    def load(self, namespace: str) -> GDriveWatchState | None:
        return self._read().get(namespace)

    def save(self, namespace: str, state: GDriveWatchState) -> None:
        states = {key: value.to_payload() for key, value in self._read().items()}
        states[namespace] = state.to_payload()
        atomic_write_json(
            self.path,
            {"schema_version": WATCH_SCHEMA_VERSION, "states": states},
        )

    def _read(self) -> dict[str, GDriveWatchState]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != WATCH_SCHEMA_VERSION:
            return {}
        raw_states = payload.get("states")
        if not isinstance(raw_states, dict):
            return {}
        states: dict[str, GDriveWatchState] = {}
        for namespace, raw_state in raw_states.items():
            try:
                states[str(namespace)] = GDriveWatchState.from_payload(raw_state)
            except (TypeError, ValueError):
                continue
        return states


@dataclass(frozen=True, slots=True)
class GDriveWatchResult:
    namespace: str
    mode: WatchMode
    channel: DriveWatchChannel | None
    renewed: bool
    fallback_reason: str | None = None
    poll_interval_seconds: int | None = None


class GoogleDriveWatch:
    """Maintain one Drive changes channel per scope and degrade to polling."""

    def __init__(
        self,
        source: GoogleDriveContentSource,
        state_store: GDriveWatchStateStore,
        *,
        scope_generation: str,
        address: str,
        renewal_seconds: int = 3600,
        push_enabled: bool = True,
        poll_interval_seconds: int = 3600,
        clock: Callable[[], float] = time.time,
        channel_id_factory: Callable[[str], str] | None = None,
        retry_work_store: DriveRetryWorkStore | None = None,
    ) -> None:
        if not scope_generation:
            raise ValueError("scope_generation is required")
        if renewal_seconds < 1 or poll_interval_seconds < 1:
            raise ValueError("watch intervals must be positive")
        self.source = source
        self.state_store = state_store
        self.scope_generation = scope_generation
        self.address = address
        self.renewal_seconds = renewal_seconds
        self.push_enabled = push_enabled
        self.poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._channel_id_factory = channel_id_factory or (lambda _namespace: uuid4().hex)
        self._retry_work_store = retry_work_store

    async def ensure(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        address: str | None = None,
    ) -> GDriveWatchResult:
        namespace = self._namespace(scope)
        current = self.state_store.load(namespace)
        if not self.push_enabled or not (address or self.address):
            return self._polling(namespace, "push is disabled or has no callback address")
        if current is not None and self._is_current(current, address or self.address):
            return GDriveWatchResult(namespace, "push", current.channel, False)

        channel_address = address or self.address
        channel_id = self._channel_id_factory(namespace)
        try:
            channel = await self.source.client.watch_changes(
                scope,
                page_token,
                channel_id=channel_id,
                address=channel_address,
            )
        except Exception as error:
            if self._retry_work_store is not None:
                self._retry_work_store.schedule_failure(
                    scope_identity=self.source.scope_identity(scope),
                    source_id="channel",
                    operation="watch",
                    payload={"address": channel_address},
                    error=error,
                    now=self._clock(),
                )
            return self._polling(namespace, str(error) or type(error).__name__)

        stop_error = None
        if current is not None and current.channel is not None:
            try:
                await self.source.client.stop_channel(
                    current.channel.channel_id,
                    current.channel.resource_id,
                )
            except Exception as error:
                stop_error = str(error) or type(error).__name__
        if stop_error:
            channel = DriveWatchChannel(
                channel.channel_id,
                channel.resource_id,
                channel.expiration,
                channel.address,
                channel.status,
                stop_error,
            )
        self.state_store.save(
            namespace,
            GDriveWatchState("push", channel, self._clock(), stop_error),
        )
        return GDriveWatchResult(namespace, "push", channel, True, stop_error)

    async def renew(
        self,
        scope: DriveScope,
        page_token: str,
        *,
        address: str | None = None,
    ) -> GDriveWatchResult:
        return await self.ensure(scope, page_token, address=address)

    def _is_current(self, state: GDriveWatchState, address: str) -> bool:
        if state.mode != "push" or state.channel is None:
            return False
        expiration = state.channel.expiration / 1000
        return state.channel.address == address and expiration > self._clock() + self.renewal_seconds

    def _polling(self, namespace: str, reason: str) -> GDriveWatchResult:
        self.state_store.save(namespace, GDriveWatchState("poll", last_error=reason))
        return GDriveWatchResult(
            namespace,
            "poll",
            None,
            False,
            reason,
            self.poll_interval_seconds,
        )

    def _namespace(self, scope: DriveScope) -> str:
        return f"{self.scope_generation}:{self.source.scope_identity(scope)}"


__all__ = [
    "GDriveWatchResult",
    "GDriveWatchState",
    "GDriveWatchStateStore",
    "GoogleDriveWatch",
    "WATCH_SCHEMA_VERSION",
    "WATCH_STATE_FILENAME",
]
