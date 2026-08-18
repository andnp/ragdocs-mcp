"""Durable, source-scoped state for Google Drive synchronization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sqlite3
from typing import Protocol, runtime_checkable

STATE_SCHEMA_VERSION = 1
DEFAULT_BUSY_TIMEOUT_MS = 5_000


class GDriveStateError(RuntimeError):
    """The durable Google Drive state database cannot be used safely."""


class UnsupportedGDriveStateSchemaError(GDriveStateError):
    """The state database uses a schema version this repository cannot read."""


@dataclass(frozen=True, slots=True)
class GDriveScopeIdentity:
    """Identify one source, workspace, and synchronization scope."""

    source_kind: str
    workspace_id: str
    scope_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_kind", self.source_kind),
            ("workspace_id", self.workspace_id),
            ("scope_identity", self.scope_identity),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def as_parameters(self) -> tuple[str, str, str]:
        """Return the identity values in database-key order."""

        return self.source_kind, self.workspace_id, self.scope_identity


@dataclass(frozen=True, slots=True)
class GDriveCheckpoint:
    """Versioned cursors for one Drive scope."""

    identity: GDriveScopeIdentity
    inventory_start_token: str | None = None
    inventory_page_token: str | None = None
    inventory_batch: int = 0
    changes_token: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_optional_text(self.inventory_start_token, "inventory_start_token")
        _validate_optional_text(self.inventory_page_token, "inventory_page_token")
        _validate_optional_text(self.changes_token, "changes_token")
        _validate_non_negative_int(self.inventory_batch, "inventory_batch")

    def inventory_started(self, start_token: str) -> "GDriveCheckpoint":
        """Return a checkpoint for an inventory that is about to begin."""

        _validate_required_text(start_token, "start_token")
        return replace(
            self,
            inventory_start_token=start_token,
            inventory_page_token=None,
            inventory_batch=0,
            changes_token=None,
        )

    def inventory_batch_indexed(
        self,
        *,
        page_token: str | None,
        batch: int,
    ) -> "GDriveCheckpoint":
        """Return progress after one indexed inventory batch."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before a batch is indexed")
        if batch != self.inventory_batch + 1:
            raise ValueError("inventory batches must advance in order")
        _validate_optional_text(page_token, "page_token")
        return replace(self, inventory_page_token=page_token, inventory_batch=batch)

    def changes_indexed(self, changes_token: str) -> "GDriveCheckpoint":
        """Return progress after one indexed changes batch."""

        if self.inventory_start_token is None:
            raise ValueError("inventory must start before changes are indexed")
        _validate_required_text(changes_token, "changes_token")
        return replace(self, changes_token=changes_token)


@dataclass(frozen=True, slots=True)
class GDriveSyncStatus:
    """Versioned evaluated status for one Drive scope."""

    identity: GDriveScopeIdentity
    status: str
    last_success_at: float | None = None
    last_error: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.status, "status")
        if self.last_success_at is not None and not math.isfinite(self.last_success_at):
            raise ValueError("last_success_at must be finite or null")
        _validate_optional_text(self.last_error, "last_error")


@dataclass(frozen=True, slots=True)
class GDriveMembership:
    """Versioned visibility membership for one stable Drive record."""

    identity: GDriveScopeIdentity
    source_id: str
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.source_id, "source_id")


@dataclass(frozen=True, slots=True)
class GDriveScopeMembershipSnapshot:
    """The durable source IDs visible in one Drive scope."""

    identity: GDriveScopeIdentity
    source_ids: tuple[str, ...] = ()
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("scope membership source IDs must be unique")
        for source_id in self.source_ids:
            _validate_required_text(source_id, "source_id")


@dataclass(frozen=True, slots=True)
class GDriveBackfillCursor:
    """Versioned bounded backfill progress for one scope generation."""

    identity: GDriveScopeIdentity
    generation: str
    page_token: str | None = None
    batch: int = 0
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.generation, "generation")
        _validate_optional_text(self.page_token, "page_token")
        _validate_non_negative_int(self.batch, "batch")


@dataclass(frozen=True, slots=True)
class GDriveWatchState:
    """Versioned push-watch state for one Drive scope."""

    identity: GDriveScopeIdentity
    channel_id: str
    resource_id: str | None
    expiration: int
    address: str
    status: str = "active"
    last_error: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_record_version(self.schema_version)
        _validate_required_text(self.channel_id, "channel_id")
        _validate_optional_text(self.resource_id, "resource_id")
        _validate_non_negative_int(self.expiration, "expiration")
        _validate_required_text(self.address, "address")
        _validate_required_text(self.status, "status")
        _validate_optional_text(self.last_error, "last_error")


CheckpointUpdater = Callable[[GDriveCheckpoint | None], GDriveCheckpoint]


@runtime_checkable
class GDriveStatePort(Protocol):
    """Application capabilities required from durable Drive state."""

    def save_sync_status(self, status: GDriveSyncStatus) -> None: ...

    def add_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]: ...

    def load_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
    ) -> GDriveScopeMembershipSnapshot: ...

    def replace_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
        source_ids: Iterable[str],
    ) -> tuple[str, ...]: ...

    def remove_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]: ...

    def memberships_for_source(
        self,
        source_kind: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str, ...]: ...


class GDriveStateRepository:
    """Persist all Google Drive synchronization state in one SQLite file."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def load_checkpoint(self, identity: GDriveScopeIdentity) -> GDriveCheckpoint | None:
        """Load a checkpoint, treating missing or malformed rows as empty."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT inventory_start_token, inventory_page_token,
                       inventory_batch, changes_token, schema_version
                FROM checkpoints
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
        return _checkpoint_from_row(identity, row)

    def save_checkpoint(self, checkpoint: GDriveCheckpoint) -> None:
        """Atomically replace one checkpoint."""

        with self._transaction() as connection:
            self._write_checkpoint(connection, checkpoint)

    def update_checkpoint(
        self,
        identity: GDriveScopeIdentity,
        updater: CheckpointUpdater,
    ) -> GDriveCheckpoint:
        """Read, modify, and replace a checkpoint in one write transaction."""

        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT inventory_start_token, inventory_page_token,
                       inventory_batch, changes_token, schema_version
                FROM checkpoints
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
            current = _checkpoint_from_row(identity, row)
            checkpoint = updater(current)
            if checkpoint.identity != identity:
                raise ValueError("checkpoint identity cannot change during an update")
            self._write_checkpoint(connection, checkpoint)
        return checkpoint

    def begin_inventory(
        self,
        identity: GDriveScopeIdentity,
        start_token: str,
    ) -> GDriveCheckpoint:
        """Persist an inventory start token before provider enumeration."""

        def update(current: GDriveCheckpoint | None) -> GDriveCheckpoint:
            return (current or GDriveCheckpoint(identity)).inventory_started(start_token)

        return self.update_checkpoint(identity, update)

    def persist_inventory_batch(
        self,
        identity: GDriveScopeIdentity,
        *,
        page_token: str | None,
        batch: int,
    ) -> GDriveCheckpoint:
        """Persist inventory progress after the corresponding index mutation."""

        def update(current: GDriveCheckpoint | None) -> GDriveCheckpoint:
            if current is None:
                raise ValueError("inventory must begin before progress is persisted")
            return current.inventory_batch_indexed(page_token=page_token, batch=batch)

        return self.update_checkpoint(identity, update)

    def persist_changes(
        self,
        identity: GDriveScopeIdentity,
        changes_token: str,
    ) -> GDriveCheckpoint:
        """Persist a changes cursor after the corresponding index mutation."""

        def update(current: GDriveCheckpoint | None) -> GDriveCheckpoint:
            if current is None:
                raise ValueError("inventory must begin before changes are persisted")
            return current.changes_indexed(changes_token)

        return self.update_checkpoint(identity, update)

    def load_sync_status(self, identity: GDriveScopeIdentity) -> GDriveSyncStatus | None:
        """Load one scope status, treating malformed rows as empty."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT status, last_success_at, last_error, schema_version
                FROM sync_status
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
        return _sync_status_from_row(identity, row)

    def save_sync_status(self, status: GDriveSyncStatus) -> None:
        """Atomically replace one scope status."""

        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO sync_status (
                    source_kind, workspace_id, scope_identity, status,
                    last_success_at, last_error, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_kind, workspace_id, scope_identity) DO UPDATE SET
                    status = excluded.status,
                    last_success_at = excluded.last_success_at,
                    last_error = excluded.last_error,
                    schema_version = excluded.schema_version
                """,
                (
                    *status.identity.as_parameters(),
                    status.status,
                    status.last_success_at,
                    status.last_error,
                    status.schema_version,
                ),
            )

    def add_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]:
        """Add one set-like scope membership and return all scopes for the record."""

        membership = GDriveMembership(identity, source_id)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memberships (
                    source_kind, workspace_id, scope_identity, source_id, schema_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (*membership.identity.as_parameters(), membership.source_id, membership.schema_version),
            )
            return self._membership_scopes(connection, identity, source_id)

    def load_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
    ) -> GDriveScopeMembershipSnapshot:
        """Enumerate the durable source IDs currently visible in one scope."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT source_id, schema_version
                FROM memberships
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                ORDER BY source_id
                """,
                identity.as_parameters(),
            ).fetchall()
        source_ids: list[str] = []
        for row in rows:
            try:
                source_id = _required_text(row[0], "source_id")
                _validate_record_version(_integer(row[1], "schema_version"))
            except ValueError:
                continue
            source_ids.append(source_id)
        return GDriveScopeMembershipSnapshot(identity, tuple(source_ids))

    def replace_scope_memberships(
        self,
        identity: GDriveScopeIdentity,
        source_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Atomically replace one scope snapshot and return removed source IDs."""

        if isinstance(source_ids, str):
            raise TypeError("source_ids must be an iterable of source IDs")
        normalized = tuple(sorted({_required_text(source_id, "source_id") for source_id in source_ids}))
        GDriveScopeMembershipSnapshot(identity, normalized)
        with self._transaction() as connection:
            current = {
                source_id
                for source_id, schema_version in connection.execute(
                    """
                    SELECT source_id, schema_version
                    FROM memberships
                    WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                    """,
                    identity.as_parameters(),
                ).fetchall()
                if _safe_text(source_id) is not None
                and schema_version == STATE_SCHEMA_VERSION
            }
            connection.execute(
                """
                DELETE FROM memberships
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            )
            connection.executemany(
                """
                INSERT INTO memberships (
                    source_kind, workspace_id, scope_identity, source_id, schema_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (*identity.as_parameters(), source_id, STATE_SCHEMA_VERSION)
                    for source_id in normalized
                ],
            )
        return tuple(sorted(current.difference(normalized)))

    def remove_membership(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]:
        """Remove one scope membership and return the remaining scopes for the record."""

        _validate_required_text(source_id, "source_id")
        with self._transaction() as connection:
            connection.execute(
                """
                DELETE FROM memberships
                WHERE source_kind = ? AND workspace_id = ?
                  AND scope_identity = ? AND source_id = ?
                """,
                (*identity.as_parameters(), source_id),
            )
            return self._membership_scopes(connection, identity, source_id)

    def memberships_for(
        self,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]:
        """Return the membership for one source record within one scope."""

        _validate_required_text(source_id, "source_id")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM memberships
                WHERE source_kind = ? AND workspace_id = ?
                  AND scope_identity = ? AND source_id = ?
                """,
                (*identity.as_parameters(), source_id),
            ).fetchone()
        return (identity.scope_identity,) if row is not None else ()

    def memberships_for_source(
        self,
        source_kind: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str, ...]:
        """Return all scopes containing a source record, in stable order."""

        _validate_required_text(source_kind, "source_kind")
        _validate_required_text(workspace_id, "workspace_id")
        _validate_required_text(source_id, "source_id")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT scope_identity, schema_version
                FROM memberships
                WHERE source_kind = ? AND workspace_id = ? AND source_id = ?
                ORDER BY scope_identity
                """,
                (source_kind, workspace_id, source_id),
            ).fetchall()
        scopes: list[str] = []
        for row in rows:
            try:
                scope = _required_text(row[0], "scope_identity")
                _validate_record_version(_integer(row[1], "schema_version"))
            except ValueError:
                continue
            scopes.append(scope)
        return tuple(scopes)

    def load_backfill_cursor(
        self,
        identity: GDriveScopeIdentity,
        generation: str,
    ) -> GDriveBackfillCursor | None:
        """Load a cursor only when its generation matches."""

        _validate_required_text(generation, "generation")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT generation, page_token, batch, schema_version
                FROM backfill_cursors
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
        cursor = _backfill_from_row(identity, row)
        return cursor if cursor is not None and cursor.generation == generation else None

    def begin_backfill(
        self,
        identity: GDriveScopeIdentity,
        generation: str,
    ) -> GDriveBackfillCursor:
        """Start or reset a bounded backfill generation."""

        cursor = GDriveBackfillCursor(identity, generation)
        self.save_backfill_cursor(cursor)
        return cursor

    def save_backfill_cursor(self, cursor: GDriveBackfillCursor) -> None:
        """Atomically replace one scope's backfill cursor."""

        with self._transaction() as connection:
            self._write_backfill_cursor(connection, cursor)

    def persist_backfill_batch(
        self,
        identity: GDriveScopeIdentity,
        *,
        generation: str,
        page_token: str | None,
        batch: int,
    ) -> GDriveBackfillCursor:
        """Persist backfill progress after the corresponding index mutation."""

        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT generation, page_token, batch, schema_version
                FROM backfill_cursors
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
            current = _backfill_from_row(identity, row)
            if current is None or current.generation != generation:
                raise ValueError("backfill generation must begin before progress is persisted")
            if batch != current.batch + 1:
                raise ValueError("backfill batches must advance in order")
            cursor = replace(current, page_token=page_token, batch=batch)
            self._write_backfill_cursor(connection, cursor)
        return cursor

    def load_watch_state(self, identity: GDriveScopeIdentity) -> GDriveWatchState | None:
        """Load one scope's watch state, treating malformed rows as empty."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT channel_id, resource_id, expiration, address,
                       status, last_error, schema_version
                FROM watch_state
                WHERE source_kind = ? AND workspace_id = ? AND scope_identity = ?
                """,
                identity.as_parameters(),
            ).fetchone()
        return _watch_from_row(identity, row)

    def save_watch_state(self, watch: GDriveWatchState) -> None:
        """Atomically replace one scope's watch state."""

        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO watch_state (
                    source_kind, workspace_id, scope_identity, channel_id,
                    resource_id, expiration, address, status, last_error, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_kind, workspace_id, scope_identity) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    resource_id = excluded.resource_id,
                    expiration = excluded.expiration,
                    address = excluded.address,
                    status = excluded.status,
                    last_error = excluded.last_error,
                    schema_version = excluded.schema_version
                """,
                (
                    *watch.identity.as_parameters(),
                    watch.channel_id,
                    watch.resource_id,
                    watch.expiration,
                    watch.address,
                    watch.status,
                    watch.last_error,
                    watch.schema_version,
                ),
            )

    def _membership_scopes(
        self,
        connection: sqlite3.Connection,
        identity: GDriveScopeIdentity,
        source_id: str,
    ) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT scope_identity
            FROM memberships
            WHERE source_kind = ? AND workspace_id = ? AND source_id = ?
            ORDER BY scope_identity
            """,
            (identity.source_kind, identity.workspace_id, source_id),
        ).fetchall()
        return tuple(
            scope
            for row in rows
            if (scope := _safe_text(row[0])) is not None
        )

    def _write_checkpoint(
        self,
        connection: sqlite3.Connection,
        checkpoint: GDriveCheckpoint,
    ) -> None:
        connection.execute(
            """
            INSERT INTO checkpoints (
                source_kind, workspace_id, scope_identity, inventory_start_token,
                inventory_page_token, inventory_batch, changes_token, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_kind, workspace_id, scope_identity) DO UPDATE SET
                inventory_start_token = excluded.inventory_start_token,
                inventory_page_token = excluded.inventory_page_token,
                inventory_batch = excluded.inventory_batch,
                changes_token = excluded.changes_token,
                schema_version = excluded.schema_version
            """,
            (
                *checkpoint.identity.as_parameters(),
                checkpoint.inventory_start_token,
                checkpoint.inventory_page_token,
                checkpoint.inventory_batch,
                checkpoint.changes_token,
                checkpoint.schema_version,
            ),
        )

    def _write_backfill_cursor(
        self,
        connection: sqlite3.Connection,
        cursor: GDriveBackfillCursor,
    ) -> None:
        connection.execute(
            """
            INSERT INTO backfill_cursors (
                source_kind, workspace_id, scope_identity, generation,
                page_token, batch, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_kind, workspace_id, scope_identity) DO UPDATE SET
                generation = excluded.generation,
                page_token = excluded.page_token,
                batch = excluded.batch,
                schema_version = excluded.schema_version
            """,
            (
                *cursor.identity.as_parameters(),
                cursor.generation,
                cursor.page_token,
                cursor.batch,
                cursor.schema_version,
            ),
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
            )
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode is None or mode[0] != "wal":
                connection.close()
                raise GDriveStateError("Google Drive state database did not enable WAL")
            return connection
        except (OSError, sqlite3.DatabaseError) as error:
            raise GDriveStateError(
                f"cannot open Google Drive state database at {self.path}"
            ) from error

    def _initialize(self) -> None:
        with self._connection() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            version = _integer(row[0], "user_version") if row is not None else 0
            if version not in (0, STATE_SCHEMA_VERSION):
                raise UnsupportedGDriveStateSchemaError(
                    f"unsupported Google Drive state schema: {version}"
                )
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    self._create_schema(connection)
                    connection.execute(f"PRAGMA user_version = {STATE_SCHEMA_VERSION}")
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
            else:
                self._verify_schema(connection)

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                source_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                scope_identity TEXT NOT NULL,
                inventory_start_token TEXT,
                inventory_page_token TEXT,
                inventory_batch INTEGER NOT NULL,
                changes_token TEXT,
                schema_version INTEGER NOT NULL,
                PRIMARY KEY (source_kind, workspace_id, scope_identity)
            );
            CREATE TABLE IF NOT EXISTS sync_status (
                source_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                scope_identity TEXT NOT NULL,
                status TEXT NOT NULL,
                last_success_at REAL,
                last_error TEXT,
                schema_version INTEGER NOT NULL,
                PRIMARY KEY (source_kind, workspace_id, scope_identity)
            );
            CREATE TABLE IF NOT EXISTS memberships (
                source_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                scope_identity TEXT NOT NULL,
                source_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                PRIMARY KEY (source_kind, workspace_id, scope_identity, source_id)
            );
            CREATE TABLE IF NOT EXISTS backfill_cursors (
                source_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                scope_identity TEXT NOT NULL,
                generation TEXT NOT NULL,
                page_token TEXT,
                batch INTEGER NOT NULL,
                schema_version INTEGER NOT NULL,
                PRIMARY KEY (source_kind, workspace_id, scope_identity)
            );
            CREATE TABLE IF NOT EXISTS watch_state (
                source_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                scope_identity TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                resource_id TEXT,
                expiration INTEGER NOT NULL,
                address TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                schema_version INTEGER NOT NULL,
                PRIMARY KEY (source_kind, workspace_id, scope_identity)
            );
            """
        )

    def _verify_schema(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        tables = {_safe_text(row[0]) for row in rows}
        expected = {
            "checkpoints",
            "sync_status",
            "memberships",
            "backfill_cursors",
            "watch_state",
        }
        if not expected.issubset(tables):
            raise GDriveStateError("Google Drive state database has an incomplete schema")


def _checkpoint_from_row(
    identity: GDriveScopeIdentity,
    row: tuple[object, ...] | None,
) -> GDriveCheckpoint | None:
    if row is None:
        return None
    try:
        return GDriveCheckpoint(
            identity,
            _optional_text(row[0], "inventory_start_token"),
            _optional_text(row[1], "inventory_page_token"),
            _integer(row[2], "inventory_batch"),
            _optional_text(row[3], "changes_token"),
            _integer(row[4], "schema_version"),
        )
    except ValueError:
        return None


def _sync_status_from_row(
    identity: GDriveScopeIdentity,
    row: tuple[object, ...] | None,
) -> GDriveSyncStatus | None:
    if row is None:
        return None
    try:
        return GDriveSyncStatus(
            identity,
            _required_text(row[0], "status"),
            _optional_real(row[1], "last_success_at"),
            _optional_text(row[2], "last_error"),
            _integer(row[3], "schema_version"),
        )
    except (TypeError, ValueError):
        return None


def _backfill_from_row(
    identity: GDriveScopeIdentity,
    row: tuple[object, ...] | None,
) -> GDriveBackfillCursor | None:
    if row is None:
        return None
    try:
        return GDriveBackfillCursor(
            identity,
            _required_text(row[0], "generation"),
            _optional_text(row[1], "page_token"),
            _integer(row[2], "batch"),
            _integer(row[3], "schema_version"),
        )
    except ValueError:
        return None


def _watch_from_row(
    identity: GDriveScopeIdentity,
    row: tuple[object, ...] | None,
) -> GDriveWatchState | None:
    if row is None:
        return None
    try:
        return GDriveWatchState(
            identity,
            _required_text(row[0], "channel_id"),
            _optional_text(row[1], "resource_id"),
            _integer(row[2], "expiration"),
            _required_text(row[3], "address"),
            _required_text(row[4], "status"),
            _optional_text(row[5], "last_error"),
            _integer(row[6], "schema_version"),
        )
    except ValueError:
        return None


def _validate_record_version(version: int) -> None:
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported Google Drive state record schema: {version}")


def _validate_required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _validate_required_text(value, name)


def _validate_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _safe_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_real(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    return float(value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "STATE_SCHEMA_VERSION",
    "GDriveBackfillCursor",
    "GDriveCheckpoint",
    "GDriveMembership",
    "GDriveScopeMembershipSnapshot",
    "GDriveScopeIdentity",
    "GDriveStateError",
    "GDriveStatePort",
    "GDriveStateRepository",
    "GDriveSyncStatus",
    "GDriveWatchState",
    "UnsupportedGDriveStateSchemaError",
]
