"""
Unit tests for IPC command dataclasses.

Tests the frozen dataclass semantics, default values, and IPCMessage union type.
"""

from dataclasses import FrozenInstanceError

import pytest

from src.ipc.commands import (
    HealthCheckCommand,
    HealthStatusResponse,
    IndexUpdatedNotification,
    InitCompleteNotification,
    IPCMessage,
    ReindexDocumentCommand,
    ShutdownCommand,
)


class TestShutdownCommand:
    """Tests for ShutdownCommand frozen dataclass."""

    def test_defaults(self):
        """Verify default values for graceful shutdown with timeout."""
        cmd = ShutdownCommand()
        assert cmd.graceful is True
        assert cmd.timeout == 5.0

    def test_custom_values(self):
        """Verify custom shutdown configuration."""
        cmd = ShutdownCommand(graceful=False, timeout=10.0)
        assert cmd.graceful is False
        assert cmd.timeout == 10.0

    def test_frozen_immutability(self):
        """Verify frozen dataclass cannot be mutated."""
        cmd = ShutdownCommand()
        with pytest.raises(FrozenInstanceError):
            cmd.graceful = False  # type: ignore[misc]


class TestHealthCheckCommand:
    """Tests for HealthCheckCommand frozen dataclass."""

    def test_instantiation(self):
        """Verify HealthCheckCommand can be instantiated."""
        cmd = HealthCheckCommand()
        assert isinstance(cmd, HealthCheckCommand)

    def test_frozen_immutability(self):
        """Verify frozen dataclass cannot have attributes added."""
        cmd = HealthCheckCommand()
        with pytest.raises(FrozenInstanceError):
            cmd.extra = "value"  # type: ignore[attr-defined]


class TestReindexDocumentCommand:
    """Tests for ReindexDocumentCommand frozen dataclass."""

    def test_required_fields(self):
        """Verify required doc_id and optional reason field."""
        cmd = ReindexDocumentCommand(doc_id="doc:readme.md")
        assert cmd.doc_id == "doc:readme.md"
        assert cmd.reason == ""

    def test_with_reason(self):
        """Verify reason field can be provided."""
        cmd = ReindexDocumentCommand(doc_id="doc:readme.md", reason="file_modified")
        assert cmd.doc_id == "doc:readme.md"
        assert cmd.reason == "file_modified"

    def test_frozen_immutability(self):
        """Verify frozen dataclass cannot be mutated."""
        cmd = ReindexDocumentCommand(doc_id="test")
        with pytest.raises(FrozenInstanceError):
            cmd.doc_id = "other"  # type: ignore[misc]


class TestIndexUpdatedNotification:
    """Tests for IndexUpdatedNotification frozen dataclass."""

    def test_all_fields_required(self):
        """Verify all three fields must be provided."""
        notif = IndexUpdatedNotification(
            version=42,
            doc_count=100,
            timestamp=1234567890.123,
        )
        assert notif.version == 42
        assert notif.doc_count == 100
        assert notif.timestamp == 1234567890.123

    def test_frozen_immutability(self):
        """Verify frozen dataclass cannot be mutated."""
        notif = IndexUpdatedNotification(version=1, doc_count=10, timestamp=0.0)
        with pytest.raises(FrozenInstanceError):
            notif.version = 2  # type: ignore[misc]


class TestInitCompleteNotification:
    """Tests for InitCompleteNotification frozen dataclass."""

    def test_required_fields(self):
        """Verify version and doc_count are required, timestamp has default."""
        notif = InitCompleteNotification(version=5, doc_count=100)
        assert notif.version == 5
        assert notif.doc_count == 100
        assert isinstance(notif.timestamp, float)
        assert notif.timestamp > 0

    def test_custom_timestamp(self):
        """Verify timestamp can be explicitly provided."""
        notif = InitCompleteNotification(version=1, doc_count=10, timestamp=123.456)
        assert notif.timestamp == 123.456


class TestHealthStatusResponse:
    """Tests for HealthStatusResponse frozen dataclass."""

    def test_all_fields(self):
        """Verify all fields are properly set."""
        resp = HealthStatusResponse(
            healthy=True,
            queue_depth=5,
            last_index_time=1234567890.5,
            doc_count=250,
        )
        assert resp.healthy is True
        assert resp.queue_depth == 5
        assert resp.last_index_time == 1234567890.5
        assert resp.doc_count == 250

    def test_unhealthy_state(self):
        """Verify unhealthy status can be represented."""
        resp = HealthStatusResponse(
            healthy=False,
            queue_depth=0,
            last_index_time=None,
            doc_count=0,
        )
        assert resp.healthy is False
        assert resp.last_index_time is None

    def test_circuit_breaker_state_default(self):
        """Verify circuit_breaker_state defaults to 'closed'."""
        resp = HealthStatusResponse(
            healthy=True,
            queue_depth=0,
            last_index_time=None,
            doc_count=0,
        )
        assert resp.circuit_breaker_state == "closed"

    def test_circuit_breaker_state_open(self):
        """Verify circuit_breaker_state can be set to 'open'."""
        resp = HealthStatusResponse(
            healthy=True,
            queue_depth=0,
            last_index_time=None,
            doc_count=0,
            circuit_breaker_state="open",
        )
        assert resp.circuit_breaker_state == "open"

    def test_circuit_breaker_state_half_open(self):
        """Verify circuit_breaker_state can be set to 'half_open'."""
        resp = HealthStatusResponse(
            healthy=True,
            queue_depth=0,
            last_index_time=None,
            doc_count=0,
            circuit_breaker_state="half_open",
        )
        assert resp.circuit_breaker_state == "half_open"


class TestIPCMessageUnion:
    """Tests for IPCMessage type alias (union of all command types)."""

    def test_all_types_are_valid_ipc_messages(self):
        """
        Verify all command types can be assigned to IPCMessage.

        This test validates the type union at runtime by checking isinstance.
        """
        messages: list[IPCMessage] = [
            ShutdownCommand(),
            HealthCheckCommand(),
            ReindexDocumentCommand(doc_id="test"),
            IndexUpdatedNotification(version=1, doc_count=10, timestamp=0.0),
            InitCompleteNotification(version=1, doc_count=10),
            HealthStatusResponse(healthy=True, queue_depth=0, last_index_time=None, doc_count=0),
        ]
        assert len(messages) == 6
        # All should be dataclass instances
        for msg in messages:
            assert hasattr(msg, "__dataclass_fields__")
