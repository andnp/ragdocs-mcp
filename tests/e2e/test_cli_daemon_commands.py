from __future__ import annotations

from click.testing import CliRunner

from src.cli import cli
from src.daemon.management import DaemonInspection
from src.daemon.metadata import DaemonMetadata


def test_daemon_status_reports_not_running(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda: DaemonInspection(metadata=None, running=False, stale=False),
    )

    result = runner.invoke(cli, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Daemon status: not running" in result.output


def test_daemon_status_reports_running(monkeypatch):
    runner = CliRunner()
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1_763_700_000.0,
        status="ready",
        socket_path="/tmp/ragdocs.sock",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda: DaemonInspection(metadata=metadata, running=True, stale=False),
    )

    result = runner.invoke(cli, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Daemon status: running" in result.output
    assert "PID: 4321" in result.output
    assert "Lifecycle: ready" in result.output


def test_daemon_start_invokes_management_helper(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    def _fake_start_daemon(*, project_override, timeout_seconds, paths=None):
        observed["project_override"] = project_override
        observed["timeout_seconds"] = timeout_seconds
        return DaemonMetadata(pid=99, started_at=1.0, status="ready")

    monkeypatch.setattr("src.cli.start_daemon", _fake_start_daemon)

    result = runner.invoke(
        cli,
        ["daemon", "start", "--project", "docs", "--timeout", "3.5"],
    )

    assert result.exit_code == 0
    assert observed == {"project_override": "docs", "timeout_seconds": 3.5}
    assert "Daemon running (pid=99, status=ready)" in result.output


def test_daemon_stop_reports_stopped(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli.stop_daemon",
        lambda *, timeout_seconds: DaemonMetadata(pid=77, started_at=1.0, status="ready"),
    )

    result = runner.invoke(cli, ["daemon", "stop"])

    assert result.exit_code == 0
    assert "Daemon stopped (pid=77)" in result.output


def test_daemon_restart_invokes_management_helper(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    def _fake_restart_daemon(*, project_override, start_timeout_seconds, paths=None, stop_timeout_seconds=5.0):
        observed["project_override"] = project_override
        observed["start_timeout_seconds"] = start_timeout_seconds
        observed["stop_timeout_seconds"] = stop_timeout_seconds
        return DaemonMetadata(pid=123, started_at=1.0, status="ready")

    monkeypatch.setattr("src.cli.restart_daemon", _fake_restart_daemon)

    result = runner.invoke(
        cli,
        ["daemon", "restart", "--project", "notes", "--timeout", "4.0"],
    )

    assert result.exit_code == 0
    assert observed == {
        "project_override": "notes",
        "start_timeout_seconds": 4.0,
        "stop_timeout_seconds": 5.0,
    }
    assert "Daemon restarted (pid=123, status=ready)" in result.output