from __future__ import annotations

from click.testing import CliRunner
from pathlib import Path
import pytest

from src.cli import cli
from src.lifecycle import LifecycleState
from src.daemon import RuntimePaths
from src.daemon.management import DaemonInspection
from src.daemon.metadata import DaemonMetadata


def test_daemon_status_reports_not_running(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(metadata=None, running=False, stale=False),
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
        lambda paths=None: DaemonInspection(metadata=metadata, running=True, stale=False),
    )

    result = runner.invoke(cli, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Daemon status: running" in result.output
    assert "PID: 4321" in result.output
    assert "Lifecycle: ready" in result.output
    assert "Scope: global" in result.output


def test_daemon_status_reports_starting_when_not_ready(monkeypatch):
    runner = CliRunner()
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1_763_700_000.0,
        status="starting",
        socket_path="/tmp/ragdocs.sock",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=metadata,
            running=True,
            stale=False,
            responsive=True,
            ready=False,
        ),
    )

    result = runner.invoke(cli, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Daemon status: starting" in result.output


def test_daemon_status_json_includes_runtime_paths(monkeypatch, tmp_path):
    runner = CliRunner()
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1_763_700_000.0,
        status="ready",
        socket_path="/tmp/ragdocs.sock",
        index_db_path="/tmp/index.db",
        queue_db_path="/tmp/queue.db",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda: DaemonInspection(metadata=metadata, running=True, stale=False),
    )
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(
            lambda cls: RuntimePaths(
                root=tmp_path,
                index_db_path=tmp_path / "index.db",
                queue_db_path=tmp_path / "queue.db",
                metadata_path=tmp_path / "daemon.json",
                lock_path=tmp_path / "daemon.lock",
                socket_path=tmp_path / "daemon.sock",
            )
        ),
    )

    result = runner.invoke(cli, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    assert '"status": "running"' in result.output
    assert '"daemon_scope": "global"' in result.output
    assert '"metadata_path":' in result.output
    assert '"index_db_path": "/tmp/index.db"' in result.output


def test_daemon_status_json_includes_overview_when_available(monkeypatch, tmp_path):
    runner = CliRunner()
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1_763_700_000.0,
        status="ready_primary",
        socket_path="/tmp/ragdocs.sock",
        index_db_path="/tmp/index.db",
        queue_db_path="/tmp/queue.db",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=metadata,
            running=True,
            stale=False,
            responsive=True,
            ready=True,
        ),
    )
    monkeypatch.setattr(
        "src.cli.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: {
            "status": "ok",
            "indexed_documents": 9,
            "indexed_chunks": 21,
            "git_commits": 12,
            "git_repositories": 2,
            "configured_root_count": 2,
            "documents_roots": ["/docs/a", "/docs/b"],
            "project_context_mode": "request_only",
            "pending_count": 3,
            "scheduled_count": 1,
            "running_count": 0,
            "failed_count": 1,
            "worker_running": True,
            "worker_health": "healthy",
        },
    )
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(
            lambda cls: RuntimePaths(
                root=tmp_path,
                index_db_path=tmp_path / "index.db",
                queue_db_path=tmp_path / "queue.db",
                metadata_path=tmp_path / "daemon.json",
                lock_path=tmp_path / "daemon.lock",
                socket_path=tmp_path / "daemon.sock",
            )
        ),
    )

    result = runner.invoke(cli, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    assert '"indexed_documents": 9' in result.output
    assert '"configured_root_count": 2' in result.output
    assert '"project_context_mode": "request_only"' in result.output
    assert '"pending_count": 3' in result.output
    assert '"worker_health": "healthy"' in result.output


def test_daemon_status_json_fetches_overview_even_if_probe_not_responsive(
    monkeypatch,
    tmp_path,
):
    runner = CliRunner()
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1_763_700_000.0,
        status="ready_primary",
        socket_path="/tmp/ragdocs.sock",
        index_db_path="/tmp/index.db",
        queue_db_path="/tmp/queue.db",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=metadata,
            running=True,
            stale=False,
            responsive=False,
            ready=False,
        ),
    )
    monkeypatch.setattr(
        "src.cli.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: {
            "status": "ok",
            "worker_health": "healthy",
            "worker_pid": 9999,
            "worker_running": True,
        },
    )
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(
            lambda cls: RuntimePaths(
                root=tmp_path,
                index_db_path=tmp_path / "index.db",
                queue_db_path=tmp_path / "queue.db",
                metadata_path=tmp_path / "daemon.json",
                lock_path=tmp_path / "daemon.lock",
                socket_path=tmp_path / "daemon.sock",
            )
        ),
    )

    result = runner.invoke(cli, ["daemon", "status", "--json"])

    assert result.exit_code == 0
    assert '"worker_health": "healthy"' in result.output
    assert '"worker_pid": 9999' in result.output


def test_daemon_start_invokes_management_helper(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    def _fake_start_daemon(*, timeout_seconds, paths=None):
        observed["timeout_seconds"] = timeout_seconds
        return DaemonMetadata(pid=99, started_at=1.0, status="ready")

    monkeypatch.setattr(
        "src.cli._ignore_daemon_startup_project_option",
        lambda project: observed.setdefault("ignored_project", project),
    )
    monkeypatch.setattr("src.cli.start_daemon", _fake_start_daemon)

    result = runner.invoke(
        cli,
        ["daemon", "start", "--project", "docs", "--timeout", "3.5"],
    )

    assert result.exit_code == 0
    assert observed == {"ignored_project": "docs", "timeout_seconds": 3.5}
    assert "Daemon running (pid=99, status=ready)" in result.output


def test_daemon_run_accepts_but_ignores_project_option(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    async def _fake_run_daemon_forever():
        observed["ran"] = True

    monkeypatch.setattr(
        "src.cli._ignore_daemon_startup_project_option",
        lambda project: observed.setdefault("ignored_project", project),
    )
    monkeypatch.setattr("src.cli._run_daemon_forever", _fake_run_daemon_forever)

    result = runner.invoke(cli, ["daemon", "run", "--project", "docs"])

    assert result.exit_code == 0
    assert observed == {"ignored_project": "docs", "ran": True}


def test_daemon_internal_run_accepts_but_ignores_project_option(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    async def _fake_run_daemon_forever():
        observed["ran"] = True

    monkeypatch.setattr(
        "src.cli._ignore_daemon_startup_project_option",
        lambda project: observed.setdefault("ignored_project", project),
    )
    monkeypatch.setattr("src.cli._run_daemon_forever", _fake_run_daemon_forever)

    result = runner.invoke(cli, ["daemon-internal-run", "--project", "docs"])

    assert result.exit_code == 0
    assert observed == {"ignored_project": "docs", "ran": True}


@pytest.mark.asyncio
async def test_run_daemon_forever_releases_boot_lock_after_startup(
    monkeypatch,
    tmp_path,
):
    class _FakeLock:
        def __init__(self):
            self.release_calls = 0

        def release(self):
            self.release_calls += 1

    class _FakeCoordinator:
        def __init__(self):
            self.state = LifecycleState.UNINITIALIZED

        def install_signal_handlers(self, loop):
            return None

        async def start(self, ctx, *, background_index, db_manager, huey_worker):
            self.state = LifecycleState.TERMINATED

        async def shutdown(self):
            return None

    class _FakeHealthServer:
        def __init__(self, **kwargs):
            self.started = False
            self.stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    class _FakeWorker:
        is_running = False

    fake_lock = _FakeLock()
    fake_coordinator = _FakeCoordinator()
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )

    class _FakeContext:
        db_manager = None

    monkeypatch.setattr("src.cli.acquire_boot_lock", lambda timeout_seconds=5.0: fake_lock)
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(lambda cls: runtime_paths),
    )
    monkeypatch.setattr("src.cli.DaemonHealthServer", _FakeHealthServer)
    monkeypatch.setattr("src.cli.LifecycleCoordinator", lambda: fake_coordinator)
    monkeypatch.setattr(
        "src.cli._create_daemon_runtime",
        lambda paths: (_FakeContext(), _FakeWorker()),
    )

    from src.cli import _run_daemon_forever

    await _run_daemon_forever()

    assert fake_lock.release_calls == 1


def test_request_daemon_json_does_not_wait_for_ready_before_query(monkeypatch, tmp_path):
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )
    initializing = DaemonMetadata(
        pid=321,
        started_at=1.0,
        status="initializing",
        socket_path=str(runtime_paths.socket_path),
    )
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(lambda cls: runtime_paths),
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=initializing,
            running=True,
            stale=False,
            responsive=True,
            ready=False,
        ),
    )

    def _fail_wait_for_daemon_ready(*, timeout_seconds, paths=None):
        raise AssertionError("wait_for_daemon_ready should not be called")

    monkeypatch.setattr("src.cli.wait_for_daemon_ready", _fail_wait_for_daemon_ready, raising=False)

    observed: dict[str, object] = {}

    def _fake_request_daemon_socket(socket_path, path, payload, timeout_seconds):
        observed["socket_path"] = socket_path
        observed["path"] = path
        observed["payload"] = payload
        observed["timeout_seconds"] = timeout_seconds
        return {"status": "ok", "value": 1}

    monkeypatch.setattr("src.cli.request_daemon_socket", _fake_request_daemon_socket)

    from src.cli import _request_daemon_json

    response = _request_daemon_json(
        "/api/search/query",
        {"value": 1},
        project_override=None,
        auto_start=True,
    )

    assert response == {"status": "ok", "value": 1}
    assert observed["socket_path"] == runtime_paths.socket_path
    assert observed["path"] == "/api/search/query"


def test_request_daemon_json_does_not_wait_for_ready_for_git_history(
    monkeypatch,
    tmp_path,
):
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )
    initializing = DaemonMetadata(
        pid=321,
        started_at=1.0,
        status="initializing",
        socket_path=str(runtime_paths.socket_path),
    )

    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(lambda cls: runtime_paths),
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=initializing,
            running=True,
            stale=False,
            responsive=True,
            ready=False,
        ),
    )

    def _fail_wait_for_daemon_ready(*, timeout_seconds, paths=None):
        raise AssertionError("wait_for_daemon_ready should not be called")

    monkeypatch.setattr("src.cli.wait_for_daemon_ready", _fail_wait_for_daemon_ready, raising=False)

    observed: dict[str, object] = {}

    def _fake_request_daemon_socket(socket_path, path, payload, timeout_seconds):
        observed["socket_path"] = socket_path
        observed["path"] = path
        observed["payload"] = payload
        observed["timeout_seconds"] = timeout_seconds
        return {"status": "ok", "value": 1}

    monkeypatch.setattr("src.cli.request_daemon_socket", _fake_request_daemon_socket)

    from src.cli import _request_daemon_json

    response = _request_daemon_json(
        "/api/search/git-history",
        {"query": "daemon"},
        project_override=None,
        auto_start=True,
    )

    assert response == {"status": "ok", "value": 1}
    assert observed["socket_path"] == runtime_paths.socket_path
    assert observed["path"] == "/api/search/git-history"


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

    def _fake_restart_daemon(*, start_timeout_seconds, paths=None, stop_timeout_seconds=5.0):
        observed["start_timeout_seconds"] = start_timeout_seconds
        observed["stop_timeout_seconds"] = stop_timeout_seconds
        return DaemonMetadata(pid=123, started_at=1.0, status="ready")

    monkeypatch.setattr(
        "src.cli._ignore_daemon_startup_project_option",
        lambda project: observed.setdefault("ignored_project", project),
    )
    monkeypatch.setattr("src.cli.restart_daemon", _fake_restart_daemon)

    result = runner.invoke(
        cli,
        ["daemon", "restart", "--project", "notes", "--timeout", "4.0"],
    )

    assert result.exit_code == 0
    assert observed == {
        "ignored_project": "notes",
        "start_timeout_seconds": 4.0,
        "stop_timeout_seconds": 5.0,
    }
    assert "Daemon restarted (pid=123, status=ready)" in result.output


def test_create_daemon_runtime_builds_global_runtime_without_project_context(
    monkeypatch,
    tmp_path,
):
    observed: dict[str, object] = {}

    class _FakeIndexManager:
        pass

    class _FakeContext:
        def __init__(self):
            self.index_manager = _FakeIndexManager()
            self.commit_indexer = object()
            self.config = type(
                "_FakeConfig",
                (),
                {
                    "indexing": type(
                        "_FakeIndexing",
                        (),
                        {"task_backpressure_limit": 100},
                    )()
                },
            )()

    class _FakeWorker:
        def __init__(self, *, runtime_paths):
            observed["worker_runtime_paths"] = runtime_paths

    fake_ctx = _FakeContext()
    fake_huey = object()
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )

    def _fake_create(**kwargs):
        observed["create_kwargs"] = kwargs
        return fake_ctx

    def _fake_get_huey(path):
        observed["queue_path"] = path
        return fake_huey

    def _fake_register_tasks(
        huey,
        index_manager,
        commit_indexer=None,
        task_backpressure_limit=None,
    ):
        observed["register"] = (
            huey,
            index_manager,
            commit_indexer,
            task_backpressure_limit,
        )

    monkeypatch.setattr("src.cli.ApplicationContext.create", _fake_create)
    monkeypatch.setattr("src.cli.get_huey", _fake_get_huey)
    monkeypatch.setattr("src.cli.register_tasks", _fake_register_tasks)
    monkeypatch.setattr("src.cli.HueyWorkerProcess", _FakeWorker)

    from src.cli import _create_daemon_runtime

    ctx, worker = _create_daemon_runtime(runtime_paths)

    assert ctx is fake_ctx
    assert observed["create_kwargs"] == {
        "enable_watcher": False,
        "lazy_embeddings": True,
        "use_tasks": True,
        "index_path_override": runtime_paths.root,
        "global_runtime": True,
    }
    assert observed["queue_path"] == runtime_paths.queue_db_path
    assert observed["register"] == (
        fake_huey,
        fake_ctx.index_manager,
        fake_ctx.commit_indexer,
        100,
    )
    assert observed["worker_runtime_paths"] == runtime_paths
    assert worker is not None


def test_request_daemon_json_auto_start_ignores_project_context_for_startup(monkeypatch, tmp_path):
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(lambda cls: runtime_paths),
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=None,
            running=False,
            stale=False,
            responsive=False,
            ready=False,
        ),
    )

    def _fake_start_daemon(*, timeout_seconds=10.0, paths=None):
        observed["timeout_seconds"] = timeout_seconds
        observed["paths"] = paths
        return DaemonMetadata(
            pid=321,
            started_at=1.0,
            status="ready",
            socket_path=str(runtime_paths.socket_path),
        )

    def _fake_request_daemon_socket(socket_path, path, payload, timeout_seconds):
        observed["socket_path"] = socket_path
        observed["path"] = path
        observed["payload"] = payload
        observed["request_timeout_seconds"] = timeout_seconds
        return {"status": "ok", "value": 1}

    monkeypatch.setattr("src.cli.start_daemon", _fake_start_daemon)
    monkeypatch.setattr("src.cli.request_daemon_socket", _fake_request_daemon_socket)

    from src.cli import _request_daemon_json

    response = _request_daemon_json(
        "/api/search/query",
        {"project_context": "project-a"},
        project_override="project-a",
        auto_start=True,
    )

    assert response == {"status": "ok", "value": 1}
    assert observed == {
        "timeout_seconds": 10.0,
        "paths": runtime_paths,
        "socket_path": runtime_paths.socket_path,
        "path": "/api/search/query",
        "payload": {"project_context": "project-a"},
        "request_timeout_seconds": 30.0,
    }


def test_index_stats_reports_index_counts(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.manifest.json").write_text("{}", encoding="utf-8")

    class _FakeVector:
        def __len__(self):
            return 23

    class _FakeIndexManager:
        def __init__(self):
            self.vector = _FakeVector()
            self.loaded = False

        def load(self):
            self.loaded = True

        def get_document_count(self):
            return 7

    class _FakeCommitIndexer:
        def get_total_commits(self):
            return 11

    class _FakeIndexingConfig:
        documents_path = str(docs_dir)
        index_path = str(index_dir)
        exclude: list[str] = []
        exclude_hidden_dirs = True

    class _FakeConfig:
        indexing = _FakeIndexingConfig()

    class _FakeContext:
        def __init__(self):
            self.config = _FakeConfig()
            self.index_path = index_dir
            self.index_manager = _FakeIndexManager()
            self.commit_indexer = _FakeCommitIndexer()
            self.documents_roots = [docs_dir]
            self.watcher = None

        def discover_files(self):
            return [str(docs_dir / "a.md"), str(docs_dir / "b.md")]

        def discover_git_repositories(self):
            return [docs_dir]

        def get_index_state(self):
            return type(
                "_FakeIndexState",
                (),
                {
                    "to_dict": lambda self: {
                        "status": "ready",
                        "indexed_count": 7,
                        "total_count": 7,
                        "last_error": None,
                    }
                },
            )()

    fake_ctx = _FakeContext()

    from src.cli import _build_index_stats_payload

    payload = _build_index_stats_payload(fake_ctx)

    assert payload["indexed_documents"] == 7
    assert payload["indexed_chunks"] == 23
    assert payload["git_commits"] == 11
    assert payload["discovered_files"] == 2
    assert payload["index_db_path"] == str(index_dir / "index.db")


def test_index_stats_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "documents_path": "/docs",
            "index_path": "/index",
            "index_db_path": "/index/index.db",
            "manifest_path": "/index/index.manifest.json",
            "manifest_exists": True,
            "indexed_documents": 9,
            "indexed_chunks": 21,
            "discovered_files": 4,
            "git_commits": 12,
            "git_repositories": 2,
            "index_state": {"status": "ready", "indexed_count": 9, "total_count": 9, "last_error": None},
            "watcher_stats": {"events_received": 7, "events_processed": 5},
        },
    )

    result = runner.invoke(cli, ["index", "stats", "--json"])

    assert result.exit_code == 0
    assert '"indexed_documents": 9' in result.output
    assert '"git_commits": 12' in result.output
    assert '"index_state"' in result.output
    assert '"watcher_stats"' in result.output


def test_queue_status_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "queue_db_path": "/queue.db",
            "pending_count": 2,
            "scheduled_count": 1,
            "running_count": 0,
            "failed_count": 1,
            "worker_running": True,
            "backpressure_limit": 100,
            "backpressure_utilization": 0.02,
            "task_counts": {"_index_document": 2},
            "recent_failures": [
                {
                    "task_id": "abc",
                    "task_name": "_refresh_git_repository",
                    "error": "RuntimeError('boom')",
                    "retries": 0,
                    "traceback": "trace",
                }
            ],
        },
    )

    result = runner.invoke(cli, ["queue", "status", "--json"])

    assert result.exit_code == 0
    assert '"pending_count": 2' in result.output
    assert '"failed_count": 1' in result.output
    assert '"backpressure_limit": 100' in result.output
    assert '"_refresh_git_repository"' in result.output


def test_queue_status_requires_daemon_when_unavailable(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("src.cli._request_daemon_json", lambda *args, **kwargs: None)

    result = runner.invoke(cli, ["queue", "status", "--json"])

    assert result.exit_code == 1
    assert "Daemon unavailable" in result.output


def test_queue_status_reports_explicit_daemon_timeout(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda *args, **kwargs: {"status": "error", "error": "daemon_request_timed_out"},
    )

    result = runner.invoke(cli, ["queue", "status", "--json"])

    assert result.exit_code == 1
    assert "Daemon request timed out" in result.output


def test_request_daemon_json_uses_running_daemon(monkeypatch):
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1.0,
        status="starting",
        socket_path="/tmp/ragdocs.sock",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda paths=None: DaemonInspection(
            metadata=metadata,
            running=True,
            stale=False,
            responsive=True,
            ready=False,
        ),
    )
    monkeypatch.setattr(
        "src.cli.request_daemon_socket",
        lambda socket_path, path, payload, timeout_seconds: {"status": "ok", "path": path},
    )

    from src.cli import _request_daemon_json

    payload = _request_daemon_json(
        "/api/admin/tasks",
        {},
        project_override=None,
        auto_start=False,
    )

    assert payload == {"status": "ok", "path": "/api/admin/tasks"}


def test_query_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "query": payload["query"],
            "results": [{"doc_id": "doc-1", "score": 0.9, "content": "daemon result"}],
            "compression_stats": {},
            "strategy_stats": {},
        },
    )

    result = runner.invoke(cli, ["query", "daemon", "--json"])

    assert result.exit_code == 0
    assert '"query": "daemon"' in result.output
    assert '"daemon result"' in result.output


def test_build_initializing_search_payload_for_query_includes_machine_readable_fields():
    class _FakeIndexState:
        def to_dict(self):
            return {
                "status": "uninitialized",
                "indexed_count": 0,
                "total_count": 0,
                "last_error": None,
            }

    class _FakeContext:
        documents_roots = [Path("/docs/a"), Path("/docs/b")]
        commit_indexer = None

        def get_index_state(self):
            return _FakeIndexState()

    class _FakeCoordinator:
        state = LifecycleState.INITIALIZING

    from src.cli import _build_initializing_search_payload

    payload = _build_initializing_search_payload(
        _FakeContext(),
        _FakeCoordinator(),
        query="auth",
    )

    assert payload == {
        "status": "initializing",
        "message": "Search indices are still initializing. Retry shortly.",
        "query": "auth",
        "results": [],
        "lifecycle": "initializing",
        "daemon_scope": "global",
        "project_context_mode": "request_only",
        "configured_root_count": 2,
        "index_state": {
            "status": "uninitialized",
            "indexed_count": 0,
            "total_count": 0,
            "last_error": None,
        },
        "compression_stats": {},
        "strategy_stats": {},
    }


def test_build_initializing_search_payload_for_git_history_includes_commit_count():
    class _FakeIndexState:
        def to_dict(self):
            return {
                "status": "indexing",
                "indexed_count": 0,
                "total_count": 4,
                "last_error": None,
            }

    class _FakeCommitIndexer:
        def get_total_commits(self):
            return 17

    class _FakeContext:
        documents_roots = [Path("/docs")]
        commit_indexer = _FakeCommitIndexer()

        def get_index_state(self):
            return _FakeIndexState()

    class _FakeCoordinator:
        state = LifecycleState.INITIALIZING

    from src.cli import _build_initializing_search_payload

    payload = _build_initializing_search_payload(
        _FakeContext(),
        _FakeCoordinator(),
        query="fix bug",
        include_git_metadata=True,
    )

    assert payload["status"] == "initializing"
    assert payload["query"] == "fix bug"
    assert payload["results"] == []
    assert payload["total_commits_indexed"] == 17
    assert payload["index_state"]["status"] == "indexing"


def test_query_surfaces_initializing_response_in_human_output(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "status": "initializing",
            "message": "Search indices are still initializing. Retry shortly.",
            "query": payload["query"],
            "results": [],
            "lifecycle": "initializing",
            "configured_root_count": 2,
            "index_state": {
                "status": "uninitialized",
                "indexed_count": 0,
                "total_count": 0,
                "last_error": None,
            },
            "compression_stats": {},
            "strategy_stats": {},
        },
    )

    result = runner.invoke(cli, ["query", "daemon"])

    assert result.exit_code == 0
    assert "Search service is initializing" in result.output
    assert "Lifecycle:" in result.output
    assert "Configured roots:" in result.output
    assert "Results will appear once background initialization completes." in result.output


def test_search_commits_surfaces_initializing_response_in_human_output(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "status": "initializing",
            "message": "Search indices are still initializing. Retry shortly.",
            "query": payload["query"],
            "results": [],
            "lifecycle": "initializing",
            "configured_root_count": 1,
            "index_state": {
                "status": "indexing",
                "indexed_count": 0,
                "total_count": 8,
                "last_error": None,
            },
            "total_commits_indexed": 0,
        },
    )

    result = runner.invoke(cli, ["search-commits", "daemon"])

    assert result.exit_code == 0
    assert "Search service is initializing" in result.output
    assert "Total commits indexed:" in result.output
    assert "Results will appear once background initialization completes." in result.output


def test_query_passes_project_context_and_filter(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    def _fake_request(path, payload, project_override, auto_start, allow_error=False):
        observed["path"] = path
        observed["payload"] = payload
        observed["project_override"] = project_override
        return {
            "query": payload["query"],
            "results": [],
            "compression_stats": {},
            "strategy_stats": {},
        }

    monkeypatch.setattr("src.cli._request_daemon_json", _fake_request)

    result = runner.invoke(
        cli,
        [
            "query",
            "daemon",
            "--json",
            "--project",
            "project-a",
            "--project-filter",
            "project-a",
            "--project-filter",
            "project-b",
        ],
    )

    assert result.exit_code == 0
    assert observed["project_override"] == "project-a"
    assert observed["path"] == "/api/search/query"
    assert observed["payload"] == {
        "query": "daemon",
        "top_n": 5,
        "project_filter": ["project-a", "project-b"],
        "project_context": "project-a",
    }


def test_search_commits_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start, allow_error=False: {
            "query": payload["query"],
            "total_commits_indexed": 5,
            "results": [
                {
                    "hash": "abcdef12",
                    "title": "daemon commit",
                    "author": "Andy",
                    "committer": "Andy",
                    "timestamp": 1,
                    "message": "msg",
                    "files_changed": ["src/cli.py"],
                    "delta_truncated": "diff",
                    "score": 0.8,
                    "repo_path": "/repo",
                }
            ],
        },
    )

    result = runner.invoke(cli, ["search-commits", "daemon", "--json"])

    assert result.exit_code == 0
    assert '"query": "daemon"' in result.output
    assert '"daemon commit"' in result.output


def test_search_commits_passes_project_context_and_filter(monkeypatch):
    runner = CliRunner()
    observed: dict[str, object] = {}

    def _fake_request(path, payload, project_override, auto_start, allow_error=False):
        observed["path"] = path
        observed["payload"] = payload
        observed["project_override"] = project_override
        return {
            "query": payload["query"],
            "total_commits_indexed": 0,
            "results": [],
        }

    monkeypatch.setattr("src.cli._request_daemon_json", _fake_request)

    result = runner.invoke(
        cli,
        [
            "search-commits",
            "daemon",
            "--json",
            "--project",
            "project-a",
            "--project-filter",
            "project-a",
        ],
    )

    assert result.exit_code == 0
    assert observed["project_override"] == "project-a"
    assert observed["path"] == "/api/search/git-history"
    assert observed["payload"] == {
        "query": "daemon",
        "top_n": 5,
        "files_glob": None,
        "after_timestamp": None,
        "before_timestamp": None,
        "project_filter": ["project-a"],
        "project_context": "project-a",
    }
