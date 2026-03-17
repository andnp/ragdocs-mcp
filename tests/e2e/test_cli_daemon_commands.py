from __future__ import annotations

from click.testing import CliRunner

from src.cli import cli
from src.daemon import RuntimePaths
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
        lambda: DaemonInspection(
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
    assert '"metadata_path":' in result.output
    assert '"index_db_path": "/tmp/index.db"' in result.output


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


def test_create_daemon_runtime_enables_task_mode(monkeypatch, tmp_path):
    observed: dict[str, object] = {}

    class _FakeIndexManager:
        pass

    class _FakeContext:
        def __init__(self):
            self.index_manager = _FakeIndexManager()
            self.commit_indexer = object()

    class _FakeWorker:
        def __init__(self, huey):
            observed["worker_huey"] = huey

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

    def _fake_register_tasks(huey, index_manager, commit_indexer=None):
        observed["register"] = (huey, index_manager, commit_indexer)

    monkeypatch.setattr("src.cli.ApplicationContext.create", _fake_create)
    monkeypatch.setattr("src.cli.get_huey", _fake_get_huey)
    monkeypatch.setattr("src.cli.register_tasks", _fake_register_tasks)
    monkeypatch.setattr("src.cli.HueyWorker", _FakeWorker)

    from src.cli import _create_daemon_runtime

    ctx, worker = _create_daemon_runtime("docs", runtime_paths)

    assert ctx is fake_ctx
    assert observed["create_kwargs"] == {
        "project_override": "docs",
        "enable_watcher": True,
        "lazy_embeddings": True,
        "use_tasks": True,
        "index_path_override": runtime_paths.root,
    }
    assert observed["queue_path"] == runtime_paths.queue_db_path
    assert observed["register"] == (
        fake_huey,
        fake_ctx.index_manager,
        fake_ctx.commit_indexer,
    )
    assert worker is not None


def test_index_stats_reports_index_counts(monkeypatch, tmp_path):
    runner = CliRunner()
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

        def discover_files(self):
            return [str(docs_dir / "a.md"), str(docs_dir / "b.md")]

    fake_ctx = _FakeContext()
    monkeypatch.setattr(
        "src.cli.ApplicationContext.create",
        lambda **kwargs: fake_ctx,
    )
    monkeypatch.setattr("src.cli.discover_git_repositories", lambda *args, **kwargs: [docs_dir])

    result = runner.invoke(cli, ["index", "stats", "--json"])

    assert result.exit_code == 0
    assert '"indexed_documents": 7' in result.output
    assert '"indexed_chunks": 23' in result.output
    assert '"git_commits": 11' in result.output
    assert '"discovered_files": 2' in result.output
    assert f'"index_db_path": "{index_dir / "index.db"}"' in result.output


def test_index_stats_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start: {
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
        },
    )

    result = runner.invoke(cli, ["index", "stats", "--json"])

    assert result.exit_code == 0
    assert '"indexed_documents": 9' in result.output
    assert '"git_commits": 12' in result.output


def test_queue_status_prefers_daemon_transport(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "src.cli._request_daemon_json",
        lambda path, payload, project_override, auto_start: {
            "queue_db_path": "/queue.db",
            "pending_count": 2,
            "scheduled_count": 1,
            "running_count": 0,
            "failed_count": 1,
            "worker_running": True,
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
    assert '"_refresh_git_repository"' in result.output


def test_queue_status_falls_back_to_local_queue(monkeypatch, tmp_path):
    runner = CliRunner()
    observed: dict[str, object] = {}
    runtime_paths = RuntimePaths(
        root=tmp_path,
        index_db_path=tmp_path / "index.db",
        queue_db_path=tmp_path / "queue.db",
        metadata_path=tmp_path / "daemon.json",
        lock_path=tmp_path / "daemon.lock",
        socket_path=tmp_path / "daemon.sock",
    )

    monkeypatch.setattr("src.cli._request_daemon_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        RuntimePaths,
        "resolve",
        classmethod(lambda cls: runtime_paths),
    )
    monkeypatch.setattr("src.cli.get_huey", lambda path: observed.setdefault("path", path) or object())
    monkeypatch.setattr(
        "src.cli.get_queue_stats",
        lambda huey, worker_running=False: type(
            "_Stats",
            (),
            {
                "to_dict": lambda self: {
                    "pending_count": 0,
                    "scheduled_count": 0,
                    "running_count": 0,
                    "failed_count": 0,
                    "worker_running": worker_running,
                    "task_counts": {},
                    "recent_failures": [],
                }
            },
        )(),
    )

    result = runner.invoke(cli, ["queue", "status", "--json"])

    assert result.exit_code == 0
    assert observed["path"] == runtime_paths.queue_db_path
    assert '"queue_db_path":' in result.output


def test_request_daemon_json_uses_running_daemon(monkeypatch):
    metadata = DaemonMetadata(
        pid=4321,
        started_at=1.0,
        status="starting",
        socket_path="/tmp/ragdocs.sock",
    )
    monkeypatch.setattr(
        "src.cli.inspect_daemon",
        lambda: DaemonInspection(
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
        "/api/admin/queue-status",
        {},
        project_override=None,
        auto_start=False,
    )

    assert payload == {"status": "ok", "path": "/api/admin/queue-status"}


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