from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
from click.testing import CliRunner

pytest.importorskip("zmq")

from src.cli import cli
from src.daemon.health import probe_daemon_socket
from src.daemon.management import stop_daemon
from src.daemon.paths import RuntimePaths
from src.mcp.server import MCPServer


def _write_test_config(tmp_path: Path, docs_path: Path) -> None:
    config_dir = tmp_path / ".mcp-markdown-ragdocs"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
[indexing]
documents_path = "{docs_path}"
index_path = ".index_data"

[llm]
embedding_model = "local"

[search]
semantic_weight = 1.0
keyword_weight = 1.0

[chunking]
strategy = "header_based"
min_chunk_chars = 200
max_chunk_chars = 2000
""",
        encoding="utf-8",
    )


async def _wait_for_daemon_socket(socket_path: Path, timeout_seconds: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        metadata = await asyncio.to_thread(probe_daemon_socket, socket_path)
        if metadata is not None:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("Timed out waiting for daemon health socket")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def daemon_test_env(tmp_path: Path) -> Path:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "readme.md").write_text(
        """# Thin Client Test\n\n## Authentication\n\nAuthentication uses JWT tokens.\n""",
        encoding="utf-8",
    )
    (docs_path / "guide.md").write_text(
        """# Guide\n\n## Setup\n\nInstall dependencies first.\n""",
        encoding="utf-8",
    )
    _write_test_config(tmp_path, docs_path)
    return tmp_path


def _configure_shared_runtime_home(
    daemon_test_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RuntimePaths:
    short_state_home = Path("/tmp") / f"rd-{os.getpid()}-{daemon_test_env.name[-8:]}"
    short_state_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(short_state_home))
    return RuntimePaths.resolve()


async def _wait_for_query_result(
    runner: CliRunner,
    query_text: str,
    expected_snippet: str,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        result = await asyncio.to_thread(
            runner.invoke,
            cli,
            ["query", query_text, "--json"],
        )
        if result.exit_code == 0:
            payload = json.loads(result.output)
            if any(
                expected_snippet in item.get("content", "")
                for item in payload.get("results", [])
            ):
                return
        await asyncio.sleep(0.2)
    raise AssertionError(
        f"Timed out waiting for query '{query_text}' to return snippet '{expected_snippet}'"
    )


@pytest.mark.asyncio
async def test_daemon_backed_cli_query_and_index_stats_smoke(
    runner: CliRunner,
    daemon_test_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cwd = Path.cwd()
    os.chdir(daemon_test_env)
    runtime_paths = _configure_shared_runtime_home(daemon_test_env, monkeypatch)

    try:
        rebuild = await asyncio.to_thread(runner.invoke, cli, ["rebuild-index"])
        assert rebuild.exit_code == 0, rebuild.output

        try:
            query_result = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["query", "authentication", "--json"],
            )
            assert query_result.exit_code == 0, query_result.output
            query_payload = json.loads(query_result.output)
            assert query_payload["results"]
            assert any(
                "Authentication" in result.get("content", "")
                or "authentication" in result.get("content", "").lower()
                for result in query_payload["results"]
            )
            await _wait_for_daemon_socket(runtime_paths.socket_path)

            stats_result = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["index", "stats", "--json"],
            )
            assert stats_result.exit_code == 0, stats_result.output
            stats_payload = json.loads(stats_result.output)
            assert stats_payload["indexed_documents"] >= 2
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stop_daemon, paths=runtime_paths)
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_daemon_backed_mcp_query_documents_smoke(
    runner: CliRunner,
    daemon_test_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cwd = Path.cwd()
    os.chdir(daemon_test_env)
    runtime_paths = _configure_shared_runtime_home(daemon_test_env, monkeypatch)

    try:
        rebuild = await asyncio.to_thread(runner.invoke, cli, ["rebuild-index"])
        assert rebuild.exit_code == 0, rebuild.output

        try:
            server = MCPServer()
            contents = await server._call_remote_tool(
                "query_documents",
                {"query": "authentication", "top_n": 1},
            )

            assert contents is not None
            assert len(contents) == 1
            assert "Search Results" in contents[0].text
            await _wait_for_daemon_socket(runtime_paths.socket_path)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stop_daemon, paths=runtime_paths)
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_daemon_survives_sequential_cli_and_mcp_requests(
    runner: CliRunner,
    daemon_test_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cwd = Path.cwd()
    os.chdir(daemon_test_env)
    runtime_paths = _configure_shared_runtime_home(daemon_test_env, monkeypatch)

    try:
        rebuild = await asyncio.to_thread(runner.invoke, cli, ["rebuild-index"])
        assert rebuild.exit_code == 0, rebuild.output

        try:
            first_query = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["query", "authentication", "--json"],
            )
            assert first_query.exit_code == 0, first_query.output
            await _wait_for_daemon_socket(runtime_paths.socket_path)

            second_query = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["query", "setup", "--json"],
            )
            assert second_query.exit_code == 0, second_query.output
            await _wait_for_daemon_socket(runtime_paths.socket_path)

            server = MCPServer()
            contents = await server._call_remote_tool(
                "query_documents",
                {"query": "authentication", "top_n": 1},
            )
            assert contents is not None
            assert len(contents) == 1
            await _wait_for_daemon_socket(runtime_paths.socket_path)

            stats = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["index", "stats", "--json"],
            )
            assert stats.exit_code == 0, stats.output
            await _wait_for_daemon_socket(runtime_paths.socket_path)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stop_daemon, paths=runtime_paths)
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_mcp_run_stays_daemon_idle_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServer()

    @asynccontextmanager
    async def _fake_stdio_server():
        yield object(), object()

    async def _fake_server_run(read_stream, write_stream, init_options):
        return None

    monkeypatch.setattr(
        "src.mcp.server.stdio_server",
        _fake_stdio_server,
    )
    monkeypatch.setattr(
        server.server,
        "run",
        _fake_server_run,
    )
    monkeypatch.setattr(
        "src.mcp.server.start_daemon",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("daemon should not be contacted during stdio startup")),
    )

    await server.run()


@pytest.mark.asyncio
async def test_task_driven_document_update_survives_daemon_restart(
    runner: CliRunner,
    daemon_test_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cwd = Path.cwd()
    os.chdir(daemon_test_env)
    runtime_paths = _configure_shared_runtime_home(daemon_test_env, monkeypatch)
    guide_path = daemon_test_env / "docs" / "guide.md"
    marker = "RestartPersistenceMarker-20260317"

    try:
        rebuild = await asyncio.to_thread(runner.invoke, cli, ["rebuild-index"])
        assert rebuild.exit_code == 0, rebuild.output

        start = await asyncio.to_thread(
            runner.invoke,
            cli,
            ["daemon", "start", "--timeout", "20"],
        )
        assert start.exit_code == 0, start.output

        try:
            guide_path.write_text(
                f"# Guide\n\n## Setup\n\nInstall dependencies first.\n\n## Persistence\n\n{marker}\n",
                encoding="utf-8",
            )

            await _wait_for_query_result(runner, marker, marker)

            stop = await asyncio.to_thread(runner.invoke, cli, ["daemon", "stop"])
            assert stop.exit_code == 0, stop.output

            restart = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["daemon", "start", "--timeout", "20"],
            )
            assert restart.exit_code == 0, restart.output

            post_restart = await asyncio.to_thread(
                runner.invoke,
                cli,
                ["query", marker, "--json"],
            )
            assert post_restart.exit_code == 0, post_restart.output
            payload = json.loads(post_restart.output)
            assert any(
                marker in item.get("content", "")
                for item in payload.get("results", [])
            )
        finally:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stop_daemon, paths=runtime_paths)
    finally:
        os.chdir(original_cwd)