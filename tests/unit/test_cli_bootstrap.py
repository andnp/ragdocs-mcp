from pathlib import Path
from typing import cast

import pytest

from searchkernel import cli


def test_should_reexec_into_repo_venv_when_required_dependency_missing(
    monkeypatch,
    tmp_path,
):
    repo_python = tmp_path / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("", encoding="utf-8")
    repo_python.chmod(0o755)

    monkeypatch.delenv(cli._CLI_REEXEC_GUARD, raising=False)
    monkeypatch.setattr(cli, "_repo_venv_python", lambda: repo_python)
    monkeypatch.setattr(cli.sys, "executable", "/usr/bin/python")

    assert cli._should_reexec_into_repo_venv() is True


def test_should_not_reexec_when_current_interpreter_is_repo_venv(
    monkeypatch,
    tmp_path,
):
    repo_python = tmp_path / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("", encoding="utf-8")
    repo_python.chmod(0o755)

    monkeypatch.delenv(cli._CLI_REEXEC_GUARD, raising=False)
    monkeypatch.setattr(cli, "_repo_venv_python", lambda: repo_python)
    monkeypatch.setattr(cli.sys, "executable", str(repo_python.resolve()))

    assert cli._should_reexec_into_repo_venv() is False


def test_reexec_into_repo_venv_uses_module_invocation(monkeypatch, tmp_path):
    repo_python = tmp_path / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("", encoding="utf-8")
    repo_python.chmod(0o755)

    observed: dict[str, object] = {}

    def _fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        observed["path"] = path
        observed["argv"] = argv
        observed["env"] = env
        raise RuntimeError("reexec intercepted")

    monkeypatch.setattr(cli, "_repo_venv_python", lambda: repo_python)
    monkeypatch.setattr(cli.os, "execve", _fake_execve)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["searchkernel/cli.py", "daemon-internal-run", "--runtime-root", "/tmp/runtime"],
    )
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    with pytest.raises(RuntimeError, match="reexec intercepted"):
        cli._reexec_into_repo_venv()

    repo_root = Path(cli.__file__).resolve().parents[1]
    assert observed["path"] == str(repo_python)
    assert observed["argv"] == [
        str(repo_python),
        "-m",
        "searchkernel.cli",
        "daemon-internal-run",
        "--runtime-root",
        "/tmp/runtime",
    ]
    env = cast(dict[str, str], observed["env"])
    assert env[cli._CLI_REEXEC_GUARD] == "1"
    assert env["PYTHONPATH"].startswith(str(repo_root))
    assert env["PYTHONPATH"].endswith("/existing/pythonpath")
