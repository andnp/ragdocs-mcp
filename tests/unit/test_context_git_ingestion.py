from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from mcp_markdown_ragdocs.config import Config, IndexingConfig, ProjectConfig
from mcp_markdown_ragdocs.context import ApplicationContext


def test_startup_git_ingestion_assigns_project_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    config = Config(
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "index"),
        ),
        projects=[ProjectConfig(name="repo-project", path=str(repo))],
    )
    observed: dict[str, object] = {}

    class _Source:
        def __init__(self, _git_dir: Path, *, workspace_id: str | None) -> None:
            observed["workspace_id"] = workspace_id

        def iter_records(self):
            return iter(())

    monkeypatch.setattr(
        "mcp_markdown_ragdocs.adapters.sources.git.GitContentSource",
        _Source,
    )
    context = cast(Any, object.__new__(ApplicationContext))
    context.config = config
    context.index_manager = SimpleNamespace(index_record=lambda _record: True)

    context._ingest_git_records_into_kernel_index([git_dir])

    assert observed["workspace_id"] == "repo-project"
