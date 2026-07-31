import tomllib
from pathlib import Path

import pytest

from ragdocs.config import (
    ProjectConfig,
    derive_auto_registration_root,
    detect_project,
    ensure_runtime_project_registered,
    get_project_root_warnings,
)


@pytest.fixture
def temp_config_home(tmp_path, monkeypatch):
    config_dir = tmp_path / ".config" / "mcp-markdown-ragdocs"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    return config_dir


@pytest.fixture
def bare_worktree_layout(tmp_path):
    repo_root = tmp_path / "repo"
    common_git_dir = repo_root / ".git"
    common_git_dir.mkdir(parents=True)
    (common_git_dir / "config").write_text(
        "[core]\n\tbare = true\n",
        encoding="utf-8",
    )
    (common_git_dir / "HEAD").write_text(
        "ref: refs/heads/main\n",
        encoding="utf-8",
    )

    worktrees_dir = common_git_dir / "worktrees"
    worktrees_dir.mkdir()

    def _add_worktree(name: str, branch: str) -> Path:
        root = repo_root / name
        root.mkdir(parents=True)
        admin_dir = worktrees_dir / name
        admin_dir.mkdir()
        git_file = root / ".git"
        git_file.write_text(f"gitdir: {admin_dir}\n", encoding="utf-8")
        (admin_dir / "gitdir").write_text(str(git_file), encoding="utf-8")
        (admin_dir / "commondir").write_text("../..\n", encoding="utf-8")
        (admin_dir / "HEAD").write_text(
            f"ref: refs/heads/{branch}\n",
            encoding="utf-8",
        )
        return root

    main_root = _add_worktree("main", "main")
    feature_root = _add_worktree("feature", "feature/test")

    return {
        "repo_root": repo_root,
        "common_git_dir": common_git_dir,
        "main": main_root,
        "feature": feature_root,
    }


@pytest.fixture
def default_branch_worktree_layout(tmp_path):
    repo_root = tmp_path / "repo"
    common_git_dir = repo_root / ".git"
    common_git_dir.mkdir(parents=True)
    (common_git_dir / "config").write_text(
        "[core]\n\tbare = true\n",
        encoding="utf-8",
    )
    (common_git_dir / "HEAD").write_text(
        "ref: refs/heads/trunk\n",
        encoding="utf-8",
    )

    worktrees_dir = common_git_dir / "worktrees"
    worktrees_dir.mkdir()

    def _add_worktree(name: str, branch: str) -> Path:
        root = repo_root / name
        root.mkdir(parents=True)
        admin_dir = worktrees_dir / name
        admin_dir.mkdir()
        git_file = root / ".git"
        git_file.write_text(f"gitdir: {admin_dir}\n", encoding="utf-8")
        (admin_dir / "gitdir").write_text(str(git_file), encoding="utf-8")
        (admin_dir / "commondir").write_text("../..\n", encoding="utf-8")
        (admin_dir / "HEAD").write_text(
            f"ref: refs/heads/{branch}\n",
            encoding="utf-8",
        )
        return root

    trunk_root = _add_worktree("stable", "trunk")
    feature_root = _add_worktree("experiment", "feature/experiment")

    return {
        "repo_root": repo_root,
        "common_git_dir": common_git_dir,
        "trunk": trunk_root,
        "feature": feature_root,
    }


def test_derive_auto_registration_root_prefers_main_worktree(bare_worktree_layout):
    result = derive_auto_registration_root(bare_worktree_layout["feature"])

    assert result == bare_worktree_layout["main"].resolve()


def test_derive_auto_registration_root_prefers_default_branch_when_main_missing(
    default_branch_worktree_layout,
):
    result = derive_auto_registration_root(default_branch_worktree_layout["feature"])

    assert result == default_branch_worktree_layout["trunk"].resolve()


def test_detect_project_maps_sibling_worktree_to_registered_canonical_project(
    bare_worktree_layout,
):
    projects = [
        ProjectConfig(name="repo-main", path=str(bare_worktree_layout["main"])),
    ]

    result = detect_project(cwd=bare_worktree_layout["feature"], projects=projects)

    assert result == "repo-main"


def test_ensure_runtime_project_registered_persists_canonical_worktree(
    bare_worktree_layout,
    temp_config_home,
):
    result = ensure_runtime_project_registered(cwd=bare_worktree_layout["feature"])

    assert result.changed is True
    assert result.project_name == "main"
    assert result.project_path == str(bare_worktree_layout["main"].resolve())

    config_path = temp_config_home / "config.toml"
    with open(config_path, "rb") as handle:
        data = tomllib.load(handle)

    assert data["projects"] == [
        {
            "name": "main",
            "path": str(bare_worktree_layout["main"].resolve()),
        }
    ]


def test_get_project_root_warnings_for_duplicate_worktree_repo_identity(
    bare_worktree_layout,
):
    projects = [
        ProjectConfig(name="repo-main", path=str(bare_worktree_layout["main"])),
        ProjectConfig(
            name="repo-feature",
            path=str(bare_worktree_layout["feature"]),
        ),
    ]

    warnings = get_project_root_warnings(projects)

    assert warnings == [
        (
            "Registered projects repo-feature, repo-main point to git worktrees from "
            f"the same repository identity '{bare_worktree_layout['common_git_dir']}'."
        )
    ]
