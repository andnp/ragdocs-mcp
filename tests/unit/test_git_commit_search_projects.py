from src.config import Config, IndexingConfig, ProjectConfig
from src.git.commit_search import apply_project_semantics


def _config() -> Config:
    return Config(
        indexing=IndexingConfig(documents_path="/docs", index_path="/index"),
        projects=[
            ProjectConfig(name="project-a", path="/repos/project-a"),
            ProjectConfig(name="project-b", path="/repos/project-b"),
        ],
    )


def test_apply_project_semantics_filters_and_boosts_matching_project() -> None:
    commits = [
        {"hash": "a", "score": 0.4, "repo_path": "/repos/project-a/.git"},
        {"hash": "b", "score": 0.5, "repo_path": "/repos/project-b/.git"},
    ]

    results = apply_project_semantics(
        commits,
        config=_config(),
        project_filter={"project-a"},
        project_context="project-a",
    )

    assert len(results) == 1
    assert results[0]["hash"] == "a"
    assert results[0]["project_id"] == "project-a"
    assert results[0]["score"] == 0.48


def test_apply_project_semantics_is_noop_without_filter_or_context() -> None:
    commits = [{"hash": "a", "score": 0.4, "repo_path": "/repos/project-a/.git"}]

    results = apply_project_semantics(
        commits,
        config=_config(),
        project_filter=None,
        project_context=None,
    )

    assert results[0]["project_id"] == "project-a"
    assert results[0]["score"] == 0.4


def test_apply_project_semantics_drops_unknown_project_when_filtering() -> None:
    commits = [{"hash": "a", "score": 0.4, "repo_path": "/repos/unknown/.git"}]

    results = apply_project_semantics(
        commits,
        config=_config(),
        project_filter={"project-a"},
        project_context=None,
    )

    assert results == []
