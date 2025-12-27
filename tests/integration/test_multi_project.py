import pytest
from src.config import Config
from src.indexing.manager import IndexManager
from src.indices.vector import VectorIndex
from src.indices.keyword import KeywordIndex
from src.indices.graph import GraphStore


@pytest.fixture
def projects_setup(tmp_path, monkeypatch):
    project_a = tmp_path / "project_a"
    project_a.mkdir()

    project_b = tmp_path / "project_b"
    project_b.mkdir()

    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_dir = tmp_path / ".config" / "mcp-markdown-ragdocs"
    global_config_dir.mkdir(parents=True)

    config_content = f"""
[[projects]]
name = "project_a"
path = "{project_a}"

[[projects]]
name = "project_b"
path = "{project_b}"
"""
    (global_config_dir / "config.toml").write_text(config_content)

    return {
        "project_a": project_a,
        "project_b": project_b,
        "global_config": global_config_dir,
        "tmp": tmp_path
    }


def test_project_isolation(projects_setup):
    project_a = projects_setup["project_a"]
    project_b = projects_setup["project_b"]
    tmp_path = projects_setup["tmp"]

    data_dir = tmp_path / ".local" / "share" / "mcp-markdown-ragdocs"

    config_a = Config()
    config_a.indexing.index_path = str(data_dir / "project_a")
    config_a.indexing.documents_path = str(project_a)

    manager_a = IndexManager(
        config_a,
        VectorIndex(),
        KeywordIndex(),
        GraphStore()
    )

    doc_a = project_a / "doc.md"
    doc_a.write_text("# Project A Document")
    manager_a.index_document(str(doc_a))
    manager_a.persist()

    config_b = Config()
    config_b.indexing.index_path = str(data_dir / "project_b")
    config_b.indexing.documents_path = str(project_b)

    manager_b = IndexManager(
        config_b,
        VectorIndex(),
        KeywordIndex(),
        GraphStore()
    )

    doc_b = project_b / "doc.md"
    doc_b.write_text("# Project B Document")
    manager_b.index_document(str(doc_b))
    manager_b.persist()

    assert (data_dir / "project_a" / "vector").exists()
    assert (data_dir / "project_b" / "vector").exists()

    assert manager_a.get_document_count() == 1
    assert manager_b.get_document_count() == 1

    manager_a_reload = IndexManager(
        config_a,
        VectorIndex(),
        KeywordIndex(),
        GraphStore()
    )
    manager_a_reload.load()

    assert manager_a_reload.get_document_count() == 1
