from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from src import cli as cli_module


class _FakeDbManager:
    def __init__(self):
        self.close_calls = 0
        self.initialize_schema_calls = 0

    def close(self):
        self.close_calls += 1

    def initialize_schema(self):
        self.initialize_schema_calls += 1


class _FakeVector:
    def __init__(self):
        self.build_calls: list[tuple[int, int]] = []
        self._concept_vocabulary: dict[str, int] = {}

    def build_concept_vocabulary(self, max_terms: int, min_frequency: int):
        self.build_calls.append((max_terms, min_frequency))
        self._concept_vocabulary = {"rebuild": 1}


class _FakeIndexManager:
    def __init__(self):
        self.index_documents_calls: list[dict[str, object]] = []
        self.persist_calls = 0
        self.vector = _FakeVector()

    def index_document(self, file_path: str, force: bool = False):
        raise AssertionError(
            f"rebuild should batch index documents, but called index_document({file_path!r}, force={force!r})"
        )

    def index_documents(
        self,
        file_paths: list[str],
        force: bool = False,
        persist: bool = False,
    ):
        self.index_documents_calls.append(
            {
                "file_paths": list(file_paths),
                "force": force,
                "persist": persist,
            }
        )

    def persist(self):
        self.persist_calls += 1


def _build_fake_context(tmp_path: Path, files_to_index: list[str]):
    docs_path = tmp_path / "docs"
    index_path = tmp_path / ".index_data"
    db_manager = _FakeDbManager()
    index_manager = _FakeIndexManager()
    config = SimpleNamespace(
        indexing=SimpleNamespace(
            documents_path=str(docs_path),
            rebuild_checkpoint_interval=2,
        ),
        llm=SimpleNamespace(embedding_model="local"),
        chunking=SimpleNamespace(
            strategy="header_based",
            min_chunk_chars=200,
            max_chunk_chars=2000,
            overlap_chars=100,
        ),
        git_indexing=SimpleNamespace(enabled=False),
    )
    return SimpleNamespace(
        index_path=index_path,
        db_manager=db_manager,
        config=config,
        documents_roots=[docs_path],
        index_manager=index_manager,
        commit_indexer=None,
        discover_files=lambda: list(files_to_index),
        discover_git_repositories=lambda: [],
    )


def test_rebuild_index_batches_with_force_and_preserves_checkpoint_manifesting(
    tmp_path: Path,
    monkeypatch,
):
    docs_path = tmp_path / "docs"
    docs_path.mkdir(parents=True)
    files_to_index = []
    for name in ["alpha", "beta", "gamma"]:
        file_path = docs_path / f"{name}.md"
        file_path.write_text(f"# {name}\n", encoding="utf-8")
        files_to_index.append(str(file_path))

    ctx = _build_fake_context(tmp_path, files_to_index)
    saved_manifests: list[tuple[Path, object]] = []

    monkeypatch.setattr(cli_module, "_ensure_runtime_auto_registration", lambda project: None)
    monkeypatch.setattr(cli_module, "_clear_document_index_artifacts", lambda index_path: None)
    monkeypatch.setattr(
        cli_module.ApplicationContext,
        "create",
        classmethod(lambda cls, **kwargs: ctx),
    )
    monkeypatch.setattr(
        cli_module,
        "save_manifest",
        lambda index_path, manifest: saved_manifests.append((index_path, manifest)),
    )

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["rebuild-index"])

    assert result.exit_code == 0, result.output
    assert ctx.db_manager.close_calls == 1
    assert ctx.db_manager.initialize_schema_calls == 1
    assert ctx.index_manager.index_documents_calls == [
        {
            "file_paths": files_to_index[:2],
            "force": True,
            "persist": False,
        },
        {
            "file_paths": files_to_index[2:],
            "force": True,
            "persist": False,
        },
    ]
    assert ctx.index_manager.persist_calls == 4
    assert ctx.index_manager.vector.build_calls == [(2000, 3)]
    assert [len(manifest.indexed_files) for _, manifest in saved_manifests] == [2, 3, 3]
    assert "Checkpoint persisted: 2/3 documents" in result.output
    assert "Checkpoint persisted: 3/3 documents" in result.output
    assert "Successfully rebuilt index: 3 documents indexed" in result.output