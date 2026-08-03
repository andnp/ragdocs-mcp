import pytest

from mcp_markdown_ragdocs.app.composition import build_kernel
from mcp_markdown_ragdocs.app.search import build_reranker, to_record_search_config
from mcp_markdown_ragdocs.config import Config, IndexingConfig, SearchConfig


def test_reranking_is_disabled_by_default() -> None:
    config = SearchConfig()

    assert build_reranker(config) is None
    assert to_record_search_config(config).rerank_budget == 0


def test_reranking_requires_model_and_budget() -> None:
    with pytest.raises(ValueError, match="configured together"):
        SearchConfig(rerank_budget=1)

    with pytest.raises(ValueError, match="configured together"):
        SearchConfig(reranker_model="cross-encoder/test")


def test_enabled_reranking_validates_dependency_without_loading_model(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("mcp_markdown_ragdocs.config.importlib.util.find_spec", lambda _: object())
    config = SearchConfig(reranker_model="cross-encoder/test", rerank_budget=2)

    assert to_record_search_config(config).rerank_budget == 2
    reranker = build_reranker(config)
    assert reranker is not None
    assert reranker.model_name == "cross-encoder/test"

    monkeypatch.setenv("MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS", "1")
    context = build_kernel(
        Config(
            indexing=IndexingConfig(
                documents_path=str(tmp_path),
                index_path=str(tmp_path / "index"),
            ),
            search=config,
        ),
        enable_watcher=False,
    )
    assert context.index_manager.kernel.pipeline._reranker is not None
