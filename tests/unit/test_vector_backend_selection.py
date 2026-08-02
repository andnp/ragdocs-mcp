import pytest

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.indexing.record_manager import build_embedding_provider


def test_test_mode_builds_deterministic_provider(monkeypatch) -> None:
    monkeypatch.setenv("MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS", "1")

    provider = build_embedding_provider(Config(), "ignored-in-test-mode")

    assert provider.model_name == "__deterministic_fake__"
    assert provider.dim == 384


def test_canonical_provider_rejects_non_ollama_runtime(monkeypatch) -> None:
    monkeypatch.delenv("MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS", raising=False)
    config = Config()
    config.embedding.provider = "unsupported"

    with pytest.raises(ValueError, match="canonical indexing requires"):
        build_embedding_provider(config, "model")
