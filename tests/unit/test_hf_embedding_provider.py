"""Unit tests for the HuggingFace EmbeddingProvider adapter.

These load the real Qwen3-Embedding-0.6B model (~1.2GB on first run), so
they are marked slow. A session-scoped fixture shares the loaded model.
"""

import math

import pytest

from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider
from searchkernel.ports.embedding import EmbeddingProvider

pytestmark = pytest.mark.slow


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


@pytest.fixture(scope="module")
def provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider()


def test_satisfies_port(provider: HuggingFaceEmbeddingProvider):
    assert isinstance(provider, EmbeddingProvider)


def test_native_dim_is_1024(provider: HuggingFaceEmbeddingProvider):
    assert provider.dim == 1024


def test_embed_returns_normalized_vectors_of_dim(
    provider: HuggingFaceEmbeddingProvider,
):
    texts = ["Cats are small carnivorous mammals.", "PostgreSQL is a database."]
    vecs = provider.embed(texts)

    assert len(vecs) == len(texts)
    for vec in vecs:
        assert len(vec) == provider.dim
        assert _l2_norm(vec) == pytest.approx(1.0, abs=1e-2)


def test_embed_query_differs_from_embed(provider: HuggingFaceEmbeddingProvider):
    text = "How do neural networks learn?"

    doc_vec = provider.embed([text])[0]
    query_vec = provider.embed_query(text)

    assert len(query_vec) == provider.dim
    assert _l2_norm(query_vec) == pytest.approx(1.0, abs=1e-2)
    # The query instruction prompt must actually change the embedding.
    assert query_vec != doc_vec
    max_delta = max(abs(a - b) for a, b in zip(query_vec, doc_vec))
    assert max_delta > 1e-4


def test_truncate_dim_yields_truncated_normalized_vectors():
    provider = HuggingFaceEmbeddingProvider(truncate_dim=512)

    assert provider.dim == 512
    vec = provider.embed(["Matryoshka representation learning."])[0]
    assert len(vec) == 512
    assert _l2_norm(vec) == pytest.approx(1.0, abs=1e-2)
