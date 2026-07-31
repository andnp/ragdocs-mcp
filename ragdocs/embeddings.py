from __future__ import annotations

import hashlib
import os
import re

import numpy as np
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

TEST_FAKE_EMBEDDINGS_ENV_VAR = "MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS"
TEST_FAKE_EMBEDDING_MODEL_NAME = "__deterministic_fake__"


def should_use_test_fake_embeddings() -> bool:
    return os.getenv(TEST_FAKE_EMBEDDINGS_ENV_VAR) == "1"


class DeterministicFakeEmbeddingModel(MockEmbedding):
    """Deterministic, network-free embedding model for tests and test-mode CLIs."""

    def __init__(self, dimension: int = 384):
        super().__init__(embed_dim=dimension, model_name="deterministic-fake")

    def _vector_for_text(self, text: str) -> list[float]:
        vector = np.zeros(self.embed_dim, dtype=np.float32)
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()) or ["__empty__"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            primary = int.from_bytes(digest[:8], "big") % self.embed_dim
            secondary = int.from_bytes(digest[8:16], "big") % self.embed_dim
            sign = 1.0 if digest[16] % 2 == 0 else -1.0

            vector[primary] += 1.0
            vector[secondary] += 0.5 * sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector_for_text(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector_for_text(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vector_for_text(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector_for_text(query)
