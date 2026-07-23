"""HuggingFace / sentence-transformers EmbeddingProvider adapter.

In-process embedding via ``sentence_transformers.SentenceTransformer``.
Defaults to Qwen3-Embedding-0.6B. No Ollama, no external service.

This is an ADDITIVE port implementation. The live embedding path
(``searchkernel/indices/vector.py``, BAAI/bge-small-en-v1.5) is untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from searchkernel.domain import Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Documented Qwen3-Embedding query instruction, used only if the loaded model
# does not expose a "query" prompt via sentence-transformers.
_QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
)


class HuggingFaceEmbeddingProvider:
    """EmbeddingProvider backed by a sentence-transformers model.

    Qwen3-Embedding is asymmetric: queries take an instruction prompt,
    documents do not. ``embed`` embeds DOCUMENTS (no prompt); ``embed_query``
    applies the query instruction. The EmbeddingProvider port itself has no
    query variant yet -- that asymmetry lives here until W4a lifts it into
    the port contract.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        truncate_dim: int | None = None,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._truncate_dim = truncate_dim
        # Matryoshka (MRL): passing truncate_dim makes the model emit
        # truncated + re-normalized vectors directly.
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, truncate_dim=truncate_dim, device=device
        )
        native_dim = self._model.get_sentence_embedding_dimension()
        self.dim: int = truncate_dim if truncate_dim is not None else int(native_dim)
        # Whether the model ships a named "query" prompt we can reference.
        self._has_query_prompt = "query" in getattr(self._model, "prompts", {})

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed DOCUMENTS (no instruction prompt), L2-normalized."""
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> Vector:
        """Embed a single QUERY with the Qwen3 instruction prompt applied."""
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[Vector]:
        """Embed QUERIES with the Qwen3 instruction prompt applied."""
        if self._has_query_prompt:
            embeddings = self._model.encode(
                texts,
                prompt_name="query",
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            embeddings = self._model.encode(
                texts,
                prompt=_QWEN3_QUERY_INSTRUCTION,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings.tolist()
