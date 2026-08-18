"""Pre-download HuggingFace models required by the `real_embeddings`-marked test suite.

Run this once locally (or as a CI cache-warm step) before running tests offline:

    uv run python scripts/download_test_models.py

After this completes, the test suite runs fully offline (HF_HUB_OFFLINE=1) and
never touches the network, avoiding HuggingFace Hub rate limiting under
parallel test execution.

This warms the models through the same adapter classes production code uses
(rather than a raw huggingface_hub.snapshot_download) so the cache lands in
whatever directory each adapter actually reads from - llama_index's
HuggingFaceEmbedding caches to LLAMA_INDEX_CACHE_DIR (not HF_HOME), which a
generic snapshot_download would miss.
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

from searchkernel.adapters.rerank import HuggingFaceReranker


def main() -> None:
    print("Downloading BAAI/bge-small-en-v1.5 (embedding model)...")
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embedding_model.get_text_embedding("warmup")

    print("Downloading sentence-transformers/all-MiniLM-L6-v2 (legacy test embedding model)...")
    legacy_embedding_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    legacy_embedding_model.get_text_embedding("warmup")

    print("Downloading Qwen/Qwen3-Reranker-0.6B (reranker adapter model)...")
    reranker = HuggingFaceReranker()
    reranker.rerank("warmup", ["warmup document"])

    print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 (application reranker)...")
    pipeline_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pipeline_reranker.predict([("warmup", "warmup document content")])

    print("All test models cached.")


if __name__ == "__main__":
    main()
