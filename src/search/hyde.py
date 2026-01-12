import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class VectorSearchable(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[dict]:
        ...

    def get_text_embedding(self, text: str) -> list[float]:
        ...


def search_with_hypothesis(
    vector_index: VectorSearchable,
    hypothesis: str,
    top_k: int = 10,
    excluded_files: set[str] | None = None,
    docs_root: Path | None = None,
):
    logger.info(f"HyDE search with hypothesis: {hypothesis[:100]}...")
    results = vector_index.search(hypothesis, top_k, excluded_files, docs_root)
    logger.info(f"HyDE search returned {len(results)} results")
    return results
