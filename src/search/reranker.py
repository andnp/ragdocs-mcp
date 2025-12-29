from typing import Protocol


class ContentProvider(Protocol):
    def __call__(self, chunk_id: str, /) -> str | None: ...


class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        content_provider: ContentProvider,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        self._ensure_model_loaded()
        assert self._model is not None

        pairs: list[tuple[str, str, str]] = []
        for chunk_id, _ in candidates:
            content = content_provider(chunk_id)
            if content:
                pairs.append((chunk_id, query, content))

        if not pairs:
            return candidates[:top_n]

        query_content_pairs = [(query, content) for _, _, content in pairs]
        scores = self._model.predict(query_content_pairs)

        chunk_scores = [
            (chunk_id, float(score))
            for (chunk_id, _, _), score in zip(pairs, scores, strict=False)
        ]

        missing_ids = {cid for cid, _ in candidates} - {cid for cid, _, _ in pairs}
        for chunk_id, original_score in candidates:
            if chunk_id in missing_ids:
                chunk_scores.append((chunk_id, original_score * 0.5))

        reranked = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return reranked[:top_n]
