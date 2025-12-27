def filter_by_confidence(
    results: list[tuple[str, float]],
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    if threshold <= 0.0:
        return results
    return [(chunk_id, score) for chunk_id, score in results if score >= threshold]


def limit_per_document(
    results: list[tuple[str, float]],
    max_per_doc: int = 0,
) -> list[tuple[str, float]]:
    if max_per_doc <= 0:
        return results

    doc_counts: dict[str, int] = {}
    limited: list[tuple[str, float]] = []

    for chunk_id, score in results:
        doc_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id
        current_count = doc_counts.get(doc_id, 0)
        if current_count < max_per_doc:
            limited.append((chunk_id, score))
            doc_counts[doc_id] = current_count + 1

    return limited
