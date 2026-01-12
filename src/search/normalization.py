def normalize_scores(scores: list[float]):
    if not scores:
        return []

    if len(scores) == 1:
        return [1.0]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def normalize_result_scores(
    results: list[tuple[str, float]]
):
    if not results:
        return []

    if len(results) == 1:
        return [(results[0][0], 1.0)]

    scores = [score for _, score in results]
    normalized = normalize_scores(scores)

    return [(doc_id, norm_score) for (doc_id, _), norm_score in zip(results, normalized)]
