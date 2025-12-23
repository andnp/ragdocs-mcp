from datetime import datetime, timezone


def rrf_score(rank: int, k: int):
    return 1 / (k + rank)


def apply_recency_boost(
    doc_id: str, score: float, modified_times: dict[str, float], tiers: list[tuple[int, float]]
):
    if doc_id not in modified_times:
        return score

    modified_time = modified_times[doc_id]
    now = datetime.now(timezone.utc).timestamp()
    age_days = (now - modified_time) / 86400

    for days, multiplier in sorted(tiers, key=lambda x: x[0]):
        if age_days <= days:
            return score * multiplier

    return score


def fuse_results(
    results: dict[str, list[str]],
    k: int,
    weights: dict[str, float],
    modified_times: dict[str, float],
):
    scores: dict[str, float] = {}

    for strategy, doc_ids in results.items():
        weight = weights.get(strategy, 1.0)
        for rank, doc_id in enumerate(doc_ids):
            score = rrf_score(rank, k) * weight
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    tiers = [(7, 1.2), (30, 1.1)]
    boosted_scores = [
        (doc_id, apply_recency_boost(doc_id, score, modified_times, tiers))
        for doc_id, score in scores.items()
    ]

    return sorted(boosted_scores, key=lambda x: x[1], reverse=True)
