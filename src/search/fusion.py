from datetime import datetime, timezone

from src.search.calibration import calibrate_results
from src.search.normalization import normalize_scores as normalize_float_scores
from src.search.time_scoring import (
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
)
from src.search.variance import compute_dynamic_weights


def rrf_score(rank: int, k: int):
    return 1 / (k + rank)


def apply_recency_boost(
    doc_id: str, score: float, modified_times: dict[str, float], tiers: list[tuple[int, float]]
):
    if doc_id not in modified_times:
        return score

    modified_time = modified_times[doc_id]
    timestamp = datetime.fromtimestamp(modified_time, timezone.utc)

    config = TierConfig()
    if len(tiers) >= 1:
        config.recent_days = tiers[0][0]
        config.recent_boost = tiers[0][1]
    if len(tiers) >= 2:
        config.moderate_days = tiers[1][0]
        config.moderate_boost = tiers[1][1]

    return apply_time_boost(score, timestamp, TimeScoreMode.TIERS, config)


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


def fuse_results_v2(
    results: dict[str, list[tuple[str, float]]],
    k: int,
    base_weights: dict[str, float],
    modified_times: dict[str, float],
    use_dynamic_weights: bool = True,
    variance_threshold: float = 0.1,
    min_weight_factor: float = 0.5,
):
    weights = dict(base_weights)

    if use_dynamic_weights:
        vector_scores = [score for _, score in results.get("semantic", [])]
        keyword_scores = [score for _, score in results.get("keyword", [])]

        if vector_scores and keyword_scores:
            base_vector = base_weights.get("semantic", 1.0)
            base_keyword = base_weights.get("keyword", 1.0)

            adj_vector, adj_keyword = compute_dynamic_weights(
                vector_scores,
                keyword_scores,
                base_vector,
                base_keyword,
                variance_threshold,
                min_weight_factor,
            )
            weights["semantic"] = adj_vector
            weights["keyword"] = adj_keyword

    scores: dict[str, float] = {}

    for strategy, result_list in results.items():
        weight = weights.get(strategy, 1.0)
        strategy_scores = [score for _, score in result_list]
        normalized = normalize_float_scores(strategy_scores) if strategy_scores else []

        for i, (doc_id, _) in enumerate(result_list):
            rrf = rrf_score(i, k) * weight
            norm_score = normalized[i] if i < len(normalized) else 0.0
            combined = rrf + (norm_score * weight * 0.5)
            scores[doc_id] = scores.get(doc_id, 0.0) + combined

    tiers = [(7, 1.2), (30, 1.1)]
    boosted_scores = [
        (doc_id, apply_recency_boost(doc_id, score, modified_times, tiers))
        for doc_id, score in scores.items()
    ]

    return sorted(boosted_scores, key=lambda x: x[1], reverse=True)


def normalize_final_scores(
    fused_results: list[tuple[str, float]],
    threshold: float = 0.035,
    steepness: float = 150.0
) -> list[tuple[str, float]]:
    """
    Normalize fused scores to [0,1] range and apply sigmoid calibration.

    The fused scores from fuse_results() are typically ~0.01-0.08, but scores
    from fuse_results_v2() can be much higher (~0.5-2.0) due to added normalized
    strategy scores. We first min-max normalize to a consistent range, then
    apply sigmoid calibration to convert to confidence scores.
    """
    if not fused_results:
        return []

    if len(fused_results) == 1:
        # Single result: apply calibration directly (relative to 0)
        return calibrate_results(fused_results, threshold, steepness)

    # Min-max normalize to bring scores into expected calibration range
    scores = [score for _, score in fused_results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All identical scores - calibrate at the midpoint
        return [(doc_id, 0.5) for doc_id, _ in fused_results]

    # Scale scores to [0, 0.08] range (typical raw RRF range)
    # This maps max -> 0.08 (well above calibration threshold)
    # and min -> 0.0 (well below threshold)
    target_max = 0.08
    normalized = [
        (doc_id, (score - min_score) / (max_score - min_score) * target_max)
        for doc_id, score in fused_results
    ]

    return calibrate_results(normalized, threshold, steepness)
