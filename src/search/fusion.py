from datetime import datetime, timezone

from src.search.time_scoring import (
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
)


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
