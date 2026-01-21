from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.search.calibration import calibrate_results
from src.search.normalization import normalize_result_scores
from src.search.time_scoring import (
    DecayConfig,
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
)
from src.search.variance import compute_dynamic_weights


def rrf_score(rank: int, k: float):
    return 1 / (k + rank)


@dataclass
class ScorePipelineConfig:
    rrf_k: float = 60.0
    strategy_weights: dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.6, "keyword": 0.3, "graph": 0.1
    })

    use_dynamic_weights: bool = False
    variance_threshold: float = 0.1
    min_weight_factor: float = 0.5

    calibration_threshold: float = 0.5
    calibration_steepness: float = 10.0

    time_scoring_mode: str | None = None
    time_scoring_config: TierConfig | DecayConfig | None = None


class ScorePipeline:
    """
    Unified pipeline for score processing.

    Stages:
    1. Fuse: Combine multi-strategy results using RRF → output varies (~0.01-0.1)
    2. Normalize: Scale to [0, 1] range
    3. Calibrate: Apply sigmoid for confidence interpretation
    4. Boost: Apply time-based adjustments (optional)
    """

    def __init__(self, config: ScorePipelineConfig | None = None):
        self.config = config or ScorePipelineConfig()

    def run(
        self,
        strategy_results: dict[str, list[tuple[str, float]]],
        timestamps: dict[str, datetime] | None = None,
    ) -> list[tuple[str, float]]:
        fused = self.fuse(strategy_results)

        normalized = self.normalize(fused)

        calibrated = self.calibrate(normalized)

        if timestamps is not None and self.config.time_scoring_mode is not None:
            return self.boost(calibrated, timestamps)

        return calibrated

    def fuse(
        self, strategy_results: dict[str, list[tuple[str, float]]]
    ) -> list[tuple[str, float]]:
        if not strategy_results:
            return []

        weights = dict(self.config.strategy_weights)

        if self.config.use_dynamic_weights:
            vector_scores = [score for _, score in strategy_results.get("semantic", [])]
            keyword_scores = [score for _, score in strategy_results.get("keyword", [])]

            if vector_scores and keyword_scores:
                base_vector = weights.get("semantic", 1.0)
                base_keyword = weights.get("keyword", 1.0)

                adj_vector, adj_keyword = compute_dynamic_weights(
                    vector_scores,
                    keyword_scores,
                    base_vector,
                    base_keyword,
                    self.config.variance_threshold,
                    self.config.min_weight_factor,
                )
                weights["semantic"] = adj_vector
                weights["keyword"] = adj_keyword

        scores: dict[str, float] = {}

        for strategy, result_list in strategy_results.items():
            weight = weights.get(strategy, 1.0)

            for rank, (doc_id, _original_score) in enumerate(result_list):
                rrf = rrf_score(rank, self.config.rrf_k) * weight
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def normalize(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        if not results:
            return []

        return normalize_result_scores(results)

    def calibrate(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        if not results:
            return []

        return calibrate_results(
            results,
            threshold=self.config.calibration_threshold,
            steepness=self.config.calibration_steepness,
        )

    def boost(
        self,
        results: list[tuple[str, float]],
        timestamps: dict[str, datetime],
    ) -> list[tuple[str, float]]:
        if not results or not timestamps:
            return results

        mode_str = self.config.time_scoring_mode
        if mode_str is None:
            return results

        mode = TimeScoreMode(mode_str)
        config = self.config.time_scoring_config

        if config is None:
            config = TierConfig() if mode == TimeScoreMode.TIERS else DecayConfig()

        reference_time = datetime.now(timezone.utc)

        boosted = []
        for doc_id, score in results:
            timestamp = timestamps.get(doc_id)
            boosted_score = apply_time_boost(
                score, timestamp, mode, config, reference_time=reference_time
            )
            boosted.append((doc_id, boosted_score))

        return sorted(boosted, key=lambda x: x[1], reverse=True)
