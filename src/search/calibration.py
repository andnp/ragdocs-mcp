import math


def calibrate_score(
    rrf_score: float,
    threshold: float = 0.04,
    steepness: float = 150.0
):
    """Convert RRF score to calibrated confidence [0,1]."""
    exponent = -steepness * (rrf_score - threshold)
    exponent = max(-20, min(20, exponent))  # Prevent overflow
    calibrated = 1.0 / (1.0 + math.exp(exponent))
    return max(0.0, min(1.0, calibrated))


def calibrate_results(
    fused_results: list[tuple[str, float]],
    threshold: float = 0.04,
    steepness: float = 150.0
):
    """Calibrate all RRF scores in result list."""
    if not fused_results:
        return []

    return [
        (doc_id, calibrate_score(score, threshold, steepness))
        for doc_id, score in fused_results
    ]
