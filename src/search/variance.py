def calculate_variance(scores: list[float]):
    if not scores or len(scores) < 2:
        return 0.0

    mean = sum(scores) / len(scores)
    squared_diffs = [(s - mean) ** 2 for s in scores]
    return sum(squared_diffs) / len(scores)


def compute_dynamic_weights(
    vector_scores: list[float],
    keyword_scores: list[float],
    base_vector_weight: float,
    base_keyword_weight: float,
    variance_threshold: float = 0.1,
    min_weight_factor: float = 0.5,
):
    vector_variance = calculate_variance(vector_scores) if vector_scores else 0.0
    keyword_variance = calculate_variance(keyword_scores) if keyword_scores else 0.0

    vector_factor = 1.0
    keyword_factor = 1.0

    if vector_variance < variance_threshold and vector_scores:
        vector_factor = max(min_weight_factor, vector_variance / variance_threshold)

    if keyword_variance < variance_threshold and keyword_scores:
        keyword_factor = max(min_weight_factor, keyword_variance / variance_threshold)

    adjusted_vector = base_vector_weight * vector_factor
    adjusted_keyword = base_keyword_weight * keyword_factor

    total = adjusted_vector + adjusted_keyword
    if total > 0:
        adjusted_vector = adjusted_vector * (base_vector_weight + base_keyword_weight) / total
        adjusted_keyword = adjusted_keyword * (base_vector_weight + base_keyword_weight) / total

    return adjusted_vector, adjusted_keyword
