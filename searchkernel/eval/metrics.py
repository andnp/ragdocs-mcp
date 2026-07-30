"""Pure evaluation metrics for ranked retrieval quality.

Exact implementations of recall@k, nDCG@k, MRR (Mean Reciprocal Rank), etc.
These are pure functions with no I/O or dependencies on indices/models.
"""

import math
from collections.abc import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str], k: int) -> float:
    """Compute recall@k: fraction of relevant items in top-k results.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).
        k: Cutoff rank.

    Returns:
        Recall@k in [0, 1]. Returns 0 if no relevant items exist.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int,
    gains: dict[str, float] | None = None,
) -> float:
    """Compute nDCG@k: normalized discounted cumulative gain.

    Assumes binary relevance by default (relevant=1, non-relevant=0).
    Optionally accepts custom gains for graded relevance.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).
        k: Cutoff rank.
        gains: Optional dict mapping result IDs to relevance scores.
               If None, binary relevance (1 for relevant, 0 for non-relevant).

    Returns:
        nDCG@k in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Compute DCG@k
    dcg = 0.0
    for i, result_id in enumerate(ranked_ids[:k]):
        rank = i + 1  # 1-based ranking
        if result_id in relevant_set:
            gain = gains.get(result_id, 1.0) if gains else 1.0
            dcg += gain / math.log2(rank + 1)

    # Compute ideal DCG (IDCG): order relevant items at top
    idcg = 0.0
    for i in range(min(k, len(relevant_ids))):
        rank = i + 1
        if gains:
            # Use highest gains for ideal ranking
            sorted_gains = sorted(gains.values(), reverse=True)
            if i < len(sorted_gains):
                gain = sorted_gains[i]
            else:
                gain = 1.0
        else:
            gain = 1.0
        idcg += gain / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str]) -> float:
    """Compute MRR (Mean Reciprocal Rank): 1 / rank of first relevant item.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).

    Returns:
        MRR in [0, 1]. Returns 0 if no relevant item is found.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for i, result_id in enumerate(ranked_ids):
        if result_id in relevant_set:
            return 1.0 / (i + 1)  # 1-based ranking

    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str]) -> float:
    """Compute Average Precision (AP): mean of precision@k at each relevant hit.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).

    Returns:
        AP in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    num_relevant = len(relevant_set)

    ap = 0.0
    num_hits = 0
    for i, result_id in enumerate(ranked_ids):
        if result_id in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            ap += precision_at_i

    return ap / num_relevant
