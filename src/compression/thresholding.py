from src.models import ChunkResult


def filter_by_score(results: list[ChunkResult], min_score: float = 0.3):
    return [r for r in results if r.score >= min_score]
