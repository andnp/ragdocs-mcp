from src.models import CompressionStats
from src.search.pipeline import SearchPipeline, SearchPipelineConfig

# Diverse content to survive n-gram dedup (default threshold 0.7)
_DIVERSE_CONTENT = {
    "chunk_a": "Python is a versatile programming language used for web development and data science",
    "chunk_b": "Machine learning algorithms require large datasets for effective model training",
    "chunk_c": "Database indexing improves query performance significantly in production systems",
    "chunk_d": "Cloud computing enables scalable infrastructure deployment on demand worldwide",
    "chunk_e": "Network security protocols protect data transmission from unauthorized access",
    "doc_a_chunk_0": "Kubernetes orchestrates containerized applications across distributed clusters efficiently",
    "doc_a_chunk_1": "React components manage user interface state through composable reusable patterns",
    "doc_a_chunk_2": "PostgreSQL provides advanced relational database features including JSON support",
    "doc_b_chunk_0": "Redis caching layer reduces latency for frequently accessed data significantly",
}


def _diverse_content(chunk_id: str) -> str:
    return _DIVERSE_CONTENT.get(chunk_id, f"Unique document content about {chunk_id}")


class TestSearchPipelineConfig:
    def test_default_values(self):
        config = SearchPipelineConfig()

        assert config.min_confidence == 0.0
        assert config.max_chunks_per_doc == 0
        assert config.dedup_threshold == 0.85
        assert config.reranking_enabled is True
        assert config.rerank_top_n == 10

    def test_custom_values(self):
        config = SearchPipelineConfig(
            min_confidence=0.5,
            max_chunks_per_doc=3,
            dedup_threshold=0.9,
            reranking_enabled=False,
            rerank_top_n=5,
        )

        assert config.min_confidence == 0.5
        assert config.max_chunks_per_doc == 3
        assert config.dedup_threshold == 0.9
        assert config.reranking_enabled is False
        assert config.rerank_top_n == 5


class TestSearchPipelineEmptyInput:
    def test_empty_fused_results_returns_empty(self):
        config = SearchPipelineConfig()
        pipeline = SearchPipeline(config)

        def get_embedding(chunk_id: str):
            return [0.1, 0.2, 0.3]

        def get_content(chunk_id: str):
            return f"content for {chunk_id}"

        results, stats = pipeline.process([], get_embedding, get_content, "query")

        assert results == []
        assert stats.original_count == 0
        assert stats.after_threshold == 0
        assert stats.after_content_dedup == 0
        assert stats.after_ngram_dedup == 0
        assert stats.after_dedup == 0
        assert stats.after_doc_limit == 0
        assert stats.clusters_merged == 0


class TestSearchPipelineThresholdFilter:
    def test_content_is_fetched_lazily_after_threshold_pruning(self):
        config = SearchPipelineConfig(
            min_confidence=0.75,
            reranking_enabled=False,
        )
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 0.95),
            ("chunk_b", 0.85),
            ("chunk_c", 0.70),
            ("chunk_d", 0.40),
        ]
        content_calls: list[str] = []

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            content_calls.append(chunk_id)
            return _diverse_content(chunk_id)

        results, stats = pipeline.process(
            fused,
            get_embedding,
            get_content,
            "query",
            top_n=10,
        )

        assert len(results) == 2
        assert stats.after_threshold == 2
        assert sorted(content_calls) == ["chunk_a", "chunk_b"]

    def test_filter_by_confidence_removes_low_scores(self):
        config = SearchPipelineConfig(
            min_confidence=0.5,
        )
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 1.0),
            ("chunk_b", 0.8),
            ("chunk_c", 0.6),
            ("chunk_d", 0.4),
            ("chunk_e", 0.2),
        ]

        def get_embedding(chunk_id: str):
            return None

        # Use diverse content to avoid dedup removing test data
        content_map = {
            "chunk_a": "Python is a versatile programming language used for web development",
            "chunk_b": "Machine learning algorithms require large datasets for training models",
            "chunk_c": "Database indexing improves query performance significantly in production",
            "chunk_d": "Cloud computing enables scalable infrastructure deployment on demand",
            "chunk_e": "Network security protocols protect data transmission from unauthorized access",
        }

        def get_content(chunk_id: str):
            return content_map.get(chunk_id, "")

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        assert len(results) >= 2
        chunk_ids = [r[0] for r in results]
        assert "chunk_a" in chunk_ids
        assert "chunk_b" in chunk_ids
        assert stats.original_count == 5
        assert stats.after_threshold >= 2

    def test_zero_threshold_keeps_all(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
        )
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.1),
            ("chunk_c", 0.0),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        # All pass threshold=0.0
        assert stats.after_threshold == 3


class TestSearchPipelineDocLimit:
    def test_limits_chunks_per_document(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            max_chunks_per_doc=2,
        )
        pipeline = SearchPipeline(config)

        fused = [
            ("doc_a_chunk_0", 0.9),
            ("doc_a_chunk_1", 0.8),
            ("doc_a_chunk_2", 0.7),
            ("doc_b_chunk_0", 0.6),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        doc_a_chunks = [r for r in results if r[0].startswith("doc_a")]
        assert len(doc_a_chunks) <= 2
        assert stats.after_doc_limit <= stats.after_dedup

    def test_zero_doc_limit_keeps_all(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            max_chunks_per_doc=0,
            reranking_enabled=False,
        )
        pipeline = SearchPipeline(config)

        fused = [
            ("doc_a_chunk_0", 0.9),
            ("doc_a_chunk_1", 0.8),
            ("doc_a_chunk_2", 0.7),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return _diverse_content(chunk_id)

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        assert len(results) == 3


class TestSearchPipelineDeduplication:
    def test_embeddings_are_not_fetched_when_content_dedup_leaves_single_candidate(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            reranking_enabled=False,
        )
        pipeline = SearchPipeline(config)

        embedding_calls: list[str] = []

        def get_embedding(chunk_id: str):
            embedding_calls.append(chunk_id)
            return [1.0, 0.0, 0.0]

        def get_content(chunk_id: str):
            return "same content for dedup"

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        results, stats = pipeline.process(
            fused,
            get_embedding,
            get_content,
            "query",
            top_n=10,
        )

        assert results == [("chunk_a", 0.9)]
        assert stats.after_content_dedup == 1
        assert stats.after_dedup == 1
        assert embedding_calls == []

    def test_removes_similar_chunks(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            dedup_threshold=0.9,
            reranking_enabled=False,
        )
        pipeline = SearchPipeline(config)

        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],
            "chunk_c": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        def get_content(chunk_id: str):
            return _diverse_content(chunk_id)

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        assert len(results) == 2
        chunk_ids = [r[0] for r in results]
        assert "chunk_a" in chunk_ids
        assert "chunk_c" in chunk_ids
        assert "chunk_b" not in chunk_ids
        assert stats.clusters_merged == 1


class TestSearchPipelineTopN:
    def test_limits_to_top_n(self):
        config = SearchPipelineConfig()
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
            ("chunk_d", 0.6),
            ("chunk_e", 0.5),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=3
        )

        assert len(results) <= 3
        chunk_ids = [r[0] for r in results]
        assert "chunk_a" in chunk_ids


class TestSearchPipelineCompressionStats:
    def test_returns_correct_stats(self):
        config = SearchPipelineConfig(
            min_confidence=0.3,
            max_chunks_per_doc=2,
            dedup_threshold=0.99,
        )
        pipeline = SearchPipeline(config)

        embeddings = {
            "doc_a_chunk_0": [1.0, 0.0, 0.0],
            "doc_a_chunk_1": [1.0, 0.0, 0.0],
            "doc_a_chunk_2": [0.9, 0.1, 0.0],
            "doc_b_chunk_0": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        fused = [
            ("doc_a_chunk_0", 1.0),
            ("doc_a_chunk_1", 0.7),
            ("doc_a_chunk_2", 0.4),
            ("doc_b_chunk_0", 0.1),
        ]

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        assert isinstance(stats, CompressionStats)
        assert stats.original_count == 4
        assert stats.after_threshold <= stats.original_count
        assert stats.after_content_dedup <= stats.after_threshold
        assert stats.after_ngram_dedup <= stats.after_content_dedup
        assert stats.after_dedup <= stats.after_ngram_dedup
        assert stats.after_doc_limit <= stats.after_dedup

    def test_stats_with_normalization(self):
        """Test that very high fusion scores get clamped to [0, 1] range.

        After reranking, scores are cross-encoder outputs (already in [0, 1]).
        The clamp ensures no score exceeds 1.0 regardless of source.
        """
        config = SearchPipelineConfig(min_confidence=0.0, reranking_enabled=False)
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 100.0),
            ("chunk_b", 50.0),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return _diverse_content(chunk_id)

        results, stats = pipeline.process(
            fused, get_embedding, get_content, "query", top_n=10
        )

        assert len(results) == 2, "Both results should survive pipeline"
        # Scores are clamped to [0, 1] range
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1] range"
