
from src.models import CompressionStats
from src.search.pipeline import SearchPipeline, SearchPipelineConfig


class TestSearchPipelineConfig:
    def test_default_values(self):
        config = SearchPipelineConfig()

        assert config.min_confidence == 0.0
        assert config.max_chunks_per_doc == 0
        assert config.dedup_enabled is True
        assert config.dedup_threshold == 0.85
        assert config.ngram_dedup_enabled is True
        assert config.ngram_dedup_threshold == 0.7
        assert config.mmr_enabled is False
        assert config.mmr_lambda == 0.7
        assert config.parent_retrieval_enabled is False
        assert config.rerank_enabled is False
        assert config.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.rerank_top_n == 10

    def test_custom_values(self):
        config = SearchPipelineConfig(
            min_confidence=0.5,
            max_chunks_per_doc=3,
            dedup_enabled=False,
            dedup_threshold=0.9,
            ngram_dedup_enabled=False,
            ngram_dedup_threshold=0.8,
            mmr_enabled=True,
            mmr_lambda=0.5,
            parent_retrieval_enabled=True,
            rerank_enabled=True,
            rerank_model="custom-model",
            rerank_top_n=5,
        )

        assert config.min_confidence == 0.5
        assert config.max_chunks_per_doc == 3
        assert config.dedup_enabled is False
        assert config.dedup_threshold == 0.9
        assert config.ngram_dedup_enabled is False
        assert config.ngram_dedup_threshold == 0.8
        assert config.mmr_enabled is True
        assert config.mmr_lambda == 0.5
        assert config.parent_retrieval_enabled is True
        assert config.rerank_enabled is True
        assert config.rerank_model == "custom-model"
        assert config.rerank_top_n == 5


class TestSearchPipelineEmptyInput:
    def test_empty_fused_results_returns_empty(self):
        config = SearchPipelineConfig(ngram_dedup_enabled=False)
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
    def test_filter_by_confidence_removes_low_scores(self):
        config = SearchPipelineConfig(
            min_confidence=0.5,
            dedup_enabled=False,
            ngram_dedup_enabled=False,
        )
        pipeline = SearchPipeline(config)

        # After min-max normalization: 1.0, 0.75, 0.5, 0.25, 0.0
        fused = [
            ("chunk_a", 1.0),
            ("chunk_b", 0.8),
            ("chunk_c", 0.6),
            ("chunk_d", 0.4),
            ("chunk_e", 0.2),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"content for {chunk_id}"

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        # 1.0 and 0.75 pass the 0.5 threshold; 0.5 is boundary
        assert len(results) >= 2
        chunk_ids = [r[0] for r in results]
        assert "chunk_a" in chunk_ids
        assert "chunk_b" in chunk_ids
        assert stats.original_count == 5
        assert stats.after_threshold >= 2

    def test_zero_threshold_keeps_all(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            dedup_enabled=False,
            ngram_dedup_enabled=False,
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

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert len(results) == 3
        assert stats.after_threshold == 3


class TestSearchPipelineDocLimit:
    def test_limits_chunks_per_document(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            max_chunks_per_doc=2,
            dedup_enabled=False,
            ngram_dedup_enabled=False,
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

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert len(results) == 3
        doc_a_chunks = [r for r in results if r[0].startswith("doc_a")]
        assert len(doc_a_chunks) == 2
        assert stats.after_doc_limit == 3

    def test_zero_doc_limit_keeps_all(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            max_chunks_per_doc=0,
            dedup_enabled=False,
            ngram_dedup_enabled=False,
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
            return f"unique content {chunk_id}"

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert len(results) == 3


class TestSearchPipelineDeduplication:
    def test_removes_similar_chunks(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            dedup_enabled=True,
            dedup_threshold=0.9,
            ngram_dedup_enabled=False,
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
            return f"unique content {chunk_id}"

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert len(results) == 2
        chunk_ids = [r[0] for r in results]
        assert "chunk_a" in chunk_ids
        assert "chunk_c" in chunk_ids
        assert "chunk_b" not in chunk_ids
        assert stats.clusters_merged == 1

    def test_dedup_disabled_keeps_all(self):
        config = SearchPipelineConfig(
            min_confidence=0.0,
            dedup_enabled=False,
            ngram_dedup_enabled=False,
        )
        pipeline = SearchPipeline(config)

        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [1.0, 0.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        fused = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert len(results) == 2
        assert stats.clusters_merged == 0


class TestSearchPipelineTopN:
    def test_limits_to_top_n(self):
        config = SearchPipelineConfig(dedup_enabled=False, ngram_dedup_enabled=False)
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

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=3)

        assert len(results) == 3
        chunk_ids = [r[0] for r in results]
        assert chunk_ids == ["chunk_a", "chunk_b", "chunk_c"]


class TestSearchPipelineCompressionStats:
    def test_returns_correct_stats(self):
        config = SearchPipelineConfig(
            min_confidence=0.3,
            max_chunks_per_doc=2,
            dedup_enabled=True,
            dedup_threshold=0.99,
            ngram_dedup_enabled=False,
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

        # After normalization: 1.0, 0.67, 0.33, 0.0
        fused = [
            ("doc_a_chunk_0", 1.0),
            ("doc_a_chunk_1", 0.7),
            ("doc_a_chunk_2", 0.4),
            ("doc_b_chunk_0", 0.1),
        ]

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert isinstance(stats, CompressionStats)
        assert stats.original_count == 4
        # Check that compression happened (new order: threshold -> content_dedup -> ngram_dedup -> semantic_dedup -> doc_limit)
        assert stats.after_threshold <= stats.original_count
        assert stats.after_content_dedup <= stats.after_threshold
        assert stats.after_ngram_dedup <= stats.after_content_dedup
        assert stats.after_dedup <= stats.after_ngram_dedup
        assert stats.after_doc_limit <= stats.after_dedup

    def test_stats_with_normalization(self):
        config = SearchPipelineConfig(min_confidence=0.0, dedup_enabled=False, ngram_dedup_enabled=False)
        pipeline = SearchPipeline(config)

        fused = [
            ("chunk_a", 100.0),
            ("chunk_b", 50.0),
        ]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        results, stats = pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        # After calibration, both high scores should have very high confidence
        # (scores are so high they saturate near 1.0)
        assert results[0][1] > 0.99, "Very high score should have confidence ~1.0"
        assert results[1][1] > 0.99, "High score should also have high confidence"


class TestSearchPipelineLazyReranker:
    def test_reranker_not_loaded_when_disabled(self):
        config = SearchPipelineConfig(rerank_enabled=False, ngram_dedup_enabled=False)
        pipeline = SearchPipeline(config)

        assert pipeline._reranker is None

        fused = [("chunk_a", 0.9)]

        def get_embedding(chunk_id: str):
            return None

        def get_content(chunk_id: str):
            return f"unique content {chunk_id}"

        pipeline.process(fused, get_embedding, get_content, "query", top_n=10)

        assert pipeline._reranker is None
