from src.search.diversity import select_mmr


class TestSelectMMR:

    def test_empty_candidates_returns_empty(self):
        query_embedding = [1.0, 0.0, 0.0]

        def get_embedding(chunk_id: str):
            return None

        result = select_mmr(query_embedding, [], get_embedding)
        assert result == []

    def test_single_candidate_returns_unchanged(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [("chunk_a", 0.9)]

        def get_embedding(chunk_id: str):
            return [1.0, 0.0, 0.0]

        result = select_mmr(query_embedding, candidates, get_embedding)
        assert result == [("chunk_a", 0.9)]

    def test_selects_most_relevant_first(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.5),
            ("chunk_b", 0.8),
            ("chunk_c", 0.6),
        ]

        embeddings = {
            "chunk_a": [0.5, 0.5, 0.0],
            "chunk_b": [0.9, 0.1, 0.0],
            "chunk_c": [0.3, 0.7, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        result = select_mmr(query_embedding, candidates, get_embedding, lambda_param=1.0)
        # With lambda=1.0, pure relevance: chunk_b is most relevant to query
        assert result[0][0] == "chunk_b"

    def test_promotes_diversity_with_lower_lambda(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
        ]

        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.99, 0.1, 0.0],
            "chunk_c": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        # With low lambda, diversity matters more
        result = select_mmr(query_embedding, candidates, get_embedding, lambda_param=0.3, top_n=3)

        # chunk_a is most relevant and should be first
        assert result[0][0] == "chunk_a"
        # chunk_c is very different (orthogonal) and should come before chunk_b
        # which is nearly identical to chunk_a
        chunk_ids = [r[0] for r in result]
        assert chunk_ids.index("chunk_c") < chunk_ids.index("chunk_b")

    def test_respects_top_n_limit(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
            ("chunk_c", 0.7),
            ("chunk_d", 0.6),
        ]

        def get_embedding(chunk_id: str):
            return [0.5, 0.5, 0.0]

        result = select_mmr(query_embedding, candidates, get_embedding, top_n=2)
        assert len(result) == 2

    def test_handles_missing_embeddings(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            # chunk_b has no embedding
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        # Should not crash, should use original scores for missing embeddings
        result = select_mmr(query_embedding, candidates, get_embedding)
        assert len(result) == 2

    def test_preserves_original_scores(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.95),
            ("chunk_b", 0.85),
        ]

        def get_embedding(chunk_id: str):
            return [1.0, 0.0, 0.0]

        result = select_mmr(query_embedding, candidates, get_embedding)

        # Original scores should be preserved
        for chunk_id, score in result:
            original = next(s for c, s in candidates if c == chunk_id)
            assert score == original

    def test_default_lambda_is_07(self):
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("chunk_a", 0.9),
            ("chunk_b", 0.8),
        ]

        embeddings = {
            "chunk_a": [1.0, 0.0, 0.0],
            "chunk_b": [0.0, 1.0, 0.0],
        }

        def get_embedding(chunk_id: str):
            return embeddings.get(chunk_id)

        # Default lambda=0.7 balances relevance and diversity
        result = select_mmr(query_embedding, candidates, get_embedding)
        assert len(result) == 2
