import threading

from src.indices.vector import VectorIndex


class TestVectorIndexLazyLoading:
    def test_init_without_model_defers_loading(self):
        """
        VectorIndex initialized without an embedding model should not load the
        model until required. This enables fast cold starts.
        """
        index = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
        assert index._model_loaded is False
        assert index._embedding_model is None

    def test_warm_up_with_injected_model(self, shared_embedding_model):
        """
        When an embedding model is injected, it should be immediately available.
        """
        index = VectorIndex(embedding_model=shared_embedding_model)
        assert index._model_loaded is True
        index.warm_up()
        assert index._model_loaded is True

    def test_search_on_empty_index_returns_empty(self):
        """
        Searching an empty index should return empty results without requiring
        model initialization.
        """
        index = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
        results = index.search("test query")
        assert results == []

    def test_model_loaded_flag_with_injected_model(self, shared_embedding_model):
        """
        Injecting an embedding model should set the model_loaded flag immediately.
        """
        index = VectorIndex(embedding_model=shared_embedding_model)
        assert index._model_loaded is True
        assert index._embedding_model is shared_embedding_model

    def test_ensure_model_loaded_thread_safety(self, shared_embedding_model):
        """
        Multiple threads calling _ensure_model_loaded should not cause issues.
        """
        index = VectorIndex(embedding_model=shared_embedding_model)

        call_count = 0
        original_ensure = index._ensure_model_loaded

        def counting_ensure() -> None:
            nonlocal call_count
            call_count += 1
            original_ensure()

        index._ensure_model_loaded = counting_ensure  # type: ignore[method-assign]

        threads = [threading.Thread(target=index._ensure_model_loaded) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 10


class TestVectorIndexProtocol:
    def test_add_document(self, shared_embedding_model):
        index = VectorIndex(embedding_model=shared_embedding_model)
        index.add_document("doc1", "Test content", {"key": "value"})
        assert len(index) == 1

    def test_remove_document(self, shared_embedding_model):
        index = VectorIndex(embedding_model=shared_embedding_model)
        index.add_document("doc1", "Test content", {"key": "value"})
        index.remove_document("doc1")
        assert len(index) == 0

    def test_clear(self, shared_embedding_model):
        index = VectorIndex(embedding_model=shared_embedding_model)
        index.add_document("doc1", "Test content", {})
        index.add_document("doc2", "Test content 2", {})
        assert len(index) == 2
        index.clear()
        assert len(index) == 0

    def test_len(self, shared_embedding_model):
        index = VectorIndex(embedding_model=shared_embedding_model)
        assert len(index) == 0
        index.add_document("doc1", "Test content", {})
        assert len(index) == 1
