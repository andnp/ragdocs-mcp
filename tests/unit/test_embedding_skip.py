from pathlib import Path

from tests.conftest import create_test_document


def test_unchanged_record_content_uses_embedding_cache(
    record_manager, monkeypatch
) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    doc_path = create_test_document(
        docs_dir,
        "guide",
        "# Guide\n\nStable content for the embedding cache.",
    )
    calls: list[list[str]] = []
    provider = record_manager.embedding_provider
    original_embed = provider.embed

    def track_embed(texts: list[str]):
        calls.append(texts)
        return original_embed(texts)

    monkeypatch.setattr(provider, "embed", track_embed)
    assert record_manager.index_document(doc_path) is True
    first_call_count = len(calls)
    assert first_call_count > 0

    monkeypatch.setattr(
        record_manager,
        "prepare_document",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("unchanged documents should not be reparsed")
        ),
    )
    assert record_manager.index_document(doc_path) is True
    assert len(calls) == first_call_count
