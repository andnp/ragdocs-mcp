from pathlib import Path


def test_utf8_bom_and_unicode_documents_index(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    bom_path = docs_dir / "bom.md"
    bom_path.write_bytes(b"\xef\xbb\xbf# BOM\n\nContent")
    unicode_path = docs_dir / "unicode.md"
    unicode_path.write_text("# Unicode 🚀\n\n中文 日本語")

    assert record_manager.index_document(str(bom_path)) is True
    assert record_manager.index_document(str(unicode_path)) is True
    assert record_manager.get_document_count() == 2
    assert record_manager.get_failed_files() == []


def test_invalid_encoding_uses_parser_fallback(record_manager) -> None:
    docs_dir = Path(record_manager._config.indexing.documents_path)
    path = docs_dir / "broken.md"
    path.write_bytes(b"# Broken\n\xff\xfe")

    assert record_manager.index_document(str(path)) is True
    assert record_manager.get_failed_files() == []

    path.write_text("# Repaired\n\nValid UTF-8")
    assert record_manager.index_document(str(path)) is True
    assert record_manager.get_document_count() == 1
