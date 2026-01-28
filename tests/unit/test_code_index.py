import tempfile
from pathlib import Path

from src.indices.code import CodeIndex
from src.models import CodeBlock


class TestCodeIndex:
    def test_add_and_search_code_block(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="def getUserById(user_id):\n    return db.query(user_id)",
            language="python",
        )
        index.add_code_block(code_block)

        results = index.search("getUserById", top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc1_chunk_0"
        assert results[0]["doc_id"] == "doc1"

    def test_search_camel_case_split(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="function calculateTotalAmount() { return sum; }",
            language="javascript",
        )
        index.add_code_block(code_block)

        results = index.search("total", top_k=5)
        assert len(results) == 1

        results = index.search("amount", top_k=5)
        assert len(results) == 1

    def test_search_snake_case_split(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="def get_user_by_id(user_id):\n    pass",
            language="python",
        )
        index.add_code_block(code_block)

        results = index.search("user", top_k=5)
        assert len(results) == 1

        results = index.search("get user", top_k=5)
        assert len(results) == 1

    def test_remove_by_doc_id(self):
        index = CodeIndex()

        for i in range(3):
            code_block = CodeBlock(
                id=f"doc1_code_{i}",
                doc_id="doc1",
                chunk_id=f"doc1_chunk_{i}",
                content=f"function test{i}() {{ }}",
                language="javascript",
            )
            index.add_code_block(code_block)

        code_block2 = CodeBlock(
            id="doc2_code_0",
            doc_id="doc2",
            chunk_id="doc2_chunk_0",
            content="function other() { }",
            language="javascript",
        )
        index.add_code_block(code_block2)

        assert len(index) == 4

        index.remove_by_doc_id("doc1")

        assert len(index) == 1
        results = index.search("test", top_k=10)
        assert len(results) == 0

        results = index.search("other", top_k=10)
        assert len(results) == 1

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "code_index"

            index = CodeIndex()
            code_block = CodeBlock(
                id="doc1_code_0",
                doc_id="doc1",
                chunk_id="doc1_chunk_0",
                content="def main(): pass",
                language="python",
            )
            index.add_code_block(code_block)
            index.persist(index_path)

            new_index = CodeIndex()
            new_index.load(index_path)

            results = new_index.search("main", top_k=5)
            assert len(results) == 1
            assert results[0]["doc_id"] == "doc1"

    def test_empty_query_returns_empty(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="some code",
            language="python",
        )
        index.add_code_block(code_block)

        results = index.search("", top_k=5)
        assert results == []

        results = index.search("   ", top_k=5)
        assert results == []

    def test_clear(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="def test(): pass",
            language="python",
        )
        index.add_code_block(code_block)
        assert len(index) == 1

        index.clear()
        assert len(index) == 0

    def test_stores_language(self):
        index = CodeIndex()
        code_block = CodeBlock(
            id="doc1_code_0",
            doc_id="doc1",
            chunk_id="doc1_chunk_0",
            content="fn main() {}",
            language="rust",
        )
        index.add_code_block(code_block)

        results = index.search("main", top_k=5)
        assert len(results) == 1
        assert results[0]["language"] == "rust"


# ============================================================================
# Missing MAIN Index Regression Tests (Issue: whoosh.index.IndexError)
# ============================================================================


def test_code_index_load_handles_missing_main_index(tmp_path):
    """
    load() gracefully handles directory with missing MAIN index segment.

    When the index directory exists but lacks valid index files (e.g., only
    contains partial files or is empty), whoosh raises IndexError with message
    "Index 'MAIN' does not exist". The index should reinitialize rather than crash.
    """
    index_dir = tmp_path / "incomplete_code_index"
    index_dir.mkdir()

    # Create an incomplete index structure (directory exists but no MAIN segment)
    (index_dir / "WRITELOCK").touch()

    code_index = CodeIndex()
    code_index.load(index_dir)

    # Should have reinitialized - verify by adding and searching
    code_block = CodeBlock(
        id="recovery_code_0",
        doc_id="recovery_doc",
        chunk_id="recovery_chunk_0",
        content="def recovered_function(): pass",
        language="python",
    )
    code_index.add_code_block(code_block)
    results = code_index.search("recovered_function", top_k=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "recovery_doc"


def test_code_index_load_handles_empty_directory(tmp_path):
    """
    load() handles completely empty directory that passes exists() check.

    Edge case where directory exists but is completely empty - no index files at all.
    """
    empty_dir = tmp_path / "empty_code_index"
    empty_dir.mkdir()

    code_index = CodeIndex()
    code_index.load(empty_dir)

    # Should work normally after reinitialization
    code_block = CodeBlock(
        id="empty_recovery_code_0",
        doc_id="empty_recovery_doc",
        chunk_id="empty_recovery_chunk_0",
        content="class EmptyRecovery:\n    pass",
        language="python",
    )
    code_index.add_code_block(code_block)
    results = code_index.search("EmptyRecovery", top_k=5)
    assert len(results) == 1
