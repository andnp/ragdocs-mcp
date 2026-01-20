"""
Unit tests for CodeIndex corruption handling.

Tests that Whoosh segment corruption is handled gracefully with reinitialization.
"""

import glob
from pathlib import Path

import pytest

from src.indices.code import CodeIndex
from src.models import CodeBlock


@pytest.fixture
def code_index():
    return CodeIndex()


@pytest.fixture
def sample_code_block():
    return CodeBlock(
        id="code_block_1",
        doc_id="test_doc",
        chunk_id="test_chunk",
        content="def hello_world():\n    print('Hello, World!')",
        language="python",
    )


def test_code_index_search_handles_corrupted_segment(tmp_path, code_index, sample_code_block):
    """
    Search operation handles corrupted segment files gracefully.

    When Whoosh segment files are corrupted, search should detect
    the issue, reinitialize the index, and return empty results.
    """
    code_index.add_code_block(sample_code_block)

    index_path = tmp_path / "corrupted_code_index"
    code_index.persist(index_path)
    code_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    assert len(seg_files) > 0, "Expected segment files after persist"
    for seg in seg_files:
        Path(seg).unlink()

    results = code_index.search("hello_world", top_k=5)

    assert results == []


def test_code_index_remove_handles_corrupted_segment(tmp_path, code_index, sample_code_block):
    """
    Remove operation handles corrupted segment files gracefully.

    When Whoosh segment files are corrupted, remove should detect
    the issue, reinitialize the index, and not crash.
    """
    code_index.add_code_block(sample_code_block)

    index_path = tmp_path / "corrupted_remove_index"
    code_index.persist(index_path)
    code_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    assert len(seg_files) > 0, "Expected segment files after persist"
    for seg in seg_files:
        Path(seg).unlink()

    code_index.remove_by_doc_id("test_doc")

    results = code_index.search("hello", top_k=5)
    assert isinstance(results, list)


def test_code_index_recovery_allows_reindexing(tmp_path, code_index, sample_code_block):
    """
    After corruption recovery, new code blocks can be indexed.

    Tests the full cycle: create index, persist, corrupt, detect
    corruption, reinitialize, then add new code blocks.
    """
    code_index.add_code_block(sample_code_block)

    index_path = tmp_path / "recovery_test_index"
    code_index.persist(index_path)
    code_index.load(index_path)

    seg_files = glob.glob(str(index_path / "*.seg"))
    for seg in seg_files:
        Path(seg).unlink()

    code_index.search("trigger corruption detection", top_k=5)

    new_code_block = CodeBlock(
        id="code_block_2",
        doc_id="new_doc",
        chunk_id="new_chunk",
        content="class NewClass:\n    pass",
        language="python",
    )
    code_index.add_code_block(new_code_block)

    results = code_index.search("NewClass", top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "code_block_2"


def test_code_index_empty_query_returns_empty(code_index, sample_code_block):
    """
    Empty query returns empty results without errors.

    Tests that empty/whitespace queries are handled correctly.
    """
    code_index.add_code_block(sample_code_block)

    assert code_index.search("", top_k=5) == []
    assert code_index.search("   ", top_k=5) == []


def test_code_index_persist_and_load(tmp_path, code_index, sample_code_block):
    """
    CodeIndex persists and loads correctly.

    Tests basic persist/load functionality.
    """
    code_index.add_code_block(sample_code_block)

    persist_path = tmp_path / "code_index"
    code_index.persist(persist_path)

    assert persist_path.exists()

    code_index2 = CodeIndex()
    code_index2.load(persist_path)

    results = code_index2.search("hello_world", top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "code_block_1"


def test_code_index_load_nonexistent_path(tmp_path):
    """
    Loading from nonexistent path initializes fresh index.

    Tests that missing path creates new empty index.
    """
    code_index = CodeIndex()
    nonexistent_path = tmp_path / "nonexistent"

    code_index.load(nonexistent_path)

    code_block = CodeBlock(
        id="new_block",
        doc_id="doc",
        chunk_id="chunk",
        content="const test = 42;",
        language="javascript",
    )
    code_index.add_code_block(code_block)

    results = code_index.search("test", top_k=5)
    assert len(results) == 1


def test_code_index_camel_case_search(code_index):
    """
    CodeIndex splits camelCase for better search.

    Tests that camelCase identifiers are searchable by parts.
    """
    code_block = CodeBlock(
        id="camel_block",
        doc_id="doc",
        chunk_id="chunk",
        content="function getUserProfile() { return profile; }",
        language="javascript",
    )
    code_index.add_code_block(code_block)

    results = code_index.search("user profile", top_k=5)
    assert len(results) == 1

    results = code_index.search("getUserProfile", top_k=5)
    assert len(results) == 1


def test_code_index_snake_case_search(code_index):
    """
    CodeIndex splits snake_case for better search.

    Tests that snake_case identifiers are searchable by parts.
    """
    code_block = CodeBlock(
        id="snake_block",
        doc_id="doc",
        chunk_id="chunk",
        content="def get_user_profile():\n    return profile",
        language="python",
    )
    code_index.add_code_block(code_block)

    results = code_index.search("user profile", top_k=5)
    assert len(results) == 1

    results = code_index.search("get_user_profile", top_k=5)
    assert len(results) == 1


def test_code_index_multiple_code_blocks(code_index):
    """
    CodeIndex handles multiple code blocks correctly.

    Tests adding and searching across multiple code blocks.
    """
    blocks = [
        CodeBlock(
            id=f"block_{i}",
            doc_id=f"doc_{i}",
            chunk_id=f"chunk_{i}",
            content=content,
            language=lang,
        )
        for i, (content, lang) in enumerate([
            ("def authentication_handler():\n    pass", "python"),
            ("class AuthenticationService { }", "java"),
            ("function validateInput(data) { }", "javascript"),
        ])
    ]

    for block in blocks:
        code_index.add_code_block(block)

    results = code_index.search("authentication", top_k=5)
    assert len(results) == 2

    results = code_index.search("validate", top_k=5)
    assert len(results) == 1


def test_code_index_schema_mismatch_triggers_rebuild(tmp_path):
    """
    Loading an index with mismatched schema triggers rebuild.

    When persisted index has different schema, index should rebuild.
    """
    from whoosh import index as whoosh_index
    from whoosh.fields import ID, TEXT, Schema

    old_schema = Schema(
        id=ID(stored=True, unique=True),
        content=TEXT(stored=True),
    )
    index_path = tmp_path / "old_code_index"
    index_path.mkdir()
    whoosh_index.create_in(str(index_path), old_schema)

    code_index = CodeIndex()
    code_index.load(index_path)

    code_block = CodeBlock(
        id="new_block",
        doc_id="doc",
        chunk_id="chunk",
        content="const x = 1;",
        language="javascript",
    )
    code_index.add_code_block(code_block)

    results = code_index.search("const", top_k=5)
    assert len(results) == 1


def test_code_index_clear_and_reuse(code_index, sample_code_block):
    """
    CodeIndex can be cleared and reused.

    Tests that clear() properly resets the index.
    """
    code_index.add_code_block(sample_code_block)
    assert len(code_index) == 1

    code_index.clear()
    assert len(code_index) == 0

    new_block = CodeBlock(
        id="after_clear",
        doc_id="doc",
        chunk_id="chunk",
        content="new content",
        language="text",
    )
    code_index.add_code_block(new_block)

    results = code_index.search("new content", top_k=5)
    assert len(results) == 1
