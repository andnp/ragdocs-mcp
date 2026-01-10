from datetime import datetime

import pytest

from src.indices.keyword import KeywordIndex
from src.models import Chunk


@pytest.fixture
def docs_root(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    return root


@pytest.fixture
def keyword_index():
    return KeywordIndex()


def test_keyword_search_without_exclusions(keyword_index, docs_root):
    chunk1 = Chunk(
        chunk_id="docs/api_chunk_0",
        doc_id="docs/api",
        content="API authentication using tokens",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "api.md"),
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk1)

    results = keyword_index.search("authentication", top_k=5)
    assert len(results) > 0
    assert any("docs/api" in r["chunk_id"] for r in results)


def test_keyword_search_with_exclusions_exact_match(keyword_index, docs_root):
    chunk1 = Chunk(
        chunk_id="docs/api_chunk_0",
        doc_id="docs/api",
        content="API authentication using tokens",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "api.md"),
        modified_time=datetime.now(),
    )

    chunk2 = Chunk(
        chunk_id="docs/guide_chunk_0",
        doc_id="docs/guide",
        content="Authentication guide for users",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "guide.md"),
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk1)
    keyword_index.add_chunk(chunk2)

    excluded = {"docs/api"}
    results = keyword_index.search("authentication", top_k=5, excluded_files=excluded, docs_root=docs_root)

    assert len(results) > 0
    assert not any("docs/api" in r["chunk_id"] for r in results)
    assert any("docs/guide" in r["chunk_id"] for r in results)


def test_keyword_search_with_exclusions_filename_match(keyword_index, docs_root):
    chunk1 = Chunk(
        chunk_id="docs/README_chunk_0",
        doc_id="docs/README",
        content="Project README documentation",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "README.md"),
        modified_time=datetime.now(),
    )

    chunk2 = Chunk(
        chunk_id="docs/guide_chunk_0",
        doc_id="docs/guide",
        content="Project guide and documentation",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "guide.md"),
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk1)
    keyword_index.add_chunk(chunk2)

    excluded = {"README"}
    results = keyword_index.search("documentation", top_k=5, excluded_files=excluded, docs_root=docs_root)

    assert len(results) > 0
    assert not any("README" in r["chunk_id"] for r in results)
    assert any("guide" in r["chunk_id"] for r in results)


def test_keyword_search_with_empty_exclusion_set(keyword_index, docs_root):
    chunk1 = Chunk(
        chunk_id="docs/api_chunk_0",
        doc_id="docs/api",
        content="API documentation",
        metadata={"tags": []},
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=50,
        file_path=str(docs_root / "docs" / "api.md"),
        modified_time=datetime.now(),
    )

    keyword_index.add_chunk(chunk1)

    results_with_empty = keyword_index.search("documentation", top_k=5, excluded_files=set(), docs_root=docs_root)
    results_without = keyword_index.search("documentation", top_k=5)

    assert len(results_with_empty) == len(results_without)


def test_keyword_search_over_fetching(keyword_index, docs_root):
    chunks = []
    for i in range(5):
        chunk = Chunk(
            chunk_id=f"docs/file{i}_chunk_0",
            doc_id=f"docs/file{i}",
            content=f"Content about API topic {i}",
            metadata={"tags": []},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=50,
            file_path=str(docs_root / "docs" / f"file{i}.md"),
            modified_time=datetime.now(),
        )
        chunks.append(chunk)
        keyword_index.add_chunk(chunk)

    excluded = {"docs/file0", "docs/file1"}
    results = keyword_index.search("API", top_k=3, excluded_files=excluded, docs_root=docs_root)

    assert len(results) <= 3
    assert not any("file0" in r["chunk_id"] for r in results)
    assert not any("file1" in r["chunk_id"] for r in results)


def test_keyword_search_with_multiple_exclusions(keyword_index, docs_root):
    chunks = []
    for name in ["api", "guide", "tutorial", "reference"]:
        chunk = Chunk(
            chunk_id=f"docs/{name}_chunk_0",
            doc_id=f"docs/{name}",
            content=f"Documentation for {name}",
            metadata={"tags": []},
            chunk_index=0,
            header_path="",
            start_pos=0,
            end_pos=50,
            file_path=str(docs_root / "docs" / f"{name}.md"),
            modified_time=datetime.now(),
        )
        chunks.append(chunk)
        keyword_index.add_chunk(chunk)

    excluded = {"docs/api", "docs/guide"}
    results = keyword_index.search("documentation", top_k=5, excluded_files=excluded, docs_root=docs_root)

    for result in results:
        assert "api" not in result["chunk_id"]
        assert "guide" not in result["chunk_id"]
