import pytest

from mcp_markdown_ragdocs.search import CanonicalSearchAdapter


class _Vector:
    def __init__(self) -> None:
        self.chunks = {
            "chunk-a": {
                "chunk_id": "chunk-a",
                "doc_id": "doc-a",
                "content": "Authentication guidance",
                "metadata": {
                    "source_kind": "note",
                    "project_id": "project-a",
                    "file_path": "a.md",
                    "header_path": "Auth",
                },
            },
            "chunk-b": {
                "chunk_id": "chunk-b",
                "doc_id": "doc-b",
                "content": "Git commit authentication fix",
                "metadata": {
                    "source_kind": "git_commit",
                    "project_id": "project-b",
                    "file_path": "repo",
                    "header_path": "Commit",
                },
            },
        }

    def get_chunk_by_id(self, chunk_id: str):
        return self.chunks.get(chunk_id)


class _Keyword:
    def __init__(self, vector: _Vector) -> None:
        self._vector = vector

    def search(self, _query: str, top_k: int = 10):
        return [
            {"chunk_id": chunk_id, "doc_id": chunk["doc_id"], "score": score}
            for chunk_id, chunk, score in (
                ("chunk-a", self._vector.chunks["chunk-a"], 2.0),
                ("chunk-b", self._vector.chunks["chunk-b"], 1.0),
            )
        ][:top_k]


class _Manager:
    def __init__(self) -> None:
        self.vector = _Vector()
        self.keyword = _Keyword(self.vector)


@pytest.fixture
def adapter() -> CanonicalSearchAdapter:
    return CanonicalSearchAdapter(_Manager())


@pytest.mark.asyncio
async def test_query_preserves_chunk_result_semantics(adapter):
    results, stats, strategy = await adapter.query(
        "authentication",
        top_k=10,
        top_n=1,
    )

    assert [result.chunk_id for result in results] == ["chunk-a"]
    assert results[0].doc_id == "doc-a"
    assert results[0].file_path == "a.md"
    assert stats.original_count == 1
    assert strategy.keyword_count == 1


@pytest.mark.asyncio
async def test_query_applies_canonical_source_filter(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        source_filter=["git_commit"],
    )

    assert [result.chunk_id for result in results] == ["chunk-b"]


@pytest.mark.asyncio
async def test_query_applies_project_filter_without_legacy_orchestrator(adapter):
    results, _, _ = await adapter.query(
        "authentication",
        top_k=10,
        top_n=10,
        project_filter=["project-b"],
    )

    assert [result.chunk_id for result in results] == ["chunk-b"]


@pytest.mark.asyncio
async def test_empty_query_returns_no_results(adapter):
    results, _, _ = await adapter.query("", top_k=10, top_n=10)

    assert results == []
