from types import SimpleNamespace

import pytest
from searchkernel.domain import ChunkResult

from mcp_markdown_ragdocs.config import Config
from mcp_markdown_ragdocs.server import QueryRequest, create_app


class FakeSearchUseCase:
    def __init__(self):
        self.request = None

    async def execute(self, request):
        self.request = request
        return SimpleNamespace(
            results=[
                ChunkResult(
                    chunk_id="chunk_1",
                    record_id="doc_1",
                    score=0.5,
                    content="result",
                    metadata={"header_path": "", "file_path": ""},
                )
            ]
        )


def _query_endpoint(app):
    return next(route.endpoint for route in app.routes if route.path == "/query_documents")


@pytest.mark.asyncio
async def test_legacy_query_documents_passes_source_and_project_filters():
    app = create_app()
    search_use_case = FakeSearchUseCase()
    app.state.search_use_case = search_use_case
    app.state.config = Config()

    response = await _query_endpoint(app)(
        QueryRequest(
            query="find commits",
            project_filter=["docs"],
            source_filter=["git_commit"],
        )
    )

    assert response.results == [
        {
            "chunk_id": "chunk_1",
            "doc_id": "doc_1",
            "score": 0.5,
            "header_path": "",
            "file_path": "",
            "content": "result",
        }
    ]
    assert search_use_case.request.project_filter == ("docs",)
    assert search_use_case.request.source_filter == ("git_commit",)


@pytest.mark.asyncio
async def test_legacy_query_documents_keeps_project_filter_default():
    app = create_app()
    search_use_case = FakeSearchUseCase()
    app.state.search_use_case = search_use_case
    app.state.config = Config()

    await _query_endpoint(app)(QueryRequest(query="find docs"))

    assert search_use_case.request.project_filter == ()
    assert search_use_case.request.source_filter == ()


def test_query_request_preserves_min_score_omission() -> None:
    assert QueryRequest(query="find docs").min_score is None
    assert QueryRequest(query="find docs", min_score=0.0).min_score == 0.0


@pytest.mark.asyncio
async def test_legacy_query_documents_forwards_search_controls():
    app = create_app()
    search_use_case = FakeSearchUseCase()
    app.state.search_use_case = search_use_case
    app.state.config = Config()

    await _query_endpoint(app)(
        QueryRequest(
            query="filtered docs",
            min_score=0.6,
            similarity_threshold=0.9,
            uniqueness_mode="one_per_document",
            excluded_files=["private.md"],
            project_filter=["docs"],
            source_filter=["note"],
        )
    )

    assert search_use_case.request.min_score == 0.6
    assert search_use_case.request.similarity_threshold == 0.9
    assert search_use_case.request.max_chunks_per_doc == 1
    assert search_use_case.request.excluded_files == {"private.md"}
