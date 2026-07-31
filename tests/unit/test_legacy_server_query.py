import pytest
from searchkernel.domain import ChunkResult

from mcp_markdown_ragdocs.server import QueryRequest, create_app


class FakeOrchestrator:
    def __init__(self):
        self.query_kwargs = None

    async def query(self, query, **kwargs):
        self.query_kwargs = {"query": query, **kwargs}
        return (
            [
                ChunkResult(
                    chunk_id="chunk_1",
                    record_id="doc_1",
                    score=0.5,
                    content="result",
                    metadata={"header_path": "", "file_path": ""},
                )
            ],
            None,
            None,
        )


def _query_endpoint(app):
    return next(route.endpoint for route in app.routes if route.path == "/query_documents")


@pytest.mark.asyncio
async def test_legacy_query_documents_passes_source_and_project_filters():
    app = create_app()
    orchestrator = FakeOrchestrator()
    app.state.orchestrator = orchestrator

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
    assert orchestrator.query_kwargs["project_filter"] == ["docs"]
    assert orchestrator.query_kwargs["source_filter"] == ["git_commit"]


@pytest.mark.asyncio
async def test_legacy_query_documents_keeps_project_filter_default():
    app = create_app()
    orchestrator = FakeOrchestrator()
    app.state.orchestrator = orchestrator

    await _query_endpoint(app)(QueryRequest(query="find docs"))

    assert orchestrator.query_kwargs["project_filter"] == []
    assert orchestrator.query_kwargs["source_filter"] is None
