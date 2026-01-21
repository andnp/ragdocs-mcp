import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.context import ApplicationContext
from src.indexing.manifest import load_manifest
from src.search.pipeline import SearchPipelineConfig
from src.search.utils import classify_query_type, truncate_content

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, ge=1, le=100, description="Maximum results to return")


class QueryResponse(BaseModel):
    results: list[dict[str, str | float]]


class HealthResponse(BaseModel):
    status: str


class IndexingServiceStatus(BaseModel):
    pending_queue_size: int
    last_sync_time: str | None
    failed_files: list[dict[str, str]]


class IndicesStatus(BaseModel):
    document_count: int
    index_version: str


class StatusResponse(BaseModel):
    server_status: str
    indexing_service: IndexingServiceStatus
    indices: IndicesStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = ApplicationContext.create(
        project_override=None,
        enable_watcher=True,
        lazy_embeddings=True,
    )

    await ctx.start(background_index=False)

    app.state.ctx = ctx
    app.state.config = ctx.config
    app.state.indices = (
        ctx.index_manager.vector,
        ctx.index_manager.keyword,
        ctx.index_manager.graph,
    )
    app.state.manager = ctx.index_manager
    app.state.orchestrator = ctx.orchestrator
    app.state.watcher = ctx.watcher
    app.state.reconciliation_task = ctx.reconciliation_task
    app.state.index_path = ctx.index_path
    app.state.current_manifest = ctx.current_manifest

    yield

    await ctx.stop()


def create_app():
    app = FastAPI(lifespan=lifespan)

    async def _execute_query(
        orchestrator,
        query: str,
        top_n: int,
        max_chunks_per_doc: int = 0,
    ):
        top_k = max(20, top_n * 4)

        pipeline_config = SearchPipelineConfig(
            min_confidence=0.0,
            max_chunks_per_doc=max_chunks_per_doc,
            dedup_enabled=True,
            dedup_threshold=0.85,
            rerank_enabled=False,
        )

        results, _, _ = await orchestrator.query(
            query,
            top_k=top_k,
            top_n=top_n,
            pipeline_config=pipeline_config,
        )

        query_type = classify_query_type(query)

        formatted_results = []
        for i, result in enumerate(results):
            result_dict = result.to_dict()
            if query_type == "factual":
                result_dict["content"] = truncate_content(result_dict["content"], 200)
            formatted_results.append(result_dict)

        return formatted_results

    @app.post("/query_documents")
    async def query_documents(request: QueryRequest):
        results_dict = await _execute_query(
            app.state.orchestrator,
            request.query,
            request.top_n,
            max_chunks_per_doc=0,
        )
        return QueryResponse(results=results_dict)

    @app.post("/query_unique_documents")
    async def query_unique_documents(request: QueryRequest):
        results_dict = await _execute_query(
            app.state.orchestrator,
            request.query,
            request.top_n,
            max_chunks_per_doc=1,
        )
        return QueryResponse(results=results_dict)

    @app.get("/health")
    async def health():
        return HealthResponse(status="ok")

    @app.get("/status")
    async def status():
        config = app.state.config
        manager = app.state.manager
        watcher = app.state.watcher

        index_path = Path(config.indexing.index_path)
        saved_manifest = load_manifest(index_path)
        index_version = saved_manifest.spec_version if saved_manifest else "1.0.0"

        return StatusResponse(
            server_status="running",
            indexing_service=IndexingServiceStatus(
                pending_queue_size=watcher.get_pending_queue_size(),
                last_sync_time=watcher.get_last_sync_time(),
                failed_files=watcher.get_failed_files(),
            ),
            indices=IndicesStatus(
                document_count=manager.get_document_count(),
                index_version=index_version,
            ),
        )

    return app
