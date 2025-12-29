import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.context import ApplicationContext
from src.indexing.manifest import load_manifest

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, ge=1, le=100, description="Maximum results to return")


class QueryResponse(BaseModel):
    answer: str
    results: list[dict[str, str | float]]


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    server_status: str
    # REVIEW [MED] Type Safety: dict[str, Any] is overly broad. Consider defining
    # typed dataclasses for indexing_service (pending_queue_size: int, last_sync_time: str | None,
    # failed_files: list[dict]) and indices (document_count: int, index_version: str).
    indexing_service: dict[str, Any]
    indices: dict[str, Any]


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

    @app.post("/query_documents")
    async def query_documents(request: QueryRequest):
        orchestrator = app.state.orchestrator
        top_k = max(20, request.top_n * 4)
        results, _ = await orchestrator.query(
            request.query,
            top_k=top_k,
            top_n=request.top_n
        )
        chunk_ids = [result.chunk_id for result in results]
        answer = await orchestrator.synthesize_answer(request.query, chunk_ids)

        results_dict = [result.to_dict() for result in results]

        return QueryResponse(answer=answer, results=results_dict)

    @app.post("/query_documents_stream")
    async def query_documents_stream(request: QueryRequest):
        orchestrator = app.state.orchestrator
        top_k = max(20, request.top_n * 4)
        results, _ = await orchestrator.query(
            request.query,
            top_k=top_k,
            top_n=request.top_n
        )
        chunk_ids = [result.chunk_id for result in results]

        async def event_stream() -> AsyncIterator[str]:
            results_dict = [result.to_dict() for result in results]
            yield "event: results\n"
            yield f"data: {json.dumps({'results': results_dict})}\n\n"

            async for event in orchestrator.synthesize_answer_stream(request.query, chunk_ids):
                yield f"event: {event['event']}\n"
                yield f"data: {json.dumps(event['data'])}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

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
            indexing_service={
                "pending_queue_size": watcher.get_pending_queue_size(),
                "last_sync_time": watcher.get_last_sync_time(),
                "failed_files": watcher.get_failed_files(),
            },
            indices={
                "document_count": manager.get_document_count(),
                "index_version": index_version,
            },
        )

    return app
