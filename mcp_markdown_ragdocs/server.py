import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from searchkernel.api import (
    FEDERATION_CONTRACT_VERSION,
    SearchRequest,
    classify_query_type,
    load_manifest,
    truncate_content,
)
from starlette.responses import JSONResponse

from mcp_markdown_ragdocs.app.search import SearchQuery
from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.context import ApplicationContext
from mcp_markdown_ragdocs.federation import (
    RAGDOCS_SOURCE,
    FederationRequestError,
    build_federation_capabilities,
    execute_federation_search,
    load_gdrive_source_health,
)
from mcp_markdown_ragdocs.models import ChunkResult

logger = logging.getLogger(__name__)

MAX_FEDERATION_REQUEST_BYTES = 256 * 1024
DEFAULT_FEDERATION_TIMEOUT_SECONDS = 30.0
MAX_CORRELATION_ID_LENGTH = 256


class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=5, ge=1, le=100, description="Maximum results to return")
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    uniqueness_mode: Literal["allow_multiple", "one_per_document"] = "one_per_document"
    max_chunks_per_doc: int = Field(default=0, ge=0, le=100)
    excluded_files: list[str] = Field(default_factory=list)
    project_filter: list[str] = Field(default_factory=list)
    source_filter: list[str] | None = None
    project_context: str | None = None


class QueryResponse(BaseModel):
    results: list[dict[str, object]]


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
    configure_runtime_threads()
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
    app.state.search_use_case = ctx.search_use_case
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
        project_filter: list[str] | None = None,
        source_filter: list[str] | None = None,
        project_context: str | None = None,
        excluded_files: set[str] | None = None,
        min_score: float | None = None,
        similarity_threshold: float | None = None,
    ):
        search_use_case = getattr(orchestrator, "search_use_case", None)
        if search_use_case is not None:
            execution = await search_use_case.execute(
                SearchQuery(
                    query=query,
                    top_n=top_n,
                    project_filter=tuple(project_filter or ()),
                    source_filter=tuple(source_filter or ()),
                    project_context=project_context,
                    excluded_files=frozenset(excluded_files or ()),
                    min_score=min_score,
                    similarity_threshold=similarity_threshold,
                    max_chunks_per_doc=max_chunks_per_doc,
                )
            )
            results = execution.results
        else:
            top_k = max(20, top_n * 4)
            if project_filter:
                top_k = max(top_k, top_n * 10)
            results, _, _ = await orchestrator.query(
                query,
                top_k=top_k,
                top_n=top_n,
                pipeline_config=None,
                project_filter=project_filter,
                source_filter=source_filter,
                project_context=project_context,
                excluded_files=excluded_files,
                min_score=min_score,
                similarity_threshold=similarity_threshold,
                max_chunks_per_doc=max_chunks_per_doc,
            )

        query_type = classify_query_type(query)

        formatted_results = []
        for i, result in enumerate(results):
            result = (
                result
                if isinstance(result, ChunkResult)
                else ChunkResult.from_domain(result)
            )
            result_dict = result.to_dict()
            if query_type == "factual":
                result_dict["content"] = truncate_content(result.content, 200)
            formatted_results.append(result_dict)

        return formatted_results

    @app.post("/query_documents")
    async def query_documents(request: QueryRequest):
        results_dict = await _execute_query(
            app.state.orchestrator,
            request.query,
            request.top_n,
            max_chunks_per_doc=(
                1 if request.uniqueness_mode == "one_per_document"
                else request.max_chunks_per_doc
            ),
            project_filter=request.project_filter,
            source_filter=request.source_filter,
            project_context=request.project_context,
            excluded_files=set(request.excluded_files),
            min_score=request.min_score,
            similarity_threshold=request.similarity_threshold,
        )
        return QueryResponse(results=results_dict)

    @app.get("/health")
    async def health():
        return HealthResponse(status="ok")

    @app.get("/v1/search/capabilities")
    async def federation_capabilities():
        config = getattr(app.state, "config", None)
        gdrive = getattr(config, "gdrive", None)
        return JSONResponse(
            build_federation_capabilities(
                gdrive_workspace_id=(
                    gdrive.workspace_id if getattr(gdrive, "enabled", False) else None
                )
            )
        )

    @app.get("/v1/health")
    async def federation_health():
        payload: dict[str, object] = {
            "status": "ok",
            "contract_version": FEDERATION_CONTRACT_VERSION,
            "source": RAGDOCS_SOURCE.to_dict(),
        }
        config = getattr(app.state, "config", None)
        gdrive = getattr(config, "gdrive", None)
        index_path = getattr(app.state, "index_path", None)
        if getattr(gdrive, "enabled", False) and index_path is not None:
            payload["source_health"] = {
                "gdrive": load_gdrive_source_health(
                    Path(index_path),
                    gdrive.workspace_id,
                )
            }
        return JSONResponse(payload)

    @app.post("/v1/search")
    async def federation_search(request: Request):
        request_id = request.headers.get("X-Request-ID", "").strip()
        trace_id = request.headers.get("X-Trace-ID", "").strip()
        if len(request_id) > MAX_CORRELATION_ID_LENGTH or len(trace_id) > MAX_CORRELATION_ID_LENGTH:
            return JSONResponse(
                {"error": "correlation_id_too_long"},
                status_code=400,
            )

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_FEDERATION_REQUEST_BYTES:
                    return JSONResponse(
                        {"error": "request_payload_too_large"},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"error": "invalid_content_length"}, status_code=400)

        body = await request.body()
        if len(body) > MAX_FEDERATION_REQUEST_BYTES:
            return JSONResponse(
                {"error": "request_payload_too_large"},
                status_code=413,
            )
        try:
            decoded = json.loads(body)
            if not isinstance(decoded, dict):
                raise ValueError("request body must be a JSON object")
            search_request = SearchRequest.from_dict(decoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return JSONResponse(
                {"error": "invalid_request", "details": str(exc)[:512]},
                status_code=400,
            )

        effective_request_id = search_request.request_id or request_id or uuid.uuid4().hex
        effective_trace_id = search_request.trace_id or trace_id
        if len(effective_request_id) > MAX_CORRELATION_ID_LENGTH or len(effective_trace_id) > MAX_CORRELATION_ID_LENGTH:
            return JSONResponse(
                {"error": "correlation_id_too_long"},
                status_code=400,
            )
        search_request = replace(
            search_request,
            request_id=effective_request_id,
            trace_id=effective_trace_id,
        )

        timeout_seconds = DEFAULT_FEDERATION_TIMEOUT_SECONDS
        if search_request.deadline_at is not None:
            timeout_seconds = min(
                timeout_seconds,
                (search_request.deadline_at.astimezone(UTC) - datetime.now(UTC)).total_seconds(),
            )
        if timeout_seconds <= 0:
            return JSONResponse(
                {"error": "request_deadline_expired", "request_id": effective_request_id},
                status_code=408,
                headers={"X-Request-ID": effective_request_id},
            )

        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is None:
            return JSONResponse(
                {"error": "search_unavailable", "request_id": effective_request_id},
                status_code=503,
                headers={"X-Request-ID": effective_request_id},
            )
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await execute_federation_search(
                    orchestrator,
                    search_request,
                    request_id=effective_request_id,
                    elapsed_start=monotonic(),
                )
        except FederationRequestError as exc:
            return JSONResponse(
                {"error": str(exc), "request_id": effective_request_id},
                status_code=exc.status_code,
                headers={"X-Request-ID": effective_request_id},
            )
        except TimeoutError:
            return JSONResponse(
                {"error": "search_deadline_exceeded", "request_id": effective_request_id},
                status_code=504,
                headers={"X-Request-ID": effective_request_id},
            )

        headers = {"X-Request-ID": effective_request_id}
        if effective_trace_id:
            headers["X-Trace-ID"] = effective_trace_id
        return JSONResponse(response.to_dict(), headers=headers)

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
