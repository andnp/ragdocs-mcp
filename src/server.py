import asyncio
import glob
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.cli import _should_include_file
from src.config import load_config
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest, should_rebuild
from src.indexing.reconciler import build_indexed_files_map, reconcile_indices
from src.indexing.watcher import FileWatcher
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)


async def _periodic_reconciliation(
    config,
    manager: IndexManager,
    index_path: Path,
    current_manifest: IndexManifest
):
    interval = config.indexing.reconciliation_interval_seconds

    while True:
        try:
            await asyncio.sleep(interval)

            logger.info("Starting periodic reconciliation")
            docs_path = Path(config.indexing.documents_path)
            pattern = str(docs_path / "**" / "*.md")
            all_files = glob.glob(pattern, recursive=config.indexing.recursive)
            discovered_files = [
                f for f in all_files
                if _should_include_file(
                    f,
                    config.indexing.include,
                    config.indexing.exclude,
                    config.indexing.exclude_hidden_dirs
                )
            ]

            # Load latest manifest
            saved_manifest = load_manifest(index_path)
            if not saved_manifest:
                logger.warning("No manifest found during reconciliation, skipping")
                continue

            files_to_add, doc_ids_to_remove = reconcile_indices(
                discovered_files,
                saved_manifest,
                docs_path
            )

            # Clean stale entries
            for doc_id in doc_ids_to_remove:
                manager.remove_document(doc_id)

            # Index new files
            for file_path in files_to_add:
                manager.index_document(file_path)

            # Persist changes if any
            if files_to_add or doc_ids_to_remove:
                manager.persist()
                current_manifest.indexed_files = build_indexed_files_map(discovered_files, docs_path)
                save_manifest(index_path, current_manifest)
                logger.info(
                    f"Periodic reconciliation complete: added {len(files_to_add)}, "
                    f"removed {len(doc_ids_to_remove)}"
                )
            else:
                logger.debug("Periodic reconciliation: no changes needed")

        except asyncio.CancelledError:
            logger.info("Periodic reconciliation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during periodic reconciliation: {e}", exc_info=True)
            # Continue running despite errors


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
    indexing_service: dict[str, Any]
    indices: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    from src.config import detect_project, resolve_index_path

    detected_project = detect_project(projects=config.projects)
    index_path_resolved = resolve_index_path(config, detected_project)

    config.indexing.index_path = str(index_path_resolved)

    vector = VectorIndex()
    keyword = KeywordIndex()
    graph = GraphStore()

    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = QueryOrchestrator(vector, keyword, graph, config, manager)

    index_path = Path(config.indexing.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    current_manifest = IndexManifest(
        spec_version="1.0.0",
        embedding_model=config.llm.embedding_model,
        parsers=config.parsers,
        chunking_config={
            "strategy": config.chunking.strategy,
            "min_chunk_chars": config.chunking.min_chunk_chars,
            "max_chunk_chars": config.chunking.max_chunk_chars,
            "overlap_chars": config.chunking.overlap_chars,
        },
    )

    saved_manifest = load_manifest(index_path)
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)

    if needs_rebuild:
        logger.info("Index rebuild required - indexing all documents")
        docs_path = Path(config.indexing.documents_path)
        pattern = str(docs_path / "**" / "*.md")
        all_files = glob.glob(pattern, recursive=config.indexing.recursive)
        files_to_index = [
            f for f in all_files
            if _should_include_file(
                f,
                config.indexing.include,
                config.indexing.exclude,
                config.indexing.exclude_hidden_dirs
            )
        ]
        for file_path in files_to_index:
            manager.index_document(file_path)
        manager.persist()

        # Update manifest with indexed files
        current_manifest.indexed_files = build_indexed_files_map(files_to_index, docs_path)
        save_manifest(index_path, current_manifest)
        logger.info(f"Initial indexing complete: {len(files_to_index)} documents indexed")
    else:
        logger.info("Loading existing indices")
        manager.load()

        # Perform startup reconciliation
        logger.info("Running startup reconciliation")
        docs_path = Path(config.indexing.documents_path)
        pattern = str(docs_path / "**" / "*.md")
        all_files = glob.glob(pattern, recursive=config.indexing.recursive)
        discovered_files = [
            f for f in all_files
            if _should_include_file(
                f,
                config.indexing.include,
                config.indexing.exclude,
                config.indexing.exclude_hidden_dirs
            )
        ]

        # saved_manifest is guaranteed to exist here since needs_rebuild is False
        if saved_manifest is None:
            raise RuntimeError("Manifest should exist when needs_rebuild is False")
        files_to_add, doc_ids_to_remove = reconcile_indices(
            discovered_files,
            saved_manifest,
            docs_path
        )

        # Clean stale entries
        for doc_id in doc_ids_to_remove:
            manager.remove_document(doc_id)

        # Index new files
        for file_path in files_to_add:
            manager.index_document(file_path)

        # Persist changes if any
        if files_to_add or doc_ids_to_remove:
            manager.persist()
            current_manifest.indexed_files = build_indexed_files_map(discovered_files, docs_path)
            save_manifest(index_path, current_manifest)
            logger.info(f"Reconciliation complete: added {len(files_to_add)}, removed {len(doc_ids_to_remove)}")
        else:
            logger.info("Reconciliation complete: no changes needed")

    watcher = FileWatcher(
        documents_path=config.indexing.documents_path,
        index_manager=manager,
    )
    watcher.start()
    logger.info("File watcher started")

    # Start periodic reconciliation task if enabled
    reconciliation_task = None
    if config.indexing.reconciliation_interval_seconds > 0:
        reconciliation_task = asyncio.create_task(
            _periodic_reconciliation(
                config,
                manager,
                index_path,
                current_manifest
            )
        )
        logger.info(
            f"Periodic reconciliation enabled (interval: "
            f"{config.indexing.reconciliation_interval_seconds}s)"
        )
    else:
        logger.info("Periodic reconciliation disabled")

    app.state.config = config
    app.state.indices = (vector, keyword, graph)
    app.state.manager = manager
    app.state.orchestrator = orchestrator
    app.state.watcher = watcher
    app.state.reconciliation_task = reconciliation_task
    app.state.index_path = index_path
    app.state.current_manifest = current_manifest

    yield

    logger.info("Shutting down server")

    # Cancel periodic reconciliation task
    if hasattr(app.state, 'reconciliation_task') and app.state.reconciliation_task:
        app.state.reconciliation_task.cancel()
        try:
            await app.state.reconciliation_task
        except asyncio.CancelledError:
            pass
        logger.info("Reconciliation task cancelled")

    await watcher.stop()
    manager.persist()
    logger.info("Shutdown complete")


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
