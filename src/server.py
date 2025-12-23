import glob
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import load_config
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, load_manifest, save_manifest, should_rebuild
from src.indexing.watcher import FileWatcher
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex
from src.search.orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    server_status: str
    indexing_service: dict[str, Any]
    indices: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

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
    )

    saved_manifest = load_manifest(index_path)
    needs_rebuild = should_rebuild(current_manifest, saved_manifest)

    if needs_rebuild:
        logger.info("Index rebuild required - indexing all documents")
        docs_path = Path(config.indexing.documents_path)
        pattern = str(docs_path / "**" / "*.md")
        for file_path in glob.glob(pattern, recursive=config.indexing.recursive):
            manager.index_document(file_path)
        manager.persist()
        save_manifest(index_path, current_manifest)
        logger.info("Initial indexing complete")
    else:
        logger.info("Loading existing indices")
        manager.load()

    watcher = FileWatcher(
        documents_path=config.indexing.documents_path,
        index_manager=manager,
    )
    watcher.start()
    logger.info("File watcher started")

    app.state.config = config
    app.state.indices = (vector, keyword, graph)
    app.state.manager = manager
    app.state.orchestrator = orchestrator
    app.state.watcher = watcher

    yield

    logger.info("Shutting down server")
    await watcher.stop()
    manager.persist()
    logger.info("Shutdown complete")


def create_app():
    app = FastAPI(lifespan=lifespan)

    @app.post("/query_documents")
    async def query_documents(request: QueryRequest):
        orchestrator = app.state.orchestrator
        results = await orchestrator.query(request.query, top_k=10)
        answer = await orchestrator.synthesize_answer(request.query, results)
        return QueryResponse(answer=answer)

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
