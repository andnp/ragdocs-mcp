import asyncio
import multiprocessing
import tempfile
import time
from pathlib import Path

import pytest

from src.config import Config, IndexingConfig
from src.context import ApplicationContext


def run_server_file_lock_mode(index_path: str, ready_queue, error_queue):
    config = Config()
    config.indexing = IndexingConfig(
        documents_path=str(Path(__file__).parent.parent / "fixtures" / "sample_docs"),
        index_path=index_path,
        coordination_mode="file_lock",
    )

    from src.indices.vector import VectorIndex
    from src.indices.keyword import KeywordIndex
    from src.indices.graph import GraphStore
    from src.indexing.manager import IndexManager
    from src.search.orchestrator import SearchOrchestrator

    vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

    ctx = ApplicationContext(
        config=config,
        index_manager=manager,
        orchestrator=orchestrator,
        index_path=Path(index_path),
    )

    try:
        asyncio.run(ctx.start(background_index=False))
        ready_queue.put("ready")
        time.sleep(2)
    except Exception as e:
        error_queue.put(str(e))
    finally:
        try:
            asyncio.run(ctx.stop())
        except Exception:
            pass


def run_server_singleton_mode(index_path: str, ready_queue, error_queue):
    config = Config()
    config.indexing = IndexingConfig(
        documents_path=str(Path(__file__).parent.parent / "fixtures" / "sample_docs"),
        index_path=index_path,
        coordination_mode="singleton",
    )

    from src.indices.vector import VectorIndex
    from src.indices.keyword import KeywordIndex
    from src.indices.graph import GraphStore
    from src.indexing.manager import IndexManager
    from src.search.orchestrator import SearchOrchestrator

    vector = VectorIndex(embedding_model_name="BAAI/bge-small-en-v1.5")
    keyword = KeywordIndex()
    graph = GraphStore()
    manager = IndexManager(config, vector, keyword, graph)
    orchestrator = SearchOrchestrator(vector, keyword, graph, config, manager)

    ctx = ApplicationContext(
        config=config,
        index_manager=manager,
        orchestrator=orchestrator,
        index_path=Path(index_path),
    )

    try:
        asyncio.run(ctx.start(background_index=False))
        ready_queue.put("ready")
        time.sleep(2)
    except Exception as e:
        error_queue.put(str(e))
    finally:
        try:
            asyncio.run(ctx.stop())
        except Exception:
            pass


@pytest.mark.integration
def test_concurrent_server_startup_file_lock_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = str(Path(tmpdir) / "index_data")

        ready_queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()

        proc1 = multiprocessing.Process(
            target=run_server_file_lock_mode,
            args=(index_path, ready_queue, error_queue)
        )
        proc2 = multiprocessing.Process(
            target=run_server_file_lock_mode,
            args=(index_path, ready_queue, error_queue)
        )

        proc1.start()

        time.sleep(0.5)

        proc2.start()

        proc1.join(timeout=5)
        proc2.join(timeout=5)

        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())

        assert len(errors) == 0, f"Expected both instances to start, but got errors: {errors}"

        ready_count = 0
        while not ready_queue.empty():
            ready_queue.get()
            ready_count += 1

        assert ready_count == 2, f"Expected 2 instances to be ready, got {ready_count}"

        if proc1.is_alive():
            proc1.terminate()
            proc1.join(timeout=1)
        if proc2.is_alive():
            proc2.terminate()
            proc2.join(timeout=1)


@pytest.mark.integration
def test_concurrent_server_startup_singleton_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = str(Path(tmpdir) / "index_data")

        ready_queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()

        proc1 = multiprocessing.Process(
            target=run_server_singleton_mode,
            args=(index_path, ready_queue, error_queue)
        )
        proc2 = multiprocessing.Process(
            target=run_server_singleton_mode,
            args=(index_path, ready_queue, error_queue)
        )

        proc1.start()

        time.sleep(0.5)

        proc2.start()

        proc1.join(timeout=5)
        proc2.join(timeout=5)

        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())

        assert len(errors) == 1
        assert "Another mcp-markdown-ragdocs instance is already running" in errors[0]

        if proc1.is_alive():
            proc1.terminate()
            proc1.join(timeout=1)
        if proc2.is_alive():
            proc2.terminate()
            proc2.join(timeout=1)
