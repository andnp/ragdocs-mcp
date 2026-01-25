import asyncio
from multiprocessing import Event, Process, Queue
from pathlib import Path

import pytest

from src.config import (
    ChunkingConfig,
    Config,
    GitIndexingConfig,
    IndexingConfig,
    LLMConfig,
    SearchConfig,
    ServerConfig,
    WorkerConfig,
)
from src.ipc.commands import InitCompleteNotification, IndexUpdatedNotification
from src.worker.process import worker_main


@pytest.fixture
def test_config(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(docs_path),
            index_path=str(tmp_path / "index"),
            recursive=True,
        ),
        git_indexing=GitIndexingConfig(enabled=False),
        worker=WorkerConfig(
            progressive_snapshot_interval=2.0,
            progressive_snapshot_doc_count=5,
        ),
        parsers={"**/*.md": "MarkdownParser"},
        search=SearchConfig(),
        llm=LLMConfig(embedding_model="BAAI/bge-small-en-v1.5"),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
    )


def _create_test_documents(docs_path: Path, count: int):
    for i in range(count):
        doc_file = docs_path / f"doc_{i:03d}.md"
        doc_file.write_text(f"# Document {i}\n\nContent for document {i}.\n")


@pytest.mark.asyncio
async def test_progressive_snapshots_during_indexing(test_config, tmp_path, shared_embedding_model):
    docs_path = Path(test_config.indexing.documents_path)
    snapshot_base = tmp_path / "snapshots"
    snapshot_base.mkdir()

    _create_test_documents(docs_path, 15)

    command_queue = Queue()
    response_queue = Queue()
    shutdown_event = Event()

    config_dict = {
        "indexing": {
            "documents_path": test_config.indexing.documents_path,
            "index_path": test_config.indexing.index_path,
            "recursive": test_config.indexing.recursive,
            "include": test_config.indexing.include,
            "exclude": test_config.indexing.exclude,
            "exclude_hidden_dirs": test_config.indexing.exclude_hidden_dirs,
        },
        "worker": {
            "progressive_snapshot_interval": test_config.worker.progressive_snapshot_interval,
            "progressive_snapshot_doc_count": test_config.worker.progressive_snapshot_doc_count,
        },
        "git_indexing": {
            "enabled": False,
        },
    }

    worker_process = Process(
        target=worker_main,
        args=(config_dict, command_queue, response_queue, shutdown_event, snapshot_base),
    )
    worker_process.start()

    try:
        notification = response_queue.get(timeout=30)
        assert isinstance(notification, InitCompleteNotification)
        assert notification.version >= 1
        assert notification.doc_count == 15

        snapshot_dir = snapshot_base / f"v{notification.version}"
        assert snapshot_dir.exists()
        assert (snapshot_dir / "vector").exists()
        assert (snapshot_dir / "keyword").exists()
        assert (snapshot_dir / "graph").exists()

    finally:
        shutdown_event.set()
        worker_process.join(timeout=10)
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=2)


@pytest.mark.asyncio
async def test_progressive_snapshots_with_file_watching(test_config, tmp_path, shared_embedding_model):
    docs_path = Path(test_config.indexing.documents_path)
    snapshot_base = tmp_path / "snapshots"
    snapshot_base.mkdir()

    _create_test_documents(docs_path, 3)

    command_queue = Queue()
    response_queue = Queue()
    shutdown_event = Event()

    config_dict = {
        "indexing": {
            "documents_path": test_config.indexing.documents_path,
            "index_path": test_config.indexing.index_path,
            "recursive": test_config.indexing.recursive,
            "include": test_config.indexing.include,
            "exclude": test_config.indexing.exclude,
            "exclude_hidden_dirs": test_config.indexing.exclude_hidden_dirs,
        },
        "worker": {
            "progressive_snapshot_interval": test_config.worker.progressive_snapshot_interval,
            "progressive_snapshot_doc_count": test_config.worker.progressive_snapshot_doc_count,
        },
        "git_indexing": {
            "enabled": False,
        },
    }

    worker_process = Process(
        target=worker_main,
        args=(config_dict, command_queue, response_queue, shutdown_event, snapshot_base),
    )
    worker_process.start()

    try:
        init_notification = response_queue.get(timeout=30)
        assert isinstance(init_notification, InitCompleteNotification)
        assert init_notification.doc_count == 3
        initial_version = init_notification.version

        await asyncio.sleep(1)

        for i in range(3, 10):
            doc_file = docs_path / f"doc_{i:03d}.md"
            doc_file.write_text(f"# Document {i}\n\nContent for document {i}.\n")

        await asyncio.sleep(test_config.worker.progressive_snapshot_interval + 1)

        notifications = []
        try:
            while True:
                notif = response_queue.get(timeout=1)
                if isinstance(notif, IndexUpdatedNotification):
                    notifications.append(notif)
        except Exception:
            pass

        assert len(notifications) >= 1

        for notif in notifications:
            assert notif.version > initial_version

        final_notification = notifications[-1] if notifications else None
        if final_notification:
            assert final_notification.doc_count == 10

    finally:
        shutdown_event.set()
        worker_process.join(timeout=10)
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=2)
