import time
from unittest.mock import MagicMock

import pytest

from src.config import Config, IndexingConfig, WorkerConfig, SearchConfig, LLMConfig, ServerConfig, ChunkingConfig
from src.worker.process import WorkerState, _should_publish_snapshot


@pytest.fixture
def worker_state(tmp_path):
    config = Config(
        server=ServerConfig(),
        indexing=IndexingConfig(
            documents_path=str(tmp_path / "docs"),
            index_path=str(tmp_path / "index"),
        ),
        worker=WorkerConfig(
            progressive_snapshot_interval=5.0,
            progressive_snapshot_doc_count=10,
        ),
        search=SearchConfig(),
        llm=LLMConfig(),
        document_chunking=ChunkingConfig(),
        memory_chunking=ChunkingConfig(),
    )

    mock_vector = MagicMock()
    mock_keyword = MagicMock()
    mock_graph = MagicMock()
    mock_index_manager = MagicMock()
    mock_sync_publisher = MagicMock()
    mock_command_queue = MagicMock()
    mock_response_queue = MagicMock()
    mock_shutdown_event = MagicMock()

    state = WorkerState(
        config=config,
        vector=mock_vector,
        keyword=mock_keyword,
        graph=mock_graph,
        index_manager=mock_index_manager,
        sync_publisher=mock_sync_publisher,
        command_queue=mock_command_queue,
        response_queue=mock_response_queue,
        shutdown_event=mock_shutdown_event,
    )

    return state


def test_should_publish_on_empty_queue_with_new_sync(worker_state):
    worker_state.last_index_time = time.time() - 10

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert _should_publish_snapshot(worker_state, pending_count=0, last_sync=last_sync)


def test_should_not_publish_on_empty_queue_without_new_sync(worker_state):
    worker_state.last_index_time = time.time()

    from datetime import datetime, timezone
    last_sync_dt = datetime.fromtimestamp(worker_state.last_index_time - 10, tz=timezone.utc)
    last_sync = last_sync_dt.isoformat()

    assert not _should_publish_snapshot(worker_state, pending_count=0, last_sync=last_sync)


def test_should_not_publish_on_empty_queue_when_no_last_sync(worker_state):
    assert not _should_publish_snapshot(worker_state, pending_count=0, last_sync=None)


def test_should_publish_after_time_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 6

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_publish_after_doc_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 1
    worker_state.docs_indexed_since_publish = 11

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_not_publish_before_thresholds(worker_state):
    worker_state.last_publish_time = time.time() - 1
    worker_state.docs_indexed_since_publish = 5

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert not _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_not_publish_when_no_previous_publish_and_queue_not_empty(worker_state):
    worker_state.last_publish_time = None

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert not _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_publish_exactly_at_time_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 5.0

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_publish_exactly_at_doc_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 1
    worker_state.docs_indexed_since_publish = 10

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_not_publish_just_below_time_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 4.9
    worker_state.docs_indexed_since_publish = 5

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert not _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_should_not_publish_just_below_doc_threshold(worker_state):
    worker_state.last_publish_time = time.time() - 1
    worker_state.docs_indexed_since_publish = 9

    from datetime import datetime, timezone
    last_sync = datetime.now(timezone.utc).isoformat()

    assert not _should_publish_snapshot(worker_state, pending_count=10, last_sync=last_sync)


def test_config_defaults():
    config = WorkerConfig()
    assert config.progressive_snapshot_interval == 5.0
    assert config.progressive_snapshot_doc_count == 10
