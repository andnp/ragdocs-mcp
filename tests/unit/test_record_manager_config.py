from mcp_markdown_ragdocs.config import EmbeddingConfig
from mcp_markdown_ragdocs.indexing.record_manager import RecordIndexManager
from tests.conftest import make_test_config


def test_record_manager_passes_embedding_batch_size_to_ingestor(
    tmp_path,
    local_record_kernel,
    deterministic_embedding_provider,
) -> None:
    config = make_test_config(
        tmp_path,
        embedding=EmbeddingConfig(batch_size=7),
    )

    manager = RecordIndexManager(
        config,
        local_record_kernel,
        deterministic_embedding_provider,
    )

    assert manager.ingestor.embedding_batch_size == 7
