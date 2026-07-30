"""Unit tests for semantic checkpoint tracking."""

import tempfile
from pathlib import Path

from searchkernel.indexing.bootstrap_checkpoint import (
    CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    BootstrapCheckpoint,
    BootstrapFileStamp,
    get_semantic_completion_status,
    load_bootstrap_checkpoint,
    mark_semantic_work_completed,
    save_bootstrap_checkpoint,
)


class TestBootstrapCheckpointSemantic:
    """Test semantic state tracking in bootstrap checkpoint."""

    def test_checkpoint_preserves_semantic_encoder_namespace(self) -> None:
        checkpoint = BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="gen-1",
            complete=False,
            targets={},
            completed={},
            semantic_encoder_namespace="encoder-v1",
            semantic_completed={},
        )

        assert checkpoint.semantic_encoder_namespace == "encoder-v1"

    def test_checkpoint_preserves_semantic_completed(self) -> None:
        checkpoint = BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="gen-1",
            complete=False,
            targets={},
            completed={},
            semantic_encoder_namespace="encoder-v1",
            semantic_completed={"file-1": True, "file-2": False},
        )

        assert checkpoint.semantic_completed["file-1"] is True
        assert checkpoint.semantic_completed["file-2"] is False

    def test_checkpoint_to_dict_includes_semantic_fields(self) -> None:
        checkpoint = BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="gen-1",
            complete=False,
            targets={},
            completed={},
            semantic_encoder_namespace="encoder-v1",
            semantic_completed={"file-1": True},
        )

        data = checkpoint.to_dict()

        assert data["semantic_encoder_namespace"] == "encoder-v1"
        assert data["semantic_completed"]["file-1"] is True

    def test_checkpoint_default_semantic_values(self) -> None:
        checkpoint = BootstrapCheckpoint(
            schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
            generation="gen-1",
            complete=False,
            targets={},
            completed={},
        )

        assert checkpoint.semantic_encoder_namespace is None
        assert checkpoint.semantic_completed == {}


class TestMarkSemanticWorkCompleted:
    """Test semantic work completion tracking."""

    def test_mark_semantic_work_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            # Create initial checkpoint
            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={"file-1": BootstrapFileStamp("file-1", 1000, 100)},
                completed={},
                semantic_encoder_namespace=None,
                semantic_completed={},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            # Mark semantic work completed
            result = mark_semantic_work_completed(
                index_path, "encoder-v1", "file-1"
            )

            assert result is True

            # Verify checkpoint was updated
            updated = load_bootstrap_checkpoint(index_path)
            assert updated is not None
            assert updated.semantic_encoder_namespace == "encoder-v1"
            assert updated.semantic_completed["file-1"] is True

    def test_mark_semantic_work_returns_false_if_no_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            result = mark_semantic_work_completed(
                index_path, "encoder-v1", "file-1"
            )

            assert result is False

    def test_mark_semantic_work_resets_on_encoder_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            # Create checkpoint with encoder v1
            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={
                    "file-1": BootstrapFileStamp("file-1", 1000, 100),
                    "file-2": BootstrapFileStamp("file-2", 2000, 200),
                },
                completed={},
                semantic_encoder_namespace="encoder-v1",
                semantic_completed={"file-1": True},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            # Mark with different encoder
            result = mark_semantic_work_completed(
                index_path, "encoder-v2", "file-2"
            )

            assert result is True

            # Verify checkpoint reset semantic progress
            updated = load_bootstrap_checkpoint(index_path)
            assert updated is not None
            assert updated.semantic_encoder_namespace == "encoder-v2"
            # Only file-2 should be marked as completed
            assert updated.semantic_completed.get("file-1") is None
            assert updated.semantic_completed["file-2"] is True

    def test_mark_semantic_work_ignores_unknown_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={"file-1": BootstrapFileStamp("file-1", 1000, 100)},
                completed={},
                semantic_encoder_namespace=None,
                semantic_completed={},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            # Try to mark unknown file
            result = mark_semantic_work_completed(
                index_path, "encoder-v1", "unknown-file"
            )

            assert result is False


class TestGetSemanticCompletionStatus:
    """Test semantic completion status queries."""

    def test_get_semantic_completion_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={},
                completed={},
                semantic_encoder_namespace="encoder-v1",
                semantic_completed={"file-1": True, "file-2": False},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            status = get_semantic_completion_status(index_path, "encoder-v1")

            assert status["file-1"] is True
            assert status["file-2"] is False

    def test_get_semantic_completion_status_encoder_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={},
                completed={},
                semantic_encoder_namespace="encoder-v1",
                semantic_completed={"file-1": True},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            # Query with different encoder
            status = get_semantic_completion_status(index_path, "encoder-v2")

            assert status == {}

    def test_get_semantic_completion_status_no_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            status = get_semantic_completion_status(index_path, "encoder-v1")

            assert status == {}


class TestBootstrapCheckpointLoadWithSemantic:
    """Test loading checkpoints with semantic fields."""

    def test_load_checkpoint_with_semantic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={},
                completed={},
                semantic_encoder_namespace="encoder-v1",
                semantic_completed={"file-1": True},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            loaded = load_bootstrap_checkpoint(index_path)

            assert loaded is not None
            assert loaded.semantic_encoder_namespace == "encoder-v1"
            assert loaded.semantic_completed["file-1"] is True

    def test_load_checkpoint_without_semantic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)

            # Old checkpoint without semantic fields
            checkpoint = BootstrapCheckpoint(
                schema_version=CURRENT_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
                generation="gen-1",
                complete=False,
                targets={},
                completed={},
            )
            save_bootstrap_checkpoint(index_path, checkpoint)

            loaded = load_bootstrap_checkpoint(index_path)

            assert loaded is not None
            assert loaded.semantic_encoder_namespace is None
            assert loaded.semantic_completed == {}
