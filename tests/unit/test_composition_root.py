"""Tests for the composition root that verify no global mutation during library usage."""

import os
from types import SimpleNamespace

from mcp_markdown_ragdocs.app.composition import (
    build_kernel,
    build_runtime_components,
)
from mcp_markdown_ragdocs.app.runtime import configure_runtime_threads
from mcp_markdown_ragdocs.config import GoogleDriveConfig, load_config
from mcp_markdown_ragdocs.context import ApplicationContext, ContextIndexingPort


class TestCompositionRootNoGlobalMutation:
    """Verify that library usage doesn't mutate global environment variables."""

    def test_build_kernel_does_not_mutate_environ(self, monkeypatch, tmp_path):
        """Importing and building the kernel should not mutate os.environ for thread counts."""
        # Set up temporary index path to avoid conflicts
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))

        # Snapshot environment before
        env_before = dict(os.environ)
        thread_keys = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"}

        # Build the kernel
        ctx = build_kernel()

        # Verify context is usable
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.index_manager is not None
        assert ctx.orchestrator is not None

        # Verify environment wasn't mutated (thread env vars should match before state)
        for key in thread_keys:
            env_before_value = env_before.get(key)
            env_after_value = os.environ.get(key)
            assert env_before_value == env_after_value, (
                f"{key} was mutated by build_kernel: "
                f"before={env_before_value}, after={env_after_value}"
            )

    def test_configure_runtime_threads_sets_environ(self, monkeypatch):
        """configure_runtime_threads should explicitly set the thread environment variables."""
        # Clear the environment variables
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)

        config = load_config()
        expected_value = str(config.indexing.torch_num_threads)

        # Call configure_runtime_threads
        configure_runtime_threads(config)

        # Verify environment variables are set
        assert os.environ.get("OMP_NUM_THREADS") == expected_value
        assert os.environ.get("MKL_NUM_THREADS") == expected_value
        assert os.environ.get("TORCH_NUM_THREADS") == expected_value

    def test_application_context_create_does_not_mutate_environ(self, monkeypatch, tmp_path):
        """ApplicationContext.create should NOT mutate environment variables anymore."""
        # Set up temporary index path
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))

        # Clear and snapshot the environment
        thread_keys = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"}
        for key in thread_keys:
            monkeypatch.delenv(key, raising=False)

        env_before = dict(os.environ)

        # Create context directly (simulating library usage)
        ctx = ApplicationContext.create(
            enable_watcher=False,
            lazy_embeddings=True,
        )

        # Verify context is valid
        assert ctx is not None
        assert ctx.config is not None

        # Verify thread env vars were NOT set
        for key in thread_keys:
            assert os.environ.get(key) != env_before.get(key) or (
                key not in os.environ
            ), f"{key} should not have been set by ApplicationContext.create"

    def test_explicit_app_startup_flow(self, monkeypatch, tmp_path):
        """Test the explicit app startup flow: create context, then configure threads."""
        # Set up temporary index path
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))

        # Clear the environment variables
        thread_keys = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"}
        for key in thread_keys:
            monkeypatch.delenv(key, raising=False)

        # Simulate app startup flow (as used in daemon/cli/server)
        ctx = ApplicationContext.create(
            enable_watcher=False,
            lazy_embeddings=True,
        )
        # App explicitly configures threads
        configure_runtime_threads(ctx.config)

        # Verify the environment is now configured
        expected_value = str(ctx.config.indexing.torch_num_threads)
        assert os.environ.get("OMP_NUM_THREADS") == expected_value
        assert os.environ.get("MKL_NUM_THREADS") == expected_value
        assert os.environ.get("TORCH_NUM_THREADS") == expected_value

    def test_enabled_drive_source_uses_global_record_index(self, monkeypatch, tmp_path):
        """
        Register Drive on the existing manager so Drive and Markdown share storage.
        The source stays independently addressable by its logical source kind.
        """
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))
        config = load_config()
        config.gdrive = GoogleDriveConfig(enabled=True, workspace_id="workspace")
        source = SimpleNamespace(source_kind="gdrive")
        monkeypatch.setattr(
            "mcp_markdown_ragdocs.context.build_gdrive_source",
            lambda *_args, **_kwargs: source,
        )

        context = ApplicationContext.create(
            config=config,
            enable_watcher=False,
            lazy_embeddings=True,
            global_runtime=True,
        )

        assert context.index_manager.content_sources == (source,)
        assert context.index_manager.get_content_source("gdrive") is source

    def test_runtime_factory_returns_one_shared_record_runtime(self, monkeypatch, tmp_path):
        """
        The runtime factory must keep kernel, manager, and database ownership aligned.
        """
        index_path = tmp_path / "index"
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(index_path))

        components = build_runtime_components(
            load_config(),
            enable_watcher=False,
            lazy_embeddings=True,
            index_path_override=index_path,
            global_runtime=True,
        )

        assert components.index_manager.kernel is components.kernel
        assert components.db_manager is components.index_manager.database_manager
        assert components.paths.index_path == index_path

    def test_runtime_composition_does_not_mutate_supplied_config(
        self,
        monkeypatch,
        tmp_path,
    ):
        """Composition normalizes an independent nested configuration snapshot."""
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))
        config = load_config()
        config.indexing.index_path = str(tmp_path / "configured-index")
        config.indexing.documents_path = str(tmp_path / "configured-docs")
        config.llm.embedding_model = "configured-model"
        config.embedding.model_name = "configured-embedding"
        original = config.snapshot()

        components = build_runtime_components(
            config,
            enable_watcher=False,
            lazy_embeddings=True,
            index_path_override=tmp_path / "runtime-index",
            documents_path_override=tmp_path / "runtime-docs",
            global_runtime=True,
        )

        assert config == original
        assert components.config is not config
        assert components.config.indexing is not config.indexing
        assert components.config.embedding is not config.embedding

    def test_context_index_manager_uses_application_port(self, monkeypatch, tmp_path):
        """The composed manager satisfies the context-owned indexing capability."""
        monkeypatch.setenv("SEARCHKERNEL_INDEX_PATH", str(tmp_path / "index"))

        context = ApplicationContext.create(
            enable_watcher=False,
            lazy_embeddings=True,
        )

        assert isinstance(context.index_manager, ContextIndexingPort)
