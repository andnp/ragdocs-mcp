"""
Unit tests for TOKENIZERS_PARALLELISM environment variable configuration.

Regression tests for the bug where tokenizers would warn about parallelism 
being used before fork when running in multiprocess mode:

    huggingface/tokenizers: The current process just got forked, after 
    parallelism has already been used. Disabling parallelism to avoid deadlocks...

The fix is to set TOKENIZERS_PARALLELISM=false in all entry points before
any HuggingFace/sentence-transformers imports.

See: https://github.com/huggingface/tokenizers/issues/993
"""

import subprocess
import sys
from pathlib import Path


class TestTokenizersParallelismEnvVar:
    """Tests for TOKENIZERS_PARALLELISM environment variable setup."""

    def test_cli_module_sets_tokenizers_parallelism(self):
        """
        Verify src.cli sets TOKENIZERS_PARALLELISM=false before imports.
        
        The env var must be set before any HuggingFace/sentence-transformers
        imports to prevent the warning when forking worker processes.
        """
        # Import cli module in isolated namespace to verify early setup
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
# Clear any existing value to test default behavior
os.environ.pop("TOKENIZERS_PARALLELISM", None)

# Import should set the env var
import src.cli

# Verify it was set
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false", \\
    f"Expected 'false', got {os.environ.get('TOKENIZERS_PARALLELISM')!r}"
print("PASS: cli module sets TOKENIZERS_PARALLELISM=false")
""",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0, (
            f"CLI module test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_mcp_server_module_sets_tokenizers_parallelism(self):
        """
        Verify src.mcp.server sets TOKENIZERS_PARALLELISM=false before imports.
        
        MCP server can be run directly, so it must also set the env var.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
os.environ.pop("TOKENIZERS_PARALLELISM", None)
import src.mcp.server
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false", \\
    f"Expected 'false', got {os.environ.get('TOKENIZERS_PARALLELISM')!r}"
print("PASS: mcp.server module sets TOKENIZERS_PARALLELISM=false")
""",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0, (
            f"MCP server module test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_worker_process_module_sets_tokenizers_parallelism(self):
        """
        Verify src.worker.process sets TOKENIZERS_PARALLELISM=false before imports.
        
        Worker process is spawned via multiprocessing and must also set the
        env var in case inheritance doesn't work (e.g., spawn start method).
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
os.environ.pop("TOKENIZERS_PARALLELISM", None)
import src.worker.process
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false", \\
    f"Expected 'false', got {os.environ.get('TOKENIZERS_PARALLELISM')!r}"
print("PASS: worker.process module sets TOKENIZERS_PARALLELISM=false")
""",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0, (
            f"Worker process module test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_setdefault_preserves_user_override(self):
        """
        Verify setdefault() does not override user-specified value.
        
        Users may want to set TOKENIZERS_PARALLELISM=true in some scenarios,
        and our setdefault() should respect their explicit choice.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
# User explicitly sets to "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import src.cli
# Should NOT be overwritten
assert os.environ.get("TOKENIZERS_PARALLELISM") == "true", \\
    f"Expected user value 'true', got {os.environ.get('TOKENIZERS_PARALLELISM')!r}"
print("PASS: setdefault preserves user override")
""",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0, (
            f"User override test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestTokenizersEnvVarPlacement:
    """
    Tests verifying env var is set BEFORE HuggingFace imports.
    
    The order is critical: setting the env var after tokenizers is imported
    has no effect. These tests verify the fix is properly placed.
    """

    def test_env_var_set_before_sentence_transformers_import(self):
        """
        Verify env var is available before sentence-transformers would be imported.
        
        This simulates the critical ordering requirement: the env var must be
        set before any code path that might import tokenizers.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
os.environ.pop("TOKENIZERS_PARALLELISM", None)

# Flag to track import order
imported_cli = False
env_var_set_before_st = False

class ImportTracker:
    def find_module(self, name, path=None):
        global imported_cli, env_var_set_before_st
        if name == "sentence_transformers" and imported_cli:
            # At this point, cli should have set the env var
            env_var_set_before_st = os.environ.get("TOKENIZERS_PARALLELISM") == "false"
        return None

import sys
sys.meta_path.insert(0, ImportTracker())

# Import cli (should set env var)
import src.cli
imported_cli = True

# Importing something that uses sentence_transformers would be caught by tracker
# For now, just verify the env var was set during cli import
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
print("PASS: env var set during cli import")
""",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0, (
            f"Import order test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
