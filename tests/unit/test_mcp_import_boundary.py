import subprocess
import sys


def test_proxy_import_does_not_load_daemon_search_modules():
    script = """
import sys
import mcp_markdown_ragdocs.mcp.server

assert "mcp_markdown_ragdocs.mcp.tools.document_tools" not in sys.modules
assert "mcp_markdown_ragdocs.daemon.management" not in sys.modules
assert "searchkernel.api" not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
