import sys
from types import ModuleType

from mcp_markdown_ragdocs import entrypoint


def test_mcp_dispatch_bypasses_full_cli(monkeypatch):
    observed: dict[str, list[str]] = {}
    mcp_module = ModuleType("mcp_markdown_ragdocs.mcp.server")

    async def fake_mcp_main(argv):
        observed["argv"] = argv

    mcp_module.main = fake_mcp_main
    monkeypatch.setitem(sys.modules, "mcp_markdown_ragdocs.mcp.server", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp_markdown_ragdocs.cli", None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mcp-markdown-ragdocs", "mcp", "--project", "docs"],
    )

    entrypoint.main()

    assert observed["argv"] == ["--project", "docs"]
