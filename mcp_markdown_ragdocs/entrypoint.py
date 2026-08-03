from __future__ import annotations

import asyncio
import sys


def main() -> None:
    if sys.argv[1:2] != ["mcp"]:
        from mcp_markdown_ragdocs.cli import main as cli_main

        cli_main()
        return

    from mcp_markdown_ragdocs.mcp.server import main as mcp_main

    try:
        asyncio.run(mcp_main(sys.argv[2:]))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
