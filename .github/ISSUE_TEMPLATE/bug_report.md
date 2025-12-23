---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Environment

- **OS**: [e.g., Ubuntu 22.04, macOS 14.0, Windows 11]
- **Python Version**: [e.g., 3.13.1]
- **Project Version**: [e.g., 0.1.0]
- **Installation Method**: [e.g., uv, pip, Docker]

## Steps to Reproduce

1. Start server with `uv run mcp-markdown-ragdocs run`
2. Send query: `curl -X POST http://127.0.0.1:8000/query_documents -d '{"query": "..."}'`
3. Observe error in logs

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Logs

```
Paste relevant log output here.
Include server startup logs and error messages.
```

## Configuration

```toml
# Paste your config.toml (redact sensitive paths if needed)
[server]
host = "127.0.0.1"
port = 8000

[indexing]
documents_path = "./docs"
```

## Additional Context

- Number of documents indexed: [e.g., 150 markdown files]
- Index size: [e.g., 50MB]
- Any recent changes to the document collection
- Any relevant error screenshots
