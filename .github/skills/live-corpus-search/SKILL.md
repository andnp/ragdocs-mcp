---
name: live-corpus-search
description: 'Validate search changes against the live Ragdocs corpus. Use when testing the installed daemon after changing search, indexing, or searchkernel behavior.'
---

# Live Corpus Search

## When to Use
- Verify a local search or indexing change against the real multi-project corpus.
- Confirm that the globally installed Ragdocs daemon is running the current checkout.
- Compare JSON query results before and after a search change.

## Procedure
1. From the Ragdocs repository, refresh the global editable tool:
   ```bash
   uv tool install --editable --force /home/andy/Projects/personal/mcp-markdown-ragdocs
   ```
   Confirm that the install resolves the expected `andnp-searchkernel` version.

2. Restart the global daemon with enough time for the corpus to initialize:
   ```bash
   mcp-markdown-ragdocs daemon restart --timeout 120
   ```
   The default 10-second timeout is often too short for the live corpus.

3. Wait for readiness before querying:
   ```bash
   mcp-markdown-ragdocs daemon status
   mcp-markdown-ragdocs index stats
   ```
   Continue only when daemon status is `running` and index stats report `Index state: ready` with `Remaining estimate: 0`.
   If restart times out, inspect status again; the daemon may still be initializing. Use the daemon log at:
   `/home/andy/.local/state/mcp-markdown-ragdocs/daemon/daemon.log`.

4. Run queries one at a time. Do not launch concurrent CLI queries while the daemon is starting or rebuilding:
   ```bash
   mcp-markdown-ragdocs query "your query" --top-n 5 --json
   ```
   Use `--project-filter mcp-markdown-ragdocs` when evaluating the current repository's documents instead of the whole global corpus:
   ```bash
   mcp-markdown-ragdocs query "your query" \
     --project-filter mcp-markdown-ragdocs --top-n 5 --json
   ```

5. Summarize results using `file_path`, `header_path`, `score`, and
   `provenance.strategies`. Include both positive queries and an unrelated
   query when checking relevance and abstention behavior:
   ```bash
   ... --json | jq -r \
     '.results[] | [.file_path, .header_path, .score,
       (.provenance.strategies | join(","))] | @tsv'
   ```

## Operational Notes
- The daemon is global; `daemon restart` operates on
  `~/.local/state/mcp-markdown-ragdocs/daemon`.
- Querying without `--project-filter` searches all configured document roots and
  indexed Git history, which can outrank a relevant document from the current
  repository.
- A relevant result appearing in the corpus does not prove it is ranked first.
  Record the query, scope, top results, scores, and retrieval strategies.
- The CLI does not expose every search policy parameter. Use the MCP/API
  surface when testing thresholds or other request-level filters.
