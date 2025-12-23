# 4. Testing Strategy (Expanded)

A robust testing strategy is crucial to ensure the server is reliable, performs well, and correctly handles all features of the hybrid search and management system.

## 4.1. Test Directory Structure

```
tests/
├── e2e/
│   ├── test_cli.py
│   └── test_server_e2e.py
├── integration/
│   ├── test_file_watching.py
│   ├── test_hybrid_search.py
│   └── test_index_lifecycle.py
├── performance/
│   ├── test_indexing_speed.py
│   └── test_query_latency.py
└── unit/
    ├── test_config.py
    └── test_parsers.py
```

## 4.2. Unit Tests (`tests/unit/`)

-   **Goal:** To test individual, isolated components.
-   **Scope:**
    -   **Configuration Loading:** Test loading from all paths, default application, and validation.
    -   **Parsers (`test_parsers.py`):**
        -   Test the `MarkdownParser`'s ability to correctly extract aliases from YAML frontmatter.
        -   Test its ability to differentiate between standard `[[wikilinks]]` and `![[transclusions]]`.
        -   Test edge cases like malformed frontmatter or broken links.
    -   **Result Fusion Logic:** Unit test the RRF implementation with mock ranked lists.

## 4.3. Integration Tests (`tests/integration/`)

-   **Goal:** To test how different components work together.
-   **Scope:**
    -   **File Watching & Multi-Index Sync (`test_file_watching.py`):**
        -   Verify that on a file change, all three indices (Vector, Keyword, and Graph) are updated correctly. For example, after adding a file with a new link, assert that the graph store reflects this new edge.
    -   **Hybrid Search & Fusion (`test_hybrid_search.py`):**
        -   Create a small, controlled set of documents.
        -   Run a query designed to trigger results from multiple searchers (e.g., a query with a specific keyword that is also conceptually related to another document).
        -   Assert that the final fused results are ranked correctly according to the RRF strategy.
    -   **Index Lifecycle (`test_index_lifecycle.py`):**
        -   Test the index versioning system. Start a server to create an index, then stop it.
        -   Modify the "version" in the `index.manifest.json` file.
        -   Restart the server and assert that it detects the version mismatch and triggers a full re-index (this can be checked via logs or a status flag).

## 4.4. End-to-End (E2E) Tests (`tests/e2e/`)

-   **Goal:** To test the complete, running application from an external perspective.
-   **Scope:**
    -   **Full Server Lifecycle (`test_server_e2e.py`):**
        -   Start the server as a subprocess.
        -   Check the `/health` and `/status` endpoints to ensure they return correct and well-formed data.
        -   Make calls to the `/query_documents` MCP endpoint and verify plausible responses.
        -   Simulate file changes and re-query to ensure the knowledge base was updated.
    -   **CLI (`test_cli.py`):**
        -   Test the `check-config` command and assert that it prints the expected configuration.
        -   Test the `rebuild-index` command. This will involve creating an index, deleting a source file manually (to make the index stale), and then running the command and asserting that the deleted file is no longer in any of the indices.

## 4.5. Performance Tests (`tests/performance/`)

-   **Goal:** To benchmark key performance indicators.
-   **Scope:**
    -   **Indexing Speed:** Benchmark the initial build time for a large corpus, as this now involves multiple parsers and indices.
    -   **Query Latency:** Measure the average end-to-end query time, from HTTP request to HTTP response.
