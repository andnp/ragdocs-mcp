# 7. Observability and Management

To ensure the server is transparent, debuggable, and maintainable, a set of observability and management features will be implemented.

## 7.1. Administrative API Endpoints

The server will expose two non-MCP, read-only endpoints for monitoring its status.

### 7.1.1. Health Endpoint: `/health`

-   **Purpose:** A simple endpoint to confirm that the server process is running and the API is responsive.
-   **Method:** `GET`
-   **Response (Success):**
    -   Status Code: `200 OK`
    -   Body: `{"status": "ok"}`

### 7.1.2. Status Endpoint: `/status`

-   **Purpose:** To provide a detailed, real-time snapshot of the server's internal state.
-   **Method:** `GET`
-   **Response (Success):**
    -   Status Code: `200 OK`
    -   Body (JSON):
        ```json
        {
          "server_status": "running",
          "indexing_service": {
            "pending_queue_size": 5,
            "last_sync_time": "2025-12-22T14:30:00Z",
            "failed_files": [
              {
                "path": "path/to/corrupted.md",
                "error": "Invalid frontmatter format",
                "timestamp": "2025-12-22T14:25:10Z"
              }
            ]
          },
          "indices": {
            "document_count": 1257,
            "index_version": "1.1.0"
          }
        }
        ```

## 7.2. Structured Logging

-   **Strategy:** The server will use structured logging (outputting JSON objects) for all events. This allows for easy parsing, filtering, and analysis by external logging tools.
-   **Information Logged per Event:**
    -   `timestamp`
    -   `level` (e.g., `INFO`, `WARN`, `ERROR`)
    -   `message`
    -   `component` (e.g., `fastapi`, `indexing_service`, `file_watcher`)
    -   Additional context-specific fields (e.g., `file_path` for an indexing event).

## 7.3. Index Lifecycle Management

### 7.3.1. Index Versioning

-   **Problem:** Changes to the indexing logic, embedding model, or parsers can make existing indices on disk stale and incompatible.
-   **Solution:** The server will manage an `index.manifest.json` file inside the `index_path`.
    -   This file will store metadata about the conditions under which the index was built, e.g.:
        ```json
        {
          "index_spec_version": "1.1.0",
          "embedding_model": "all-MiniLM-L6-v2",
          "parsers": {
            "markdown": "0.5.2"
          }
        }
        ```
    -   On startup, the server compares its current configuration against the manifest. If there's a mismatch, it will log a warning and automatically trigger a full re-index of all documents to ensure data integrity.

### 7.3.2. Command-Line Interface (CLI)

-   **Purpose:** To provide a way for the user to perform administrative actions on the server and its index.
-   **Implementation:** The application will be runnable as a module (`python -m mcp_markdown_ragdocs`).
-   **Commands:**
    -   `run` (default): Starts the MCP server.
    -   `rebuild-index`: Triggers a full, unconditional rebuild of all indices from the source documents, ignoring the version manifest. This is useful for forcing a refresh.
    -   `check-config`: Loads and validates the configuration, printing the final resolved settings, then exits.
