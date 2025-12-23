# 3. Core Features (Hybrid Search)

## 3.1. MCP Tool: `query_documents`

The server will expose a single, powerful MCP tool that leverages a hybrid search engine.

-   **Tool Name:** `query_documents`
-   **Description:** "Queries a local knowledge base of Markdown documents using a hybrid search engine (semantic, keyword, graph, and recency) to answer a question. Use this for complex questions that involve finding specific terms, related concepts, or recently edited notes."
-   **Input:**
    -   `query` (string, required): The natural language question or topic to search for.
-   **Output:**
    -   A string containing the answer synthesized by the LLM based on the documents retrieved by the hybrid search engine.

### Example Interaction

> **LLM:** "In the new auth spec, what does the `getToken` function do?"
>
> **MCP Host:** (Calls `query_documents(query="In the new auth spec, what does the getToken function do?")`)
>
> **MCP Server:** (Receives query. The Query Orchestrator runs parallel searches:
>   - **Keyword search** finds documents with the literal string `getToken`.
>   - **Semantic search** finds documents conceptually related to "new auth spec".
>   - **Recency bias** boosts the score of recently edited spec files.
>   - The results are fused, and the most relevant chunks from the top-ranked documents are passed to the LLM.)
>
> **MCP Host:** (Returns the answer string to the LLM) "The `getToken` function in the `auth-spec.md` document is responsible for exchanging a temporary code for a permanent JWT..."

**Note:** The server will NOT expose any tools for indexing, managing files, or configuration. These are considered internal, autonomous functions.

## 3.2. Automatic Multi-Index Sync

This is the cornerstone feature of the server. When a file is changed, the server automatically parses it and updates all relevant indices to keep the search engine's knowledge perfectly in sync.

-   **On Startup:** The server will scan the configured `documents_path` and run the full indexing pipeline for each file to build the initial Vector Index, Keyword Index, and Graph Store.
-   **Real-time Updates:** The server uses `watchdog` to monitor the `documents_path`. File events are debounced and processed in batches as described below.

### 3.2.1. Indexing Pipeline

For each created or modified file in a batch, the following pipeline is executed by the appropriate parser.

#### 3.2.1.1. Markdown Parsing Details
The `MarkdownParser` will use `tree-sitter` to perform a detailed analysis of the file content, with special attention to syntax commonly used in Obsidian.

1.  **Parse YAML Frontmatter:** The parser will first look for a YAML frontmatter block at the beginning of the file. It will extract key-value pairs, with a special focus on the `aliases` key. These aliases will be stored as metadata for the document node, making the note searchable by alternative names.
2.  **Parse Content and Extract Links:** The body of the document is then parsed. The parser will identify:
    -   **Standard Wikilinks (`[[Note Name]]` or `[[Note Name|Display Text]]`):** Each link is treated as a directed edge to be added to the NetworkX graph.
    -   **Transclusions (`![[Note Name]]`):** These are treated as a special type of "embeds" edge in the graph. The content is **not** inlined during this step to avoid data duplication in the index, but the relationship is recorded. This allows for more advanced queries, like "show me all notes that embed this one."
    -   **Tags (`#some-tag`):** Tags are extracted and added to the document's metadata.

#### 3.2.1.2. Index Updates
The structured data extracted from the parsing step is then used to update the three core indices:

1.  **Update Vector Index:** The clean text content of the document is sent to LlamaIndex to update the Faiss vector store.
2.  **Update Keyword Index:** The text content and key metadata (like aliases) are sent to Whoosh to update the BM25 keyword index.
3.  **Update Graph Store:** The extracted links, aliases, and transclusions are used to add or update nodes and edges in the NetworkX graph.

For a deleted file, it is removed from all three indices. This entire process runs in the background, ensuring the API server remains responsive.

### 3.2.2. Event Debouncing

To prevent excessive re-indexing from rapid file changes, a debouncing mechanism is used.

1.  **Queueing:** The `watchdog` watcher places file events (path and event type) onto an `asyncio.Queue`.
2.  **Cooldown & Batching:** An asynchronous processor waits for a short cooldown period after receiving an event. It consolidates multiple events for the same file and processes a unique batch of file operations once the queue is quiet.
3.  **Execution:** The processor runs the full **Indexing Pipeline** described above for the files in the batch.
4.  **Error Handling:** If a file fails at any point during the pipeline (e.g., due to a parsing error from malformed content), the error will be logged and reported via the `/status` endpoint. No complex retry logic is implemented; the system will simply attempt to process the file again the next time it is modified by the user, allowing for natural, event-driven recovery.

This approach balances responsiveness with efficiency, ensuring the indices remain up-to-date without wasting system resources.

## 3.3. Configuration Management

The server will be configurable via a TOML file, but will provide strong defaults to work out-of-the-box for the most common use case.

- **Configuration File:** `config.toml`
- **Lookup Order:**
  1. A project-local `config.toml` in the current working directory.
  2. A user-global config at `$HOME/.config/mcp-markdown-ragdocs/config.toml`.
- **Defaults:** If no configuration file is found, the server will operate with sensible defaults (e.g., watch the current directory `./`, run on port 8000).
- **Key Options:** See `specs/05-configuration.md` for a full breakdown of the configuration options. The most important will be:
  - `documents_path`: The directory containing the Markdown files.
  - `server_host`: The IP address to bind the server to.
  - `server_port`: The port to run the server on.
  - `index_path`: Where to store the persisted vector index.
