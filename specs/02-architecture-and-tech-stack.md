# 2. Architecture and Technology Stack (Hybrid Search)

## 2.1. High-Level Architecture

The architecture is designed around a **hybrid search** model and a **pluggable parsing** system. It is composed of an **Indexing Service** that populates multiple data stores and a **Query Orchestrator** that fuses their results.

-   **Indexing Service:** Triggered by file changes, this service uses a **Parser Dispatcher** to select the correct parser for a given file type. The parser extracts structured data, which is then used to update three distinct indices: a Vector Index, a Keyword Index, and a Graph Store.
-   **Query Orchestrator:** Dispatches queries to the relevant searchers and applies a fusion algorithm (RRF) to produce a final, re-ranked list of documents for the LLM.

```
            ┌──────────────────┐
 File──────▶│  File Watcher    │
 Changes     │  (Watchdog)      │
            └─────────┬────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│                   Indexing Service                       │
│┌──────────────────┐ ┌────────────────┐ ┌─────────────────┐ │
││ Parser Dispatcher│▶│ Markdown Parser│▶│ Update Vector   │ │
││ (by file type)   │ │ (Tree-sitter)  │ │ Index (FAISS)   │ │
│└──────────────────┘ └───────┬────────┘ └──────┬──────────┘ │
│                             │                  │           │
│      ┌──────────────────────┘                  │           │
│      │                                         │           │
│      └──────────▶┌──────────────────┐ ┌────────▼─────────┐ │
│                  │ Update Keyword   │ │ Update Graph     │ │
│                  │ Index (Whoosh)   │ │ Store (NetworkX) │ │
│                  └──────────────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────────────┘

... Query Flow remains the same ...
```

## 2.2. Technology Stack

-   **Language:** Python 3.11+ / **Core Framework:** [Asyncio](https://docs.python.org/3/library/asyncio.html)
-   **MCP Server Framework:** [FastAPI](https://fastapi.tiangolo.com/)
-   **Pluggable Parsers:** The system will feature a pluggable architecture for document parsing.
    -   **Design:** An abstract `DocumentParser` base class will define the interface. Concrete implementations (`MarkdownParser`, `PlainTextParser`, etc.) will handle specific file types.
    -   **Default Parser:** The initial implementation will be a `MarkdownParser` using [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) to extract a rich AST (links, tags, frontmatter, etc.).
    -   **Grammar Management:** To ensure a zero-friction setup, tree-sitter's compiled language grammars will be managed as Python dependencies using a library like `py-tree-sitter-languages`. This automates the download and build process of the required grammars (e.g., for Markdown) during `uv pip install`.
-   **Hybrid Search Components:**
    -   **Semantic Search:** [LlamaIndex](https://www.llamaindex.ai/) with **Faiss**.
    -   **Keyword Search:** [Whoosh](https://whoosh.readthedocs.io/en/latest/index.html).
    -   **Graph Storage & Analysis:** [NetworkX](https://networkx.org/).
-   **File System Watching:** [Watchdog](https://github.com/gorakhargosh/watchdog).
-   **Dependency Management:** [uv](https://github.com/astral-sh/uv).

## 2.3. Data Flow

### 2.3.1. Indexing Flow (Background)

1.  **On Startup:** The server runs the full indexing pipeline for all discoverable files based on the configuration. It also checks the index version manifest and triggers a full rebuild if necessary.
2.  **File Change Event:** `watchdog` detects a file change and places it on the `asyncio.Queue`.
3.  **Debounce & Process:** The Indexing Service debounces and batches events.
4.  **For each file:**
    a.  **Parser Dispatch:** The service selects the appropriate parser from a registry based on the file's extension (e.g., `.md` -> `MarkdownParser`).
    b.  **Parsing:** The selected parser processes the file content and returns a structured `Document` object containing text, metadata, links, etc.
    c.  **Index Updates:** The data from the `Document` object is used to update all three indices: the Vector Index (LlamaIndex), the Keyword Index (Whoosh), and the Graph Store (NetworkX).
    d.  The indices and stores are persisted to disk.

### 2.3.2. Query Flow (Foreground)
*(This remains the same as the previous version: Request -> Orchestration -> Parallel Search -> Fusion -> Retrieval & Synthesis -> Response)*

## 2.4. Management and Observability

To ensure the system is transparent and maintainable, it includes several key features for observability and administration. These are detailed in `specs/07-observability-and-management.md` and include:
-   **Status Endpoints:** `/health` and `/status` for real-time monitoring.
-   **Structured Logging:** For clear, machine-readable log output.
-   **Index Versioning:** To automatically handle index rebuilds when configurations change.
-   **CLI:** For administrative tasks like forcing a re-index.