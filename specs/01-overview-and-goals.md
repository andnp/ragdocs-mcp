# 1. Overview and Goals

## 1.1. Project Vision

To create a highly autonomous, efficient, and developer-friendly MCP (Multi-turn Conversational Platform) server specialized for performing Retrieval-Augmented Generation (RAG) on a local repository of Markdown documents.

The server will operate with minimal configuration and strong defaults, abstracting away complexities like database management and API key handling. It is designed to be a "set it and forget it" tool for developers and technical users who want to query a local knowledge base using natural language.

## 1.2. Core Problem

Developers and users often have a local collection of documentation, notes, and other knowledge stored in Markdown files. Accessing this information efficiently can be cumbersome, typically relying on manual searching or `grep`. A conversational AI agent with access to this knowledge base can provide a much more intuitive and powerful interface.

Existing RAG solutions often require manual setup of vector databases, explicit indexing steps, and ongoing maintenance. This project aims to eliminate that friction.

## 1.3. Key Goals & Principles

- **Zero-Friction Setup:** The user should be able to start the server with a single command without any prior setup, and it should "just work" on a target directory of Markdown files.
- **High Autonomy:** The server must automatically detect and index file changes (creations, modifications, deletions) without any user or LLM intervention. The indexing process should be entirely transparent.
- **Strong Defaults:** The system will use sensible defaults for embedding models, vector stores, and other parameters, while still allowing for customization via a clear configuration file.
- **Stateless & Local:** The server should be entirely self-contained, using the local file system for its index. No external database services are required.
- **Robustness:** The server will be built with comprehensive test coverage, including unit, integration, end-to-end, and performance tests, to ensure reliability.

## 1.4. User Stories

- **As a developer,** I want to point the MCP server at my project's `docs/` directory so I can ask an LLM questions about the project's architecture and get answers based on the documentation.
- **As a knowledge worker,** I want to run the server on my Obsidian vault so I can have a conversational agent help me connect ideas and find information across my personal notes.
- **As a user,** I want the server to automatically update its knowledge when I add a new `.md` file to my notes folder, without needing to restart or manually re-index.
- **As an operator,** I want to configure the server using a simple TOML file to specify the document directory and server port, without needing to understand the underlying RAG pipeline.
- **As a developer,** I want the tool to be simple and not expose unnecessary complexity, so it should not have functions like `index_documents`. The indexing should be a background process.
