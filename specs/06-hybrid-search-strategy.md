# 6. Hybrid Search Strategy

This document outlines a multi-faceted hybrid search strategy to provide a highly flexible and intelligent search experience. The goal is to move beyond simple semantic search and combine multiple retrieval methods, each with unique strengths, to deliver more relevant and context-aware results.

## 6.1. Overview

The search engine will combine results from four distinct retrieval strategies:

1.  **Semantic Search (RAG):** For conceptual understanding.
2.  **Keyword Search:** For literal term matching (e.g., function names, specific phrases).
3.  **Graph Traversal:** For discovering structurally linked ideas (e.g., via `[[wikilinks]]`).
4.  **Recency Bias:** To boost the relevance of recently modified documents.

A **Query Orchestrator** will execute these searches in parallel and then use a **Reciprocal Rank Fusion (RRF)** algorithm to merge the results into a single, unified ranked list before passing it to the LLM for synthesis.

## 6.2. Structural Parsing with Tree-sitter

To enable graph and keyword search, we need a deep understanding of the structure of each Markdown file. We will use `tree-sitter` with a Markdown grammar for this.

-   **On Indexing:** Each file will be parsed into an Abstract Syntax Tree (AST).
-   **Metadata Extraction:** From the AST, we will extract:
    -   **Links:** `[[wikilinks]]` to build the knowledge graph.
    -   **Tags:** `#tags` to be used as metadata.
    -   **Headings:** To potentially break documents into smaller, more coherent chunks.
    -   **Code Blocks:** To be indexed for keyword search.
    -   **Frontmatter:** YAML frontmatter will be parsed for explicit metadata like creation dates or aliases.

## 6.3. Component Search Strategies

### 6.3.1. Semantic Search (Vector RAG)

-   **Technology:** `LlamaIndex` with a `Faiss` or `ChromaDB` vector store.
-   **Chunking Strategy:** To ensure high-quality retrieval, we will use a content-aware chunking strategy. The default implementation will use the `MarkdownNodeParser` from LlamaIndex, which intelligently splits documents based on their heading structure (`#`, `##`, etc.), keeping related paragraphs and sub-sections together within a single chunk.
-   **Process:** The resulting chunks are converted into vector embeddings. The user's query is also embedded, and a cosine similarity search retrieves the top-k most conceptually similar chunks.
-   **Strength:** Finds related concepts even if the wording is different.

### 6.3.2. Keyword Search

-   **Technology:** [Whoosh](https://whoosh.readthedocs.io/en/latest/index.html), a pure-Python full-text indexing library.
-   **Process:** A standard inverted index will be built from the text content of the documents using a BM25F scoring algorithm.
-   **Strength:** Finds documents containing exact, specific terms.

### 6.3.3. Graph Traversal

-   **Technology:** [NetworkX](https://networkx.org/) for in-memory graph representation and analysis.
-   **Algorithm: 1-Hop Neighbor Boosting.** The graph itself is not searched directly. Instead, it is used to augment the results from the other searchers.
    1.  After the initial keyword and semantic searches return a set of candidate documents, we perform a "1-hop" lookup in the NetworkX graph for each candidate.
    2.  This lookup finds all documents that are directly linked *from* the candidate and all documents that link *to* the candidate.
    3.  These "neighbor" documents are added to the pool of results given to the fusion algorithm, with a slightly lower score than the directly retrieved documents.
-   **Strength:** Surfaces structurally related content that might be missed by other searchers, enriching the context provided to the LLM.

### 6.3.4. Recency Bias

-   **Process:** This is not a search method itself, but a score multiplier applied during the fusion stage. It uses a simple and predictable tier-based system.
-   **Tiers:**
    -   Documents modified in the last **7 days** receive a **1.2x** score multiplier.
    -   Documents modified in the last **30 days** receive a **1.1x** score multiplier.
    -   Documents modified over 30 days ago receive no boost (1.0x).
-   **Strength:** Prioritizes fresh content in a simple, predictable manner.

## 6.4. Result Fusion

A simple but highly effective method for combining ranked lists from different search systems is **Reciprocal Rank Fusion (RRF)**.

-   **Process:**
    1.  Each of the search strategies produces a ranked list of document IDs.
    2.  For each document in each list, we calculate its RRF score.
    3.  The formula is `RRF_Score = 1 / (k + rank)`, where `rank` is the document's position in the list and `k` is a constant (e.g., 60) that dampens the effect of high ranks.
    4.  The final score for each document is the sum of its RRF scores from all the lists it appears in.
    5.  The documents are then re-sorted based on this final combined score.
-   **Advantage:** RRF avoids the need to normalize scores across different systems and has been shown to be very robust.
-   **Configuration:** We can introduce weights for each search strategy to allow the user to tune the search engine's behavior (e.g., prefer semantic search over keyword search).
