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

### 6.4.1. Query Expansion via Concept Vocabulary

Before semantic search executes, the query is expanded using a concept vocabulary built during indexing.

**Process:**
1. During indexing, extract unique terms from all chunks (after stopword removal)
2. Embed each term using the same model as document chunks
3. Store term→embedding mapping as concept vocabulary
4. At query time, embed user query and find top-3 nearest terms via cosine similarity
5. Append expansion terms to query for semantic search

**Configuration:** Enabled by default. Vocabulary persisted as `concept_vocabulary.json` alongside index.

**Implementation:** [src/indices/vector.py](../src/indices/vector.py) `build_concept_vocabulary()`, `expand_query()`.

### 6.4.2. Cross-Encoder Re-Ranking

After RRF fusion produces candidates, an optional cross-encoder re-ranks results for improved precision.

**Process:**
1. Take top-N candidates from RRF fusion (default N=10)
2. Score each (query, chunk_content) pair using cross-encoder model
3. Re-sort candidates by cross-encoder score
4. Return top results

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB, ~50ms for 10 candidates on CPU).

**Trade-off:** Adds 50-100ms latency but improves ranking precision.

**Configuration:**
```toml
[search]
rerank_enabled = false          # Enable cross-encoder re-ranking
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_top_n = 10               # Max candidates to re-rank
```

**Implementation:** [src/search/reranker.py](../src/search/reranker.py).

### 6.4.3. Parameters: `top_k` vs `top_n`

The hybrid search system distinguishes between two limiting parameters:

-   **`top_k`** (internal): The number of candidate documents retrieved from each individual search strategy (semantic, keyword) before fusion. This parameter is **dynamically calculated** to ensure sufficient candidate diversity for fusion.
-   **`top_n`** (user-facing): The maximum number of results returned to the user after fusion and normalization. This is the parameter exposed through the API.

**Dynamic `top_k` Calculation:**

```
top_k = max(10, top_n * 2)
```

-   **Rationale:** The 2x multiplier accounts for overlap between search strategies. Different strategies (semantic, keyword, graph) often return overlapping documents. Retrieving `top_n * 2` candidates from each strategy ensures that after deduplication and fusion, at least `top_n` unique results remain.
-   **Minimum of 10:** Guarantees sufficient candidate diversity even when `top_n` is small (e.g., `top_n=1` or `top_n=3`). Prevents degraded search quality from insufficient candidate pool.
-   **Effect:** Users requesting `top_n=25` will trigger retrieval of `top_k=50` candidates from each strategy, ensuring 25 unique results can be returned after fusion.

**Implementation Location:** The dynamic calculation is performed in the query entry points (CLI and MCP server) before calling the orchestrator. See [src/cli.py](../src/cli.py) and [src/mcp_server.py](../src/mcp_server.py) for implementation details.

## 6.5. Behavioral Invariants

The following invariants hold across all query executions:

1.  **Result count constraint:** `len(results) <= top_n`
    -   The number of returned results never exceeds the user-requested `top_n` parameter.

2.  **Candidate pool constraint:** `len(results) <= top_k` per strategy
    -   Each individual search strategy retrieves at most `top_k` candidates before fusion.
    -   With dynamic calculation, this becomes: `len(results) <= max(10, top_n * 2)` per strategy.

3.  **Score normalization:** `all(0.0 <= score <= 1.0 for _, score in results)`
    -   All returned scores are normalized to the [0.0, 1.0] range after fusion.
    -   The highest-scoring result always receives a score of 1.0.

4.  **Descending score order:** `results[i].score >= results[i+1].score`
    -   Results are returned in descending order by normalized score.

5.  **Top_k scaling:** `top_k >= 10`
    -   The internal `top_k` parameter never falls below 10, ensuring minimum candidate diversity.
    -   When `top_n > 5`, the candidate pool scales proportionally: `top_k = top_n * 2`.

6.  **Deduplication:** Chunk IDs are unique within returned results.
    -   If the same chunk appears in multiple search strategies, it contributes to RRF fusion but appears only once in final results.
