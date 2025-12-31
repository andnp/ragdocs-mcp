# NLP/RAG Techniques

This document explains the information retrieval and NLP techniques used in mcp-markdown-ragdocs. Each section covers what a technique is, why it matters, how it works, how this codebase implements it, and relevant trade-offs.

**Target audience:** Technical practitioners familiar with general machine learning (optimization, neural networks, embeddings as vectors) but not necessarily familiar with NLP/LLM/RAG-specific concepts.

---

## 1. Semantic Search with Embeddings

### What it is

Semantic search retrieves documents based on meaning rather than exact word matches. It converts text into dense vector representations (embeddings) and finds similar vectors using distance metrics.

### The problem it solves

Keyword search fails when users phrase queries differently than documents. Searching for "how to authenticate" misses documents titled "login process" or "credential management" because the exact words don't match.

### How it works

1. **Embedding model**: A neural network (transformer) encodes text into a fixed-length vector. Similar meanings produce similar vectors. This codebase uses `BAAI/bge-small-en-v1.5`, which outputs 384-dimensional vectors.

2. **Vector similarity**: Given two vectors $\mathbf{a}$ and $\mathbf{b}$, cosine similarity measures their angular distance:

$$
\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

Values range from -1 (opposite) to 1 (identical direction). For normalized vectors, this equals the dot product.

3. **FAISS indexing**: Facebook AI Similarity Search (FAISS) provides efficient nearest-neighbor lookup. This codebase uses `IndexFlatL2` (brute-force L2 distance), which is exact but scales linearly with corpus size.

### How we use it

From [src/indices/vector.py](../src/indices/vector.py):

```python
# Embedding generation (via HuggingFace)
self._embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embedding = self._embedding_model.get_text_embedding(text)

# Search returns top-k nearest neighbors
retriever = self._index.as_retriever(similarity_top_k=top_k)
nodes = retriever.retrieve(query)
```

Documents are chunked before embedding. Each chunk is embedded independently, and search returns chunk IDs ranked by similarity.

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | Embedding inference dominates query time (~50-100ms for embedding generation) |
| Storage | 384 floats × 4 bytes = 1.5KB per chunk |
| Model choice | Larger models (e.g., `bge-large`) improve quality but increase latency 3-5x |
| Failure modes | Semantic drift: "Java" (programming language) vs "Java" (island) have similar embeddings without context |
| Tuning | No hyperparameters beyond model selection; quality depends on embedding model |

---

## 2. BM25 and Keyword Search

### What it is

BM25 (Best Matching 25) is a probabilistic ranking function for keyword-based retrieval. It scores documents based on term frequency, document length, and corpus statistics.

### The problem it solves

Semantic search struggles with exact identifiers. Searching for "getToken" may not find documents containing that function name if the embedding model doesn't recognize it as a single concept. Keyword search provides exact term matching.

### How it works

**TF-IDF intuition**: Terms appearing frequently in a document but rarely in the corpus are strong signals. BM25 refines this with saturation (diminishing returns for repeated terms) and length normalization.

**BM25 scoring formula** (high-level):

$$
\text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

Where:
- $f(t, D)$ = frequency of term $t$ in document $D$
- $|D|$ = document length
- $\text{avgdl}$ = average document length in corpus
- $k_1$ = term frequency saturation parameter (typically 1.2-2.0)
- $b$ = length normalization parameter (typically 0.75)
- $\text{IDF}(t)$ = inverse document frequency of term $t$

**Field boosting (BM25F)**: Different document fields can have different importance. A match in the title is more significant than a match in the body.

### How we use it

From [src/indices/keyword.py](../src/indices/keyword.py):

```python
# Schema with field boosts
self._schema = Schema(
    title=TEXT(analyzer=stem_analyzer, field_boost=3.0),
    headers=TEXT(analyzer=stem_analyzer, field_boost=2.5),
    keywords=TEXT(analyzer=stem_analyzer, field_boost=2.5),
    description=TEXT(analyzer=stem_analyzer, field_boost=2.0),
    tags=KEYWORD(field_boost=2.0),
    aliases=TEXT(analyzer=stem_analyzer, field_boost=1.5),
    author=TEXT(analyzer=stem_analyzer, field_boost=1.0),
    content=TEXT(analyzer=stem_analyzer),  # No boost (1.0)
)

# Search with BM25F scoring
searcher = self._index.searcher(weighting=BM25F())
```

Field boost values indicate relative importance:
- Title match (3.0x): Strongest signal—query matches document topic
- Headers/keywords (2.5x): Section headings and explicit keywords
- Description/tags (2.0x): Metadata summaries
- Aliases (1.5x): Alternative names
- Content (1.0x): Body text baseline

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | Fast (~10-20ms) via inverted index |
| Storage | ~500 bytes per document |
| When it beats semantic | Exact identifiers (function names, error codes), technical jargon |
| When it fails | Synonyms ("auth" vs "authentication"), conceptual queries |
| Tuning | Field boosts, $k_1$, $b$ parameters |

---

## 3. Reciprocal Rank Fusion (RRF)

### What it is

RRF combines multiple ranked lists into a single ranking without requiring score normalization. It uses rank positions rather than raw scores.

### The problem it solves

Semantic search returns cosine similarities (0.0-1.0). BM25 returns unbounded scores (e.g., 15.3, 42.7). Combining these directly is problematic because:
1. Scales differ (0.95 vs 42.7)
2. Score distributions differ (BM25 has long tails)
3. A "good" score in one system may be mediocre in another

Normalizing scores (min-max, z-score) assumes score distributions are comparable, which often isn't true.

### How it works

RRF converts scores to ranks and applies a simple formula:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

Where:
- $d$ = document
- $R$ = set of ranked lists (semantic, keyword, graph)
- $\text{rank}_r(d)$ = position (1-indexed) of $d$ in list $r$
- $k$ = smoothing constant (default: 60)

**Why $k = 60$?** The constant determines how quickly scores decay with rank. Higher $k$ values make rankings more uniform; lower values amplify top positions. Research found $k = 60$ works well across domains.

**Example calculation:**

Document "auth.md" appears at:
- Semantic: rank 3
- Keyword: rank 5

```
RRF = 1/(60+3) + 1/(60+5) = 0.0159 + 0.0154 = 0.0313
```

Document "deploy.md" appears at:
- Semantic: rank 1

```
RRF = 1/(60+1) = 0.0164
```

Despite "deploy.md" ranking higher in semantic search, "auth.md" wins because it appears in both lists.

### How we use it

From [src/search/fusion.py](../src/search/fusion.py):

```python
def rrf_score(rank: int, k: int):
    return 1 / (k + rank)

def fuse_results(results: dict[str, list[str]], k: int, weights: dict[str, float], ...):
    scores: dict[str, float] = {}
    for strategy, doc_ids in results.items():
        weight = weights.get(strategy, 1.0)
        for rank, doc_id in enumerate(doc_ids):
            score = rrf_score(rank, k) * weight
            scores[doc_id] = scores.get(doc_id, 0.0) + score
    ...
```

We apply strategy weights to the RRF contribution:

```python
weighted_rrf = semantic_weight * (1/(k+rank_semantic)) + keyword_weight * (1/(k+rank_keyword))
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Simplicity | No ML training, no score calibration |
| Robustness | Works even when individual rankers return poor scores |
| Information loss | Ignores score magnitudes (rank 1 at 0.99 treated same as rank 1 at 0.51) |
| Tuning | $k$ constant, strategy weights |
| Failure modes | If one ranker returns all bad results, RRF still promotes its top items |

---

## 4. Query Expansion via Embeddings

### What it is

Query expansion augments user queries with semantically related terms extracted from the corpus. It bridges vocabulary mismatch between queries and documents.

### The problem it solves

Users often use different terminology than documents:
- Query: "auth" → Document contains: "authentication"
- Query: "db" → Document contains: "database"
- Query: "setup" → Document contains: "installation"

Keyword search fails because exact terms don't match. Semantic search may help, but short queries produce imprecise embeddings.

### How it works

1. **Build concept vocabulary**: During indexing, extract unique terms from all chunks. Embed each term using the same embedding model. Store term→embedding mapping.

2. **Expand at query time**: Embed the query. Find nearest terms in vocabulary via cosine similarity. Append high-similarity terms (≥ 0.5) to the query.

**Why embeddings enable this:** The embedding model learned semantic similarity. Terms with similar meanings (e.g., "auth" and "authentication") have similar vectors even though they're different strings.

### How we use it

From [src/indices/vector.py](../src/indices/vector.py):

```python
def build_concept_vocabulary(self, min_term_length: int = 3, max_terms: int = 10000):
    # Extract terms from all indexed chunks
    for chunk_id in self._chunk_id_to_node_id:
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
        for token in tokens:
            if len(token) >= min_term_length and token not in STOPWORDS:
                term_counts[token] = term_counts.get(token, 0) + 1

    # Embed top terms
    for term in top_terms:
        embedding = self._embedding_model.get_text_embedding(term)
        self._concept_vocabulary[term] = embedding

def expand_query(self, query: str, top_k: int = 3, similarity_threshold: float = 0.5):
    query_embedding = self._embedding_model.get_text_embedding(query)

    # Find nearest vocabulary terms
    for term, term_emb in self._concept_vocabulary.items():
        sim = cosine_similarity(query_embedding, term_emb)
        if sim >= similarity_threshold:
            similarities.append((term, sim))

    # Append top-3 terms not already in query
    expansion_terms = [term for term, _ in sorted(similarities)[:top_k]]
    return f"{query} {' '.join(expansion_terms)}"
```

**Example:**
```
Original: "database optimization"
Expansion terms: ["indexing", "query", "performance"]
Expanded: "database optimization indexing query performance"
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | Adds one embedding inference + vocabulary scan (~5-10ms) |
| Storage | ~1.5KB per vocabulary term (10,000 terms ≈ 15MB) |
| Quality | Depends on corpus coverage—domain-specific terms may not be well-represented |
| Failure modes | May expand to unrelated terms if similarity threshold too low |
| Tuning | `similarity_threshold` (0.5), `top_k` (3), vocabulary size |

---

## 5. Cross-Encoder Re-Ranking

### What it is

Cross-encoder re-ranking applies a more accurate (but slower) model to re-score the top candidates from initial retrieval. It computes query-document relevance jointly rather than independently.

### The problem it solves

**Bi-encoder limitation:** In semantic search, queries and documents are embedded independently. The model never sees them together, limiting its ability to capture query-document interactions.

```
Bi-encoder:    query → embed → vector_q
               doc   → embed → vector_d
               score = similarity(vector_q, vector_d)

Cross-encoder: (query, doc) → model → relevance_score
```

Cross-encoders see both texts simultaneously, enabling attention across query and document tokens. This captures nuances like negation, specificity, and context that bi-encoders miss.

### How it works

1. **Initial retrieval**: Fast methods (semantic + keyword) return top-N candidates (~100 results in ~100ms)

2. **Re-ranking**: Cross-encoder scores each (query, candidate) pair
   - Input: Concatenated query and document text
   - Output: Single relevance score
   - Model: MS-MARCO trained cross-encoders (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)

3. **Final ranking**: Sort by cross-encoder scores, return top results

**Why MS-MARCO models?** MS-MARCO is a large-scale passage retrieval dataset. Models trained on it generalize well to document search tasks.

### How we use it

From [src/search/reranker.py](../src/search/reranker.py):

```python
class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # Lazy loading

    def rerank(self, query: str, candidates: list[tuple[str, float]], ...):
        # Prepare (query, content) pairs
        query_content_pairs = [(query, content) for _, _, content in pairs]

        # Batch score all pairs
        scores = self._model.predict(query_content_pairs)

        # Sort by cross-encoder scores
        reranked = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return reranked[:top_n]
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | ~50ms for 10 candidates, ~5ms per additional candidate |
| Accuracy | Significantly better than bi-encoders for fine-grained relevance |
| When to apply | Only on top-N candidates (not full corpus—too slow) |
| Model options | `ms-marco-MiniLM-L-6-v2` (22MB, recommended), `TinyBERT` (17MB, faster), `bge-reranker-base` (110MB, higher quality) |
| Failure modes | Slower than skipped; may reorder already-good rankings incorrectly if model disagrees with domain |
| Tuning | `rerank_top_n` (how many candidates to re-rank) |

---

## 6. Semantic Deduplication

### What it is

Semantic deduplication clusters chunks with high embedding similarity and returns only one representative per cluster. It reduces redundancy in chunked retrieval.

### The problem it solves

Chunking splits documents into overlapping pieces. A relevant section may appear across multiple chunks:

```
Chunk 1: "...OAuth tokens are used for authentication. Refresh tokens..."
Chunk 2: "...Refresh tokens allow obtaining new access tokens without..."
Chunk 3: "...without re-authenticating. Token expiration is configured..."
```

All three chunks may score highly for query "OAuth token refresh". Returning all three wastes context window space with redundant information.

### How it works

1. **Compute pairwise similarity**: For each candidate chunk, retrieve its embedding

2. **Greedy clustering**: Process chunks in score order. For each chunk:
   - If similar (cosine ≥ threshold) to any already-kept chunk, skip it
   - Otherwise, keep it as a new cluster representative

3. **Threshold selection**: Default 0.85 means chunks must be 85% similar to be considered duplicates

**Why greedy?** Processing in score order ensures the highest-scoring chunk becomes the cluster representative. More sophisticated clustering (e.g., agglomerative) adds complexity without clear benefit for this use case.

### How we use it

From [src/search/dedup.py](../src/search/dedup.py):

```python
def deduplicate_by_similarity(
    results: list[tuple[str, float]],
    get_embedding: Callable[[str], list[float] | None],
    similarity_threshold: float = 0.85,
) -> tuple[list[tuple[str, float]], int]:

    kept: list[tuple[str, float]] = []
    removed: set[str] = set()
    clusters_merged = 0

    for chunk_id, score in results:  # Already sorted by score
        is_duplicate = False
        for kept_id, _ in kept:
            sim = cosine_similarity(embeddings[chunk_id], embeddings[kept_id])
            if sim >= similarity_threshold:
                is_duplicate = True
                clusters_merged += 1
                break

        if not is_duplicate:
            kept.append((chunk_id, score))

    return kept, clusters_merged
```

The pipeline also includes content-hash deduplication (exact text match) as a fast pre-filter before embedding-based deduplication.

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | O(n²) pairwise comparisons, but n is small (post-filtering candidates) |
| Information loss | May remove chunks with unique context even if embeddings are similar |
| Threshold tuning | 0.85 is conservative; 0.70 clusters more aggressively |
| Failure modes | Clusters unrelated chunks if embedding model conflates distinct concepts |
| Storage | No additional storage; uses existing chunk embeddings |

---

## 7. Heading-Weighted Embeddings

### What it is

Heading-weighted embeddings prepend hierarchical header context to chunk content before embedding. This preserves document structure in the vector representation.

### The problem it solves

Chunking loses context. Consider this document structure:

```markdown
# OAuth Guide
## Token Management
### Refresh Tokens
Tokens can be refreshed using the /refresh endpoint...
```

If we chunk at "Tokens can be refreshed...", the embedding captures "refresh endpoint" but loses "OAuth", "Token Management", and "Refresh Tokens" context. A query for "OAuth token refresh" may not find this chunk.

### How it works

Before embedding, prepend the header path to chunk content:

```
Original chunk: "Tokens can be refreshed using the /refresh endpoint..."

With header path: "OAuth Guide > Token Management > Refresh Tokens

Tokens can be refreshed using the /refresh endpoint..."
```

The embedding model now encodes "OAuth", "Token", "Management", and "Refresh" into the vector, improving retrieval for hierarchical document queries.

### How we use it

From [src/indices/vector.py](../src/indices/vector.py):

```python
def add_chunk(self, chunk: Chunk) -> None:
    # Prepend header_path to content for embedding
    embedding_text = f"{chunk.header_path}\n\n{chunk.content}" if chunk.header_path else chunk.content

    llama_doc = LlamaDocument(
        text=embedding_text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "header_path": chunk.header_path,
            ...
        },
    )
    self._index.insert_nodes([llama_doc])
```

The header path is extracted during parsing and stored in chunk metadata. The format uses " > " as a separator (e.g., "OAuth Guide > Token Management > Refresh Tokens").

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | No additional latency; header prepending happens during indexing |
| Storage | Slightly larger embeddings text, but vector size unchanged (384 dims) |
| Quality improvement | Significant for hierarchical documents; minimal for flat documents |
| Failure modes | Very long header paths may dilute chunk content in the embedding |
| Tuning | Header path format, whether to include all ancestor headers or just immediate parent |

---

## 8. Graph-Based Retrieval

### What it is

Graph-based retrieval uses document relationships (links) to surface structurally related content. Documents form a graph where edges represent wikilinks or transclusions.

### The problem it solves

Semantic and keyword search find documents matching the query. But related documents that don't match may still be valuable:

- Query: "deployment configuration"
- Semantic match: "deployment.md"
- Linked but no match: "security.md" (linked from deployment.md)

The user asking about deployment may also need security information, even if they didn't explicitly query for it.

### How it works

1. **Graph construction**: During indexing, extract wikilinks (`[[Target]]`) and build a directed graph. Nodes are document IDs; edges are links.

2. **1-hop neighbor expansion**: After semantic/keyword retrieval, look up neighbors of candidate documents in the graph.

3. **Score boosting**: Add neighbors to the result pool with reduced weight (0.5x multiplier). This prevents graph results from dominating query-matched results.

**Why 1-hop only?** Multi-hop traversal (2+) risks topic drift. A document two links away may be tangentially related. 1-hop provides immediate context without excessive scope expansion.

### How we use it

From [src/indices/graph.py](../src/indices/graph.py):

```python
def get_neighbors(self, doc_id: str, depth: int = 1):
    neighbors = set()
    current_level = {doc_id}

    for _ in range(depth):
        next_level = set()
        for node in current_level:
            successors = set(self._graph.successors(node))
            predecessors = set(self._graph.predecessors(node))
            next_level.update(successors | predecessors)
        neighbors.update(next_level)
        current_level = next_level

    neighbors.discard(doc_id)
    return list(neighbors)
```

From [src/search/orchestrator.py](../src/search/orchestrator.py):

```python
# Get 1-hop neighbors for all candidate documents
graph_neighbors = self._get_graph_neighbors(list(all_doc_ids))

# Add to result pool with reduced weight
results_dict = {
    "semantic": [...],
    "keyword": [...],
    "graph": graph_chunk_ids,  # Weighted at 0.5x in RRF
}
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | ~5-10ms for in-memory NetworkX operations |
| Storage | ~200 bytes per node + ~100 bytes per edge |
| Quality | Depends on link quality; poor linking provides poor expansion |
| Failure modes | Densely linked documents (hub nodes) may flood results |
| Tuning | Depth (1-hop vs 2-hop), neighbor weight multiplier |

---

## 9. Query Type Classification

### What it is

Query type classification detects user intent (factual, navigational, exploratory) and adjusts search strategy weights accordingly. It uses heuristic pattern matching rather than ML models.

### The problem it solves

Different query types benefit from different retrieval strategies:
- "getUserById function" → Needs exact keyword matching
- "How do I configure authentication?" → Needs semantic understanding
- "See the deployment guide" → Needs structure/navigation awareness

A one-size-fits-all weighting scheme underperforms compared to adaptive weights.

### How it works

**Classification Hierarchy:** Factual → Navigational → Exploratory (first match wins)

**Signals:**

| Query Type | Detection Signals | Weight Adjustment |
|------------|-------------------|-------------------|
| Factual | camelCase, snake_case, backticks, version numbers, quoted phrases | keyword × 1.5 |
| Navigational | "section", "guide", "docs", wikilink patterns `[[...]]` | graph × 1.5 |
| Exploratory | Question words (what, how, why), question mark | semantic × 1.3 |

### How we use it

From [src/search/classifier.py](../src/search/classifier.py):

```python
class QueryType(Enum):
    FACTUAL = "factual"
    NAVIGATIONAL = "navigational"
    EXPLORATORY = "exploratory"

def classify_query(query: str) -> QueryType:
    if _CAMEL_CASE_PATTERN.search(query) or _SNAKE_CASE_PATTERN.search(query):
        return QueryType.FACTUAL
    if any(kw in query.lower() for kw in _NAVIGATIONAL_KEYWORDS):
        return QueryType.NAVIGATIONAL
    if query.strip().endswith("?") or any(w in query.lower().split() for w in _QUESTION_WORDS):
        return QueryType.EXPLORATORY
    return QueryType.EXPLORATORY  # Default

def get_adaptive_weights(query_type: QueryType, base_weights: dict) -> dict:
    multipliers = {
        QueryType.FACTUAL: {"keyword": 1.5},
        QueryType.NAVIGATIONAL: {"graph": 1.5},
        QueryType.EXPLORATORY: {"semantic": 1.3},
    }
    ...
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | Negligible (~0.1ms regex matching) |
| Accuracy | Heuristics work for common patterns; edge cases may misclassify |
| Failure modes | Short queries without clear signals default to exploratory |
| Tuning | Multiplier values, keyword lists, regex patterns |
| Dependencies | None (pure Python regex) |

---

## 10. Code-Aware Tokenization

### What it is

Code-aware tokenization splits programming identifiers (camelCase, snake_case) into component terms for better keyword search matching.

### The problem it solves

Standard tokenizers treat `getUserById` as a single token. Searching for "user" or "id" won't match. Code-aware tokenization produces: `["getUserById", "get", "user", "by", "id"]`, enabling partial matches.

### How it works

**Splitting Rules:**

| Input | Standard | Code-Aware |
|-------|----------|------------|
| `getUserById` | `getuserbyid` | `getUserById`, `get`, `User`, `By`, `Id` |
| `parse_json_data` | `parse_json_data` | `parse_json_data`, `parse`, `json`, `data` |
| `HTTPResponse` | `httpresponse` | `HTTPResponse`, `HTTP`, `Response` |

**Implementation:**

```python
class CamelCaseSplitter(Filter):
    def __call__(self, tokens):
        for token in tokens:
            yield token  # Original token
            # Split on case transitions: aB or ABc
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)', text)
            for part in parts:
                yield Token(text=part.lower())

class SnakeCaseSplitter(Filter):
    def __call__(self, tokens):
        for token in tokens:
            yield token  # Original token
            for part in text.split("_"):
                if part:
                    yield Token(text=part.lower())
```

### How we use it

From [src/indices/code.py](../src/indices/code.py):

```python
def _create_code_analyzer():
    return RegexTokenizer(expression=_CODE_TOKEN_PATTERN) | \
           CamelCaseSplitter() | \
           SnakeCaseSplitter() | \
           LowercaseFilter()
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Index size | ~2-3x more tokens per code block |
| Latency | Minimal overhead during indexing |
| Recall improvement | Significant for code-heavy documentation |
| Precision | May match unrelated identifiers sharing subwords |
| Tuning | Token regex pattern, which splitters to apply |

---

## 11. Maximal Marginal Relevance (MMR)

### What it is

MMR is a diversity-aware selection algorithm that balances relevance to the query with novelty compared to already-selected results.

### The problem it solves

Top search results often cluster around similar content. Returning 5 chunks that all say essentially the same thing wastes context window space. MMR ensures diverse coverage of the topic.

### How it works

**MMR Formula:**

$$
\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)
$$

Where:
- $d$ = candidate document
- $q$ = query
- $S$ = already-selected documents
- $\lambda$ = relevance/diversity trade-off (0.0–1.0)

**Greedy Selection:**
1. Select highest-relevance document first
2. For each subsequent slot:
   - Score all remaining candidates using MMR formula
   - Select the one with highest MMR score
   - Add to selected set

**Lambda interpretation:**
- $\lambda = 1.0$: Pure relevance (no diversity)
- $\lambda = 0.7$: Balanced (default)
- $\lambda = 0.5$: Equal weight
- $\lambda = 0.3$: Diversity-focused

### How we use it

From [src/search/diversity.py](../src/search/diversity.py):

```python
def select_mmr(
    candidates: list[tuple[str, float]],
    get_embedding: Callable[[str], list[float] | None],
    query_embedding: list[float],
    top_n: int = 5,
    lambda_param: float = 0.7,
) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        best_mmr = -float("inf")
        best_idx = 0

        for i, (chunk_id, relevance) in enumerate(remaining):
            max_sim_to_selected = max(
                cosine_similarity(emb, selected_emb)
                for _, selected_emb in selected_embeddings
            ) if selected else 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | O(n × k) where n = candidates, k = selected; ~10-20ms typical |
| Quality | Reduces redundancy; improves coverage of subtopics |
| When to use | Result clustering is problematic; broad coverage needed |
| Failure modes | Low $\lambda$ may surface less relevant but diverse results |
| Tuning | `mmr_lambda` (0.5–0.9), `top_n` |

---

## 12. N-gram Overlap Deduplication

### What it is

N-gram deduplication uses character n-grams and Jaccard similarity to detect near-duplicate text without requiring embeddings.

### The problem it solves

Embedding-based deduplication is expensive (~0.5ms per comparison). For obvious duplicates (copy-paste with minor edits), cheaper detection suffices. N-gram dedup acts as a fast pre-filter.

### How it works

**N-gram extraction:**
```
Text: "Configure authentication"
3-grams: {"Con", "onf", "nfi", "fig", "igu", "gur", "ure", "re ", "e a", " au", "aut", "uth", ...}
```

**Jaccard similarity:**

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Two texts with Jaccard similarity ≥ threshold (default 0.7) are considered duplicates.

**Why Jaccard works:**
- Edit distance is O(n²); Jaccard is O(n) with hashing
- Character n-grams capture local text patterns
- Robust to word reordering within the n-gram window

### How we use it

From [src/search/dedup.py](../src/search/dedup.py):

```python
def get_ngrams(text: str, n: int = 3) -> set[str]:
    text = text.lower()
    return {text[i : i + n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

def deduplicate_by_ngram(
    results: list[tuple[str, float]],
    get_content: Callable[[str], str | None],
    threshold: float = 0.7,
    n: int = 3,
) -> tuple[list[tuple[str, float]], int]:
    kept: list[tuple[str, float]] = []
    kept_ngrams: list[set[str]] = []

    for chunk_id, score in results:
        content = get_content(chunk_id)
        ngrams = get_ngrams(content, n)

        is_dup = any(
            jaccard_similarity(ngrams, kept_ng) >= threshold
            for kept_ng in kept_ngrams
        )

        if not is_dup:
            kept.append((chunk_id, score))
            kept_ngrams.append(ngrams)

    return kept, len(results) - len(kept)
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Latency | ~0.1ms per comparison (vs ~0.5ms for embeddings) |
| Accuracy | Catches obvious duplicates; misses semantic duplicates |
| When to use | Before embedding-based dedup as a pre-filter |
| Failure modes | May cluster unrelated short texts; misses paraphrases |
| Tuning | `ngram_dedup_threshold` (0.6–0.8), n-gram size (3–5) |

---

## 13. Parent Document Retrieval

### What it is

Parent document retrieval decouples the retrieval unit (small chunks) from the return unit (larger parent sections). Small chunks enable precise matching; parent sections provide sufficient context for LLM consumption.

### The problem it solves

**The chunking dilemma:**
- Small chunks (~500 chars): Better retrieval precision, but insufficient context
- Large chunks (~2000 chars): More context, but less precise matching

Parent retrieval achieves both: match on small chunks, return their parent sections.

### How it works

**Two-level chunking:**

```
Document
├── Parent Section 1 (1500-2000 chars) ← Return unit
│   ├── Child Chunk 1.1 (200-500 chars) ← Retrieval unit
│   └── Child Chunk 1.2 (200-500 chars)
├── Parent Section 2
│   ├── Child Chunk 2.1
│   ├── Child Chunk 2.2
│   └── Child Chunk 2.3
└── ...
```

**Process:**
1. During indexing: Split documents into parent sections, then split parents into child chunks
2. Store `parent_chunk_id` in child chunk metadata
3. Embed and index children only (smaller, more precise)
4. At query time: Search finds matching children
5. Expansion: Map children to parents, deduplicate parents, return parent content

### How we use it

From [src/chunking/header_chunker.py](../src/chunking/header_chunker.py):

```python
def _create_parent_child_chunks(
    self,
    parent_chunks: list[Chunk],
    config: ChunkingConfig,
) -> list[Chunk]:
    result: list[Chunk] = []

    for parent in parent_chunks:
        # Split parent into smaller children
        children = self._split_into_children(parent, config)

        for i, child in enumerate(children):
            child.parent_chunk_id = parent.chunk_id
            result.append(child)

        # Also include parent for direct retrieval
        result.append(parent)

    return result
```

From [src/search/orchestrator.py](../src/search/orchestrator.py):

```python
def _expand_to_parents(
    self,
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    seen_parents: set[str] = set()
    expanded: list[tuple[str, float]] = []

    for chunk_id, score in results:
        parent_id = self._get_parent_id(chunk_id)
        if parent_id:
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                expanded.append((parent_id, score))
        else:
            expanded.append((chunk_id, score))

    return expanded
```

### Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Index size | ~1.5-2x more chunks (children + parents) |
| Latency | Minimal overhead for parent lookup |
| Context quality | Significant improvement for LLM consumption |
| When to use | Long documents with clear section structure |
| Failure modes | Poorly structured docs may have unbalanced parent sizes |
| Tuning | `parent_chunk_min_chars`, `parent_chunk_max_chars` |

---

## Processing Pipeline Summary

The complete search pipeline applies these techniques in order:

```
Query Type Classification (if adaptive_weights_enabled)
        ↓
       Query
        ↓
   Query Expansion
        ↓
[Semantic Search + Keyword Search + Code Search] (parallel)
        ↓
  Graph Neighbor Expansion
        ↓
      RRF Fusion
        ↓
    Recency Bias
        ↓
 Score Normalization
        ↓
Confidence Threshold
        ↓
Content Hash Dedup
        ↓
   N-gram Dedup (if ngram_dedup_enabled)
        ↓
  Semantic Dedup (if dedup_enabled)
        ↓
MMR Selection (if mmr_enabled) OR Per-Document Limit
        ↓
Cross-Encoder Re-Ranking (if rerank_enabled)
        ↓
Parent Expansion (if parent_retrieval_enabled)
        ↓
  Top-N Results
```

Each stage is configurable. See [configuration.md](configuration.md) for options.
