# Hybrid Search Strategy

This document explains the hybrid search system, including each search strategy, the Reciprocal Rank Fusion algorithm, and performance characteristics.

## Overview

The query orchestrator combines eleven distinct retrieval and enhancement strategies:

1. **Query Type Classification**: Adaptive weight adjustment based on query intent (optional)
2. **Query Expansion**: Vocabulary-based expansion for improved recall
3. **Semantic Search**: Conceptual similarity via vector embeddings
4. **Keyword Search**: Exact term matching via BM25F scoring with field boosts
5. **Code Search**: Specialized code block retrieval with identifier-aware tokenization (optional)
6. **Graph Traversal**: Structural relationships via wikilinks
7. **Community Detection**: Louvain clustering with co-community score boosting
8. **Recency Bias**: Temporal relevance via file modification time
9. **Score-Aware Fusion**: Dynamic weight adjustment based on score variance
10. **Cross-Encoder Re-Ranking**: Joint query-document relevance scoring (optional)
11. **HyDE**: Hypothesis-driven embedding search (optional)

Results from retrieval strategies are fused using Reciprocal Rank Fusion (RRF), then filtered, deduplicated (n-gram and semantic), diversified (MMR), and optionally re-ranked to produce a final ranked list of document chunks.

## Search Strategies

### Query Type Classification (Optional)

**Purpose:** Automatically adjust search weights based on query intent detection.

**Technology:**
- Heuristic pattern matching (regex-based)
- Zero ML dependencies

**Query Types:**

| Type | Signals | Weight Adjustment |
|------|---------|-------------------|
| Factual | camelCase, snake_case, backticks, versions, quoted phrases | keyword × 1.5 |
| Navigational | "section", "guide", "docs", wikilinks (`[[...]]`) | graph × 1.5 |
| Exploratory | Question words (what, how, why), question mark | semantic × 1.3 |

**Detection Priority:** Factual → Navigational → Exploratory (first match wins).

**Pattern Examples:**

```python
# Factual signals
_CAMEL_CASE_PATTERN = re.compile(r'[a-z][A-Z]|[A-Z]{2,}[a-z]')
_SNAKE_CASE_PATTERN = re.compile(r'\b[a-z]+_[a-z_]+\b')
_BACKTICK_PATTERN = re.compile(r'`[^`]+`')
_VERSION_PATTERN = re.compile(r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b')

# Navigational keywords
_NAVIGATIONAL_KEYWORDS = {'section', 'chapter', 'guide', 'tutorial', 'documentation', ...}

# Exploratory signals
_QUESTION_WORDS = {'what', 'how', 'why', 'when', 'where', 'which', ...}
```

**Example:**

```
Query: "getUserById function"
→ Detected: FACTUAL (camelCase signal)
→ Weights: semantic=1.0, keyword=1.5, graph=1.0

Query: "How do I configure authentication?"
→ Detected: EXPLORATORY ("How" question word)
→ Weights: semantic=1.3, keyword=1.0, graph=1.0
```

**Configuration:**

```toml
[search]
adaptive_weights_enabled = true
```

**Code Reference:** [src/search/classifier.py](../src/search/classifier.py)

### 1. Semantic Search

**Purpose:** Find documents conceptually related to the query, even when exact terms do not match.

**Technology:**
- Embedding model: HuggingFace BAAI/bge-small-en-v1.5 (384 dimensions)
- Vector index: FAISS IndexFlatL2 (cosine similarity search)
- Chunking: LlamaIndex MarkdownNodeParser (512 characters, 50 overlap)

**Process:**
1. Embed query using same model as indexed documents
2. Perform cosine similarity search in FAISS index
3. Return top-k document IDs ranked by similarity score

**When Most Effective:**
- Queries with synonyms or paraphrased concepts
- Searching for "authentication methods" finds documents about "login systems" and "credential management"
- Broad conceptual queries where exact terminology is unknown

**Example:**

Query: "How do I secure my API?"

Semantic search finds:
- "authentication.md" (discusses credentials and tokens)
- "security.md" (covers HTTPS and rate limiting)
- "api-reference.md" (mentions security headers)

Even though the word "secure" may not appear in these documents, the semantic similarity of "API security" concepts ensures retrieval.

### 2. Keyword Search

**Purpose:** Find documents containing exact terms, phrases, or technical identifiers.

**Technology:**
- Full-text index: Whoosh with BM25F scoring
- Field boosts:
  - title: 3.0x
  - headers: 2.5x
  - keywords: 2.5x
  - description: 2.0x
  - tags: 2.0x
  - aliases: 1.5x
  - author: 1.0x
- Tokenization: StandardAnalyzer (strips punctuation, lowercases)

**Process:**
1. Parse query into terms
2. Look up terms in inverted index
3. Score documents using BM25F algorithm (term frequency, inverse document frequency)
4. Return top-k document IDs ranked by BM25 score

**When Most Effective:**
- Queries with specific function names, error codes, or technical terms
- Searching for "getToken" finds documents with that exact function name
- Queries with unique identifiers that would not have semantic similarity

**Example:**

Query: "getToken function implementation"

Keyword search finds:
- "authentication.md" (contains literal string "getToken")
- "api-reference.md" (documents the `getToken` endpoint)

Semantic search might miss these if "getToken" is a custom function name without semantic context.

**Limitation:**

StandardAnalyzer strips punctuation. Queries for "C++" or "Node.js" normalize to "c" and "node". Custom analyzer required to preserve special characters.

### Code Block Search (Optional)

**Purpose:** Find code snippets in documentation using identifier-aware tokenization that handles programming language conventions.

**Technology:**
- Whoosh with BM25F scoring
- Custom analyzer with CamelCase and snake_case splitting
- Preserves code identifier structure

**Tokenization:**

Standard tokenizers break code identifiers incorrectly:

| Input | Standard Tokenizer | Code Analyzer |
|-------|-------------------|---------------|
| `getUserById` | `getuserbyid` | `getUserById`, `get`, `user`, `by`, `id` |
| `parse_json_data` | `parse_json_data` | `parse_json_data`, `parse`, `json`, `data` |
| `HTTPResponseError` | `httpresponseerror` | `HTTPResponseError`, `HTTP`, `Response`, `Error` |

**Analyzer Pipeline:**

```python
# 1. RegexTokenizer: extract alphanumeric identifiers
_CODE_TOKEN_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+')

# 2. CamelCaseSplitter: split on case transitions
"getUserById" → ["getUserById", "get", "User", "By", "Id"]

# 3. SnakeCaseSplitter: split on underscores
"parse_json" → ["parse_json", "parse", "json"]
```

**Schema:**

```python
Schema(
    id=ID(stored=True, unique=True),
    doc_id=ID(stored=True),
    chunk_id=ID(stored=True),
    content=TEXT(stored=True, analyzer=code_analyzer),
    language=ID(stored=True),
)
```

**When Most Effective:**
- Searching for function names, class names, variable names
- Finding code examples with specific patterns
- Technical documentation with embedded code blocks

**Example:**

```
Query: "getUserById"
→ Code search finds: code blocks containing getUserById, get_user_by_id
→ Matches both camelCase and snake_case variants
```

**Configuration:**

```toml
[search]
code_search_enabled = true
code_search_weight = 1.0
```

**Code Reference:** [src/indices/code.py](../src/indices/code.py)

### 3. Graph Traversal

**Purpose:** Surface structurally related documents connected via wikilinks, even when query terms do not appear in linked documents.

**Technology:**
- Graph representation: NetworkX directed graph
- Nodes: Document IDs with metadata (title, tags, aliases)
- Edges: Wikilinks (`[[Target]]`) and transclusions (`![[Target]]`)

**Process:**
1. Execute semantic and keyword searches to get initial candidate documents
2. For each candidate, perform 1-hop neighbor lookup in graph
3. Retrieve all documents directly linked from or to the candidate
4. Add neighbors to result pool with reduced score (0.5x multiplier)

**When Most Effective:**
- Queries that should surface related documentation not explicitly mentioned
- User queries for "deployment" → graph traversal includes linked "security.md" and "configuration.md" even if "deployment" does not appear in those files
- Discovering supporting context for a specific topic

**Example:**

Query: "deployment configuration"

Keyword/semantic search finds:
- "deployment.md"

Graph traversal adds:
- "security.md" (linked from deployment.md via `[[security]]`)
- "authentication.md" (linked from security.md)
- "configuration.md" (linked from deployment.md)

These linked documents enrich the context provided to the LLM without requiring the user to explicitly query for them.

**Implementation Detail:**

Only 1-hop neighbors are retrieved (direct links). Multi-hop traversal (2+ hops) was avoided due to:
- Increased computation cost
- Risk of topic drift (documents too far removed from original query)
- Diminishing relevance returns

### 4. Recency Bias

**Purpose:** Prioritize recently modified documents to surface up-to-date information.

**Implementation:**

Tier-based score multiplier applied during fusion:

| Modification Time | Multiplier |
|-------------------|------------|
| Last 7 days       | 1.2x |
| Last 30 days      | 1.1x |
| Over 30 days      | 1.0x |

These multipliers are applied directly to the RRF score during fusion. The `recency_bias` config option (default 0.5) is not currently applied to these tiers in the implementation.

**When Most Effective:**
- Documentation that changes frequently (API specs, deployment guides)
- Personal notes where recent entries are more relevant
- Queries like "recent updates" or "what changed recently"

**Example:**

Query: "authentication changes"

Without recency bias:
- "authentication.md" (last modified 90 days ago)
- "api-reference.md" (last modified 180 days ago)

With recency bias (assuming "oauth-guide.md" modified 3 days ago):
- "oauth-guide.md" (1.1x boost)
- "authentication.md"
- "api-reference.md"

The recent OAuth guide surfaces higher due to recency boost.

**Configuration:**

Disable recency bias by setting `recency_bias = 0.0` in config.

### 5. Query Expansion

**Purpose:** Improve recall by appending semantically related terms to the query.

**Technology:**
- Concept vocabulary built during index persist
- Extracts unique terms from all indexed chunks
- Embeds each term using the same embedding model
- Vocabulary persisted as `concept_vocabulary.json`

**Process:**
1. Embed the query using the embedding model
2. Compute cosine similarity against all vocabulary terms
3. Select top-3 terms with similarity ≥ 0.5
4. Append non-duplicate terms to the original query

**Example:**

Original query: "database optimization"

Expansion terms found:
- "indexing" (similarity: 0.68)
- "query" (similarity: 0.62)
- "performance" (similarity: 0.58)

Expanded query: "database optimization indexing query performance"

**When Most Effective:**
- Short queries that benefit from related terminology
- Queries using general terms where specific jargon exists in the corpus
- Technical documentation with domain-specific vocabulary

**Configuration:**

Query expansion is automatic when concept vocabulary exists. Rebuild index to generate vocabulary: `uv run mcp-markdown-ragdocs rebuild-index`

### 6. Cross-Encoder Re-Ranking

**Purpose:** Improve precision by re-scoring candidates using joint query-document relevance.

**Technology:**
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB)
- Lazy loading: model downloaded and loaded on first rerank call
- Sentence-transformers CrossEncoder implementation

**Process:**
1. Take top candidates from fusion pipeline (after filtering/dedup)
2. Pair each candidate's content with the query
3. Score all pairs using cross-encoder
4. Sort by cross-encoder scores descending
5. Return top `rerank_top_n` results

**Performance:**
- ~50ms for 10 candidates on CPU
- ~30ms for TinyBERT variant
- ~150ms for larger BAAI/bge-reranker-base

**When Most Effective:**
- Queries where initial ranking has relevant results in wrong order
- Technical queries requiring precise term matching in context
- High-stakes searches where precision matters more than latency

**Model Options:**

| Model | Size | Latency | Quality |
|-------|------|---------|--------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22MB | ~50ms | Recommended |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 17MB | ~30ms | Faster |
| `BAAI/bge-reranker-base` | 110MB | ~150ms | Higher quality |

**Configuration:**

```toml
[search]
rerank_enabled = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_top_n = 10
```

### 7. Community Detection

**Purpose:** Boost results that belong to the same document cluster as highly-ranked documents.

**Technology:**
- Louvain algorithm (NetworkX `louvain_communities`)
- Runs on undirected graph conversion
- No external dependencies

**Process:**
1. During `GraphStore.persist()`, detect communities across all document nodes
2. Store community assignments in `communities.json`
3. During search, identify communities of top-ranked seed documents
4. Boost scores of other results in the same communities

**When Most Effective:**
- Queries touching well-connected documentation sections
- Discovering related documents not explicitly linked
- Knowledge bases with dense wikilink structures

**Example:**

If "auth.md" (community 3) ranks highest, other documents in community 3 (e.g., "login.md", "sessions.md") receive a 1.1× score boost even without explicit query match.

**Configuration:**

```toml
[search.advanced]
community_detection_enabled = true
community_boost_factor = 1.1
```

**Code Reference:** [src/search/community.py](../src/search/community.py)

### 8. Score-Aware Fusion

**Purpose:** Dynamically adjust strategy weights based on score variance per query.

**Technology:**
- Variance calculation on score distributions
- Weight reduction for low-variance (uncertain) results
- Implemented in `src/search/variance.py`

**How It Works:**

1. Compute variance of vector scores and keyword scores separately
2. If variance < threshold, the strategy produces "muddy" matches (all scores similar)
3. Reduce weight proportionally: `factor = max(min_factor, variance / threshold)`
4. Renormalize weights to maintain original sum

**Formula:**

$$
W_{adjusted} = W_{base} \times \max\left(W_{min}, \frac{\sigma^2}{\sigma^2_{threshold}}\right)
$$

**When Most Effective:**
- Queries where one strategy produces uniformly low scores
- Ambiguous queries that confuse one search strategy
- Balancing semantic vs keyword when confidence differs

**Configuration:**

```toml
[search.advanced]
dynamic_weights_enabled = true
variance_threshold = 0.1
```

**Code Reference:** [src/search/variance.py](../src/search/variance.py)

### 9. HyDE (Hypothetical Document Embeddings)

**Purpose:** Improve retrieval for vague queries by searching with a hypothetical answer.

**Technology:**
- Direct embedding of hypothesis text
- Same embedding model as document indexing (BAAI/bge-small-en-v1.5)
- Implemented in `src/search/hyde.py`

**Process:**
1. User (or AI) generates hypothesis describing expected documentation
2. Hypothesis is embedded directly (no query expansion)
3. Vector similarity search using hypothesis embedding
4. Returns documents matching the hypothetical description

**When Most Effective:**
- Vague queries like "How do I add a feature?"
- Queries where describing the answer is easier than forming the question
- AI assistants that can generate plausible documentation descriptions

**Example:**

User query: "How do I add a new tool?"

AI generates hypothesis: *"To add a new tool, modify src/mcp_server.py and add a Tool definition in list_tools(). Include name, description, and inputSchema..."*

HyDE search embeds this hypothesis and finds:
- `src/mcp_server.py` code documentation
- Tool registration examples
- MCP protocol documentation

**MCP Tool:**

```json
{
  "name": "search_with_hypothesis",
  "inputSchema": {
    "hypothesis": "string (required)",
    "top_n": "integer (optional, default: 5)"
  }
}
```

**Configuration:**

```toml
[search.advanced]
hyde_enabled = true
```

**Code Reference:** [src/search/hyde.py](../src/search/hyde.py)

## Reciprocal Rank Fusion (RRF)

### Algorithm

RRF is a rank-based fusion method that combines multiple ranked lists without requiring score normalization.

**Formula:**

For each document appearing in any ranked list:

```
rrf_score(d) = Σ (1 / (k + rank(d, list_i)))
```

Where:
- `d` is a document ID
- `k` is a constant (default 60)
- `rank(d, list_i)` is the rank (1-indexed) of document `d` in list `i`
- Sum is over all lists where `d` appears

**Example:**

Document "auth.md" appears in:
- Semantic search at rank 3
- Keyword search at rank 5
- Graph traversal at rank 2

With k=60:

```
rrf_score = 1/(60+3) + 1/(60+5) + 1/(60+2)
          = 1/63 + 1/65 + 1/62
          = 0.0159 + 0.0154 + 0.0161
          = 0.0474
```

Document "deploy.md" appears in:
- Semantic search at rank 1
- Graph traversal at rank 10

```
rrf_score = 1/(60+1) + 1/(60+10)
          = 1/61 + 1/70
          = 0.0164 + 0.0143
          = 0.0307
```

Despite "deploy.md" ranking higher in semantic search, "auth.md" has a higher RRF score due to appearing in more lists.

### Weighted RRF

Strategy weights multiply each term in the RRF sum:

```
weighted_rrf_score(d) = Σ (weight_i / (k + rank(d, list_i)))
```

**Example:**

With `semantic_weight = 1.2` and `keyword_weight = 0.8`:

```
rrf_score = (1.2 / (60+3)) + (0.8 / (60+5))
          = 0.0190 + 0.0123
          = 0.0313
```

This increases the influence of semantic search results relative to keyword search.

### Advantages of RRF

1. **No score normalization required**: Ranks are universal (1, 2, 3...), unlike scores which vary by algorithm (0.95 for cosine similarity, 12.3 for BM25)
2. **Robust to outliers**: A single very high score in one list does not dominate the fusion
3. **Rewards consensus**: Documents appearing in multiple lists are naturally prioritized
4. **Simple to implement**: No machine learning or training required

### Disadvantages of RRF

1. **Loses score magnitude information**: A rank 1 document with score 0.99 is treated the same as rank 1 with score 0.51
2. **Fixed k constant**: Requires tuning for optimal performance (typical range: 20-100)
3. **Assumes list independence**: Does not account for correlation between lists

## Result Fusion Process

### Processing Pipeline

The complete pipeline processes results in this order:

```
Query Type Classification (if adaptive_weights_enabled)
        ↓
Query Expansion (concept vocabulary)
        ↓
Semantic Search + Keyword Search + Code Search (parallel)
        ↓
Graph Neighbor Boosting (1-hop)
        ↓
Community Boosting (if community_detection_enabled)
        ↓
Score-Aware RRF Fusion (with dynamic weights if dynamic_weights_enabled)
        ↓
Recency Bias (tier-based multiplier)
        ↓
Score Calibration (sigmoid, [0.0, 1.0])
        ↓
Confidence Threshold (min_confidence)
        ↓
Content Hash Deduplication (exact text match)
        ↓
N-gram Deduplication (if ngram_dedup_enabled)
        ↓
Semantic Deduplication (if dedup_enabled)
        ↓
MMR Selection (if mmr_enabled) OR Per-Document Limit
        ↓
Cross-Encoder Re-Ranking (if rerank_enabled)
        ↓
Parent Expansion (if parent_retrieval_enabled)
        ↓
Top-N Selection
```

### Step-by-Step Fusion

1. **Execute parallel searches**:
   - Semantic search returns: `[doc1, doc2, doc3, ...]` (ranked)
   - Keyword search returns: `[doc2, doc5, doc1, ...]` (ranked)

2. **Apply graph traversal**:
   - For each candidate from step 1, lookup 1-hop neighbors
   - Add neighbors to pool with 0.5x multiplier

3. **Compute RRF scores**:
   - For each unique document across all lists, sum weighted RRF terms

4. **Apply recency bias**:
   - Multiply each document's RRF score by its recency tier multiplier

5. **Sort by final score**:
   - Return top-k document IDs in descending score order

### Example End-to-End Fusion

**Query:** "API authentication"

**Semantic search results:**
1. authentication.md (cosine similarity: 0.92)
2. security.md (0.85)
3. api-reference.md (0.78)

**Keyword search results:**
1. api-reference.md (BM25: 15.3)
2. authentication.md (12.7)
3. oauth-guide.md (8.4)

**Graph traversal (1-hop from candidates):**
- deployment.md (linked from security.md, score 0.5x)
- configuration.md (linked from authentication.md, score 0.5x)

**Recency data:**
- oauth-guide.md: modified 5 days ago (1.1x boost)
- Others: modified over 30 days ago (1.0x)

**RRF Calculation (k=60, semantic_weight=1.0, keyword_weight=1.0):**

authentication.md:
- Semantic rank 1: 1/(60+1) = 0.0164
- Keyword rank 2: 1/(60+2) = 0.0161
- **RRF Score:** 0.0325
- Recency: 1.0x (>30 days old)
- **Final: 0.0325** (0.0325 × 1.0)

api-reference.md:
- Semantic rank 3: 1/(60+3) = 0.0159
- Keyword rank 1: 1/(60+1) = 0.0164
- **RRF Score:** 0.0323
- Recency: 1.0x (>30 days old)
- **Final: 0.0323** (0.0323 × 1.0)

security.md:
- Semantic rank 2: 1/(60+2) = 0.0161
- **RRF Score:** 0.0161
- Recency: 1.0x (>30 days old)
- **Final: 0.0161** (0.0161 × 1.0)

oauth-guide.md:
- Keyword rank 3: 1/(60+3) = 0.0159
- **RRF Score:** 0.0159
- Recency: 1.1x (modified 5 days ago, <7 days tier)
- **Final: 0.0175** (0.0159 × 1.1)

deployment.md:
- Graph boost: 0.5 * (1/(60+1)) = 0.0082
- **RRF Score:** 0.0082
- Recency: 1.0x (>30 days old)
- **Final: 0.0082** (0.0082 × 1.0)

### Recency Boost Algorithm

**Application Timing:** Recency boost applied **after** RRF scoring, not during rank computation.

**Tier Structure:**

| Age Range | Multiplier | Purpose |
|-----------|------------|----------|
| ≤7 days | 1.2× | Recent updates highly relevant |
| ≤30 days | 1.1× | Recent but not brand new |
| >30 days | 1.0× | No boost for older content |

**Age Calculation:**
```python
age_days = (current_timestamp - document_modified_timestamp) / 86400
```

**Tier Selection Logic:**
```python
def get_recency_multiplier(modified_timestamp, current_timestamp):
    age_days = (current_timestamp - modified_timestamp) / 86400

    # Tiers sorted by age (earliest first)
    tiers = [(7, 1.2), (30, 1.1)]

    # First tier where age <= tier_days wins
    for max_days, multiplier in tiers:
        if age_days <= max_days:
            return multiplier

    # Default for documents older than all tiers
    return 1.0
```

**Complete Fusion Algorithm:**
```python
def fuse_results(results, k, weights, modified_times, current_time):
    # Step 1: RRF scoring
    scores = {}
    for strategy, doc_ids in results.items():
        weight = weights.get(strategy, 1.0)
        for rank, doc_id in enumerate(doc_ids):
            rrf_contribution = (1 / (k + rank)) * weight
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contribution

    # Step 2: Recency boosting (applied AFTER RRF)
    boosted = []
    for doc_id, rrf_score in scores.items():
        multiplier = 1.0  # default for >30 days
        if doc_id in modified_times:
            age_days = (current_time - modified_times[doc_id]) / 86400
            if age_days <= 7:
                multiplier = 1.2
            elif age_days <= 30:
                multiplier = 1.1

        final_score = rrf_score * multiplier
        boosted.append((doc_id, final_score))

    # Step 3: Sort by final score
    return sorted(boosted, key=lambda x: x[1], reverse=True)
```

**Why After RRF:**
- RRF score represents content relevance from multiple strategies
- Recency acts as secondary signal to surface recent updates
- Prevents fresh but irrelevant documents from dominating results
- Preserves semantic/keyword/graph consensus while adding temporal awareness

**Final Ranking:**
1. authentication.md (0.0325)
2. api-reference.md (0.0323)
3. oauth-guide.md (0.0175)
4. security.md (0.0161)
5. deployment.md (0.0082)

### Score Calibration

**Purpose:** Convert raw RRF+recency scores to absolute confidence scores representing match quality independent of result set size.

**Method:** Sigmoid calibration applied after RRF fusion and recency boosting.

**Formula:**

$$
\text{calibrated\_score} = \frac{1}{1 + e^{-s \cdot (r - t)}}
$$

Where:
- $r$ = raw RRF+recency score
- $t$ = threshold (default: 0.035)
- $s$ = steepness (default: 150.0)

**Threshold Parameter:** RRF score corresponding to 50% confidence. Raw scores above threshold map to >0.5 confidence, scores below map to <0.5 confidence.

**Steepness Parameter:** Controls sigmoid curve steepness. Higher values create sharper transitions between low and high confidence.

**Calibration vs Normalization:** Calibration (sigmoid) produces **absolute confidence** from raw RRF+recency scores. Legacy min-max normalization is deprecated and is not applied to final scores; any internal normalization is limited to per-strategy scaling and does not change output confidence values.

**Score Interpretation:**

| Calibrated Score | Interpretation | Raw RRF Range | Typical Conditions |
|-----------------|----------------|---------------|---------------------|
| >0.9 | Excellent match | >0.050 | Top rank, multiple strategies agree |
| 0.7-0.9 | Good match | 0.038-0.050 | Top-3 rank, 2+ strategies |
| 0.5-0.7 | Moderate match | 0.030-0.038 | Near threshold, single strategy |
| 0.3-0.5 | Weak match | 0.020-0.030 | Low rank, peripheral relevance |
| <0.3 | Noise | <0.020 | Should be filtered |

**Properties:**

1. **Absolute Confidence:** Same raw score produces same calibrated score across queries
2. **Asymptotic Bounds:** Approaches 1.0 for high scores (~0.98 max), approaches 0.0 for low scores
3. **No Artificial Inflation:** Single-result queries scored by absolute confidence, not always 1.0
4. **Stable Semantics:** Score thresholds remain consistent across different result set sizes

**Example Calibration:**

Raw RRF+recency scores:
- authentication.md: 0.0325 → **0.42** (moderate)
- api-reference.md: 0.0323 → **0.41** (moderate)
- oauth-guide.md: 0.0175 → **0.09** (noise, filtered)
- security.md: 0.0161 → **0.07** (noise, filtered)
- deployment.md: 0.0082 → **0.01** (noise, filtered)

With `min_confidence = 0.3`, only authentication.md and api-reference.md pass filtering.

**Configuration:**

```toml
[search]
score_calibration_threshold = 0.035  # RRF score for 50% confidence
score_calibration_steepness = 150.0  # Sigmoid curve steepness
min_confidence = 0.3                  # Filter results below 30% confidence
```

**Tuning Guidelines:**

- **Lower threshold (0.025):** More lenient, higher confidence for same raw score
- **Higher threshold (0.045):** Stricter, lower confidence for same raw score
- **Lower steepness (100.0):** Gentler transitions, less separation
- **Higher steepness (200.0):** Sharper transitions, more separation

**Code Reference:** [src/search/calibration.py](../src/search/calibration.py)

**Breaking Changes from v1.5 Min-Max Normalization:**

- Top result no longer always 1.0 (typically 0.8-0.98)
- Single-result queries no longer automatically 1.0
- Scores are absolute confidence, not relative to result set
- Set `min_confidence = 0.3` to filter low-quality results

## Performance Characteristics

### Query Latency

Measured on typical hardware (8-core CPU, 16GB RAM) with 1000-document corpus:

| Component | Latency | Notes |
|-----------|---------|-------|
| Semantic search | 50-100ms | Dominated by embedding inference |
| Keyword search | 10-20ms | Fast inverted index lookup |
| Graph traversal | 5-10ms | In-memory NetworkX operations |
| RRF fusion | 1-5ms | Simple arithmetic over ranked lists |
| **Total query** | **100-150ms** | Excluding LLM synthesis |

### Scaling Characteristics

| Corpus Size | Semantic | Keyword | Graph | Total |
|-------------|----------|---------|-------|-------|
| 100 docs | 30ms | 5ms | 2ms | ~40ms |
| 1,000 docs | 80ms | 15ms | 8ms | ~110ms |
| 10,000 docs | 250ms | 40ms | 20ms | ~320ms |

**Bottleneck:** Semantic search embedding generation and FAISS similarity computation.

### Memory Usage

| Component | Memory per Document | Notes |
|-----------|---------------------|-------|
| Vector index | ~1.5KB | 384-dim float32 embeddings + mappings |
| Keyword index | ~500 bytes | Inverted index entries |
| Graph | ~200 bytes | Node metadata + edge pointers |
| **Total** | **~2.2KB** | 1000 docs ≈ 2.2MB |

### Indexing Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Single document | 200-500ms | Includes embedding generation |
| Batch (100 docs) | 30-60s | Serial processing |
| Full rebuild (1000 docs) | 5-10 minutes | Cold start, no existing index |

**Bottleneck:** Embedding model inference (HuggingFace local model).

## Strategy Trade-Offs

### Semantic Search

**Strengths:**
- Finds conceptually related content
- Handles synonyms and paraphrasing
- Discovers unexpected relevant documents

**Weaknesses:**
- Misses exact term matches (function names, error codes)
- Slower than keyword search (embedding inference)
- Requires storage for vector embeddings

**Tuning:**
- Increase `semantic_weight` to prioritize semantic results
- Use larger embedding models for better semantic understanding (at cost of speed)

### Keyword Search

**Strengths:**
- Fast retrieval (inverted index)
- Exact term matching (function names, identifiers)
- Low storage overhead

**Weaknesses:**
- Misses synonyms and paraphrasing
- Sensitive to query phrasing
- StandardAnalyzer strips punctuation (limitation for technical terms)

**Tuning:**
- Increase `keyword_weight` to prioritize exact matches
- Use custom Whoosh analyzer for preserving special characters

### Graph Traversal

**Strengths:**
- Surfaces structurally related content
- Enriches context for LLM synthesis
- Discovers documents not matching query terms

**Weaknesses:**
- Relies on quality of wikilink annotations
- Can surface tangentially related content (topic drift)
- Adds computation overhead

**Tuning:**
- Graph boost multiplier (currently hardcoded at 0.5x) could be configurable
- Multi-hop depth (currently fixed at 1-hop) could be dynamic based on query

### Recency Bias

**Strengths:**
- Prioritizes up-to-date information
- Simple tier-based calculation
- Predictable behavior

**Weaknesses:**
- File mtime may not reflect content relevance
- Not useful for static documentation
- Can de-prioritize evergreen content

**Tuning:**
- Adjust `recency_bias` (0.0 to 1.0)
- Modify tier thresholds (7 days, 30 days) in code

## Configuration Recommendations

### General Purpose Documentation

Balanced weights with moderate recency:

```toml
[search]
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
```

### API Reference / Technical Docs

Prioritize exact term matches:

```toml
[search]
semantic_weight = 0.8
keyword_weight = 1.5
recency_bias = 0.3
```

### Personal Notes (Obsidian)

Prioritize semantic connections and recency:

```toml
[search]
semantic_weight = 1.3
keyword_weight = 0.7
recency_bias = 0.8
```

### Research Papers

Prioritize semantic similarity, disable recency:

```toml
[search]
semantic_weight = 1.5
keyword_weight = 0.5
recency_bias = 0.0
```

## Deduplication Strategies

### N-gram Overlap Deduplication

**Purpose:** Fast pre-filter to remove near-duplicate chunks before expensive embedding-based deduplication.

**Technology:**
- Character n-grams (trigrams by default)
- Jaccard similarity for set comparison
- O(n) complexity per comparison

**How it Works:**

1. Convert each chunk to a set of character n-grams:
   ```
   "hello world" → {"hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"}
   ```

2. Compute Jaccard similarity:
   $$
   J(A, B) = \frac{|A \cap B|}{|A \cup B|}
   $$

3. If similarity ≥ threshold (default 0.7), mark as duplicate

**Example:**

```python
text_a = "Configure the authentication settings in config.toml"
text_b = "Configure authentication settings in the config.toml file"

ngrams_a = get_ngrams(text_a, n=3)  # 48 trigrams
ngrams_b = get_ngrams(text_b, n=3)  # 52 trigrams

intersection = 41
union = 59
jaccard = 41 / 59 = 0.69  # Below 0.7 threshold: NOT duplicate
```

**Why N-grams Before Embeddings?**

| Method | Time per Comparison | Memory |
|--------|--------------------|---------|
| N-gram Jaccard | ~0.1ms | Minimal (sets) |
| Embedding Cosine | ~0.5ms | 1.5KB per embedding |

N-gram dedup removes obvious duplicates cheaply, reducing the candidate set for expensive semantic dedup.

**Configuration:**

```toml
[search]
ngram_dedup_enabled = true
ngram_dedup_threshold = 0.7
```

**Code Reference:** [src/search/dedup.py](../src/search/dedup.py)

### Maximal Marginal Relevance (MMR)

**Purpose:** Select diverse results by penalizing similarity to already-selected items.

**MMR Formula:**

$$
\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(d, s)
$$

Where:
- $d$ = candidate document
- $q$ = query
- $S$ = already-selected documents
- $\lambda$ = relevance/diversity trade-off (0.0–1.0)

**Lambda Trade-off:**

| Lambda | Behavior |
|--------|----------|
| 1.0 | Pure relevance (no diversity penalty) |
| 0.7 | Balanced (default) |
| 0.5 | Equal weight to relevance and diversity |
| 0.3 | Diversity-focused |

**Greedy Selection Algorithm:**

```python
selected = []
remaining = all_candidates

while len(selected) < top_n and remaining:
    best_id = None
    best_mmr = -inf

    for candidate in remaining:
        relevance = similarity(candidate, query)
        max_sim_to_selected = max(similarity(candidate, s) for s in selected)
        mmr_score = lambda * relevance - (1 - lambda) * max_sim_to_selected

        if mmr_score > best_mmr:
            best_mmr = mmr_score
            best_id = candidate

    selected.append(best_id)
    remaining.remove(best_id)
```

**When to Use:**
- Query results cluster around similar content
- Need diverse perspectives on a topic
- Want to avoid repetitive chunks from same document section

**Configuration:**

```toml
[search]
mmr_enabled = true
mmr_lambda = 0.7
```

**Code Reference:** [src/search/diversity.py](../src/search/diversity.py)

---

## Parent Document Retrieval

**Purpose:** Embed small chunks for retrieval precision, but return larger parent sections for sufficient LLM context.

**The Retrieval vs. Return Problem:**

Small chunks (~500 chars) improve retrieval precision—specific sentences match queries better than paragraphs. But returning small chunks to an LLM loses context. Parent document retrieval decouples the **retrieval unit** from the **return unit**.

**Two-Level Chunking:**

| Level | Size | Purpose |
|-------|------|---------|
| Parent (Section) | 1500–2000 chars | Return unit, provides context |
| Child (Sub-chunk) | 200–1500 chars | Retrieval unit, precision matching |

**Architecture:**

```
Document → Split into Sections (parent, ~2000 chars)
         → Split Sections into Chunks (child, ~500 chars)
         → Embed children, store parent_chunk_id reference
         → Search returns child matches
         → Expand to parent sections before returning
```

**Expansion Logic:**

```python
def _expand_to_parents(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen_parents: set[str] = set()
    expanded: list[tuple[str, float]] = []

    for chunk_id, score in results:
        parent_chunk_id = get_parent_id(chunk_id)
        if parent_chunk_id:
            if parent_chunk_id not in seen_parents:
                seen_parents.add(parent_chunk_id)
                expanded.append((parent_chunk_id, score))
        else:
            # Chunk is already a parent or has no parent
            expanded.append((chunk_id, score))

    return expanded
```

**Deduplication:** Multiple child chunks may share the same parent. The expansion logic deduplicates parents, keeping the highest-scoring child's score.

**Configuration:**

```toml
[chunking]
parent_retrieval_enabled = true
parent_chunk_min_chars = 1500
parent_chunk_max_chars = 2000
```

**Note:** Requires index rebuild when enabling (`uv run mcp-markdown-ragdocs rebuild-index`).

**Code Reference:** [src/chunking/header_chunker.py](../src/chunking/header_chunker.py), [src/search/orchestrator.py](../src/search/orchestrator.py)

---

## Future Enhancements

Potential improvements not currently implemented:

1. **Multi-hop graph traversal**: Configurable depth (1-hop, 2-hop)
2. **Custom analyzers**: Preserve punctuation for technical terms
3. **Learned fusion**: Train fusion weights on query/relevance pairs
4. **Query-dependent MMR lambda**: Adjust diversity based on query type
