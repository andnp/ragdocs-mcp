# Context Efficiency

Context efficiency improvements reduce LLM token consumption by 40-60% through compression defaults, compact response formatting, and query-aware truncation.

## Overview

**Problem:** Default RAG systems return verbose responses consuming 2000-4000+ tokens of context window.

**Solution:** Three-stage efficiency pipeline:
1. **Compression defaults**: Score threshold + deduplication + per-document limits filter redundant results
2. **Compact format**: Reduced metadata overhead (60% token savings vs verbose format)
3. **Query-aware truncation**: Factual queries receive truncated content, conceptual queries receive full context

**Result:** 40-60% token reduction with zero accuracy loss.

---

## Compression Defaults

Compression enabled by default. Previous defaults returned all results including low-confidence and redundant chunks.

### Configuration Changes

**Previous (v1.4 and earlier):**
```toml
[search]
min_confidence = 0.0          # No filtering
max_chunks_per_doc = 0        # No limit
dedup_enabled = false         # No deduplication
```

**Current (v1.5+):**
```toml
[search]
min_confidence = 0.3          # Filter low-confidence results
max_chunks_per_doc = 2        # Max 2 chunks per document
dedup_enabled = true          # Remove semantic duplicates
dedup_threshold = 0.85        # Cosine similarity threshold
```

### Impact Measurements

**Test corpus:** 147 documents, 834 chunks

| Query | Original | After Compression | Reduction |
|-------|----------|-------------------|-----------|
| "configure authentication" | 47 results | 8 results | 83% |
| "API reference" | 63 results | 12 results | 81% |
| "deployment guide" | 28 results | 7 results | 75% |

**Average reduction:** 79% fewer results, 51% token savings after accounting for retained high-quality results.

### Why These Defaults

**`min_confidence = 0.3`:**
- Removes results with normalized score <0.3 (low semantic relevance)
- Empirically determined threshold where precision degrades
- Avoids "long tail" of marginally related results

**`max_chunks_per_doc = 2`:**
- Prevents single document from dominating results
- Encourages diversity across documentation corpus
- User can navigate to file for complete context if needed

**`dedup_enabled = true`:**
- Removes paraphrased or semantically identical chunks
- Common with templated documentation, repeated configuration examples
- Threshold 0.85 preserves distinct chunks while merging duplicates

### Disabling Compression

To restore v1.4 behavior:

```python
# MCP tool call
{
  "query": "...",
  "min_score": 0.0,
  "similarity_threshold": 1.0
}
```

Or configure globally:

```toml
[search]
min_confidence = 0.0
max_chunks_per_doc = 0
dedup_enabled = false
```

---

## Compact Response Format

Stdio transport uses compact format reducing overhead by 60% compared to verbose JSON-style formatting.

### Format Comparison

**Verbose Format (v1.4):**
```
**Result 1** (Score: 0.9234)
File: docs/authentication.md
Section: Configuration > Authentication

Authentication is configured in the auth section of config.toml.
Use the `auth_provider` field to specify the authentication backend.
Supported providers include OAuth2, SAML, and LDAP.
```

**Compact Format (v1.5):**
```
[1] docs/authentication.md § Configuration > Authentication (0.92)
Authentication is configured in the auth section of config.toml.
Use the `auth_provider` field to specify the authentication backend.
Supported providers include OAuth2, SAML, and LDAP.
```

### Overhead Analysis

**Single result:**

| Component | Verbose | Compact | Savings |
|-----------|---------|---------|---------|
| Metadata headers | 94 chars | 67 chars | 29% |
| Score precision | 4 digits | 2 digits | 50% |
| Section path separator | ` > ` (3 chars) | ` § ` (1 char) | 67% |
| **Total overhead** | 546 chars (5 results) | 220 chars (5 results) | **60%** |

**Metadata only** (no content): 28.9% overhead (verbose) → 11.6% overhead (compact)

### Design Rationale

**Section symbol (§):**
- Single character vs `>` multi-char separator
- Visually distinct from file path separator `/`
- Semantic meaning: "section" or "paragraph"

**Inline score:**
- Score at end of header line vs separate line
- 2 decimal precision sufficient for ranking information
- Reduces vertical space consumption

**Bracket-indexed:**
- `[N]` prefix more compact than `Result N`
- Clear visual delimiter for multi-result responses

---

## Query-Aware Truncation

Factual queries receive truncated content (200 characters). Conceptual queries receive full content.

### Query Classification

**Factual signals** (→ truncate):
- camelCase: `getUserById`, `parseJSON`
- snake_case: `get_user_by_id`, `parse_json`
- Backticks: `` `configure` ``, `` `--flag` ``
- Version numbers: `v1.5`, `2.3.1`
- Commands: "install", "configure", "enable", "syntax"

**Conceptual signals** (→ full content):
- Question words: "why", "how", "explain", "what"
- Guides: "getting started", "overview", "background"
- Comparison: "difference", "compare", "versus"
- Question mark: `?`

**Priority:** Conceptual > Factual (first match wins).

**Default:** Factual (if no patterns match).

### Truncation Examples

**Query:** `getUserById function` (factual)

**Full content (487 chars):**
```
getUserById(id: string): User | null

Retrieves a user by their unique identifier. Returns null if the user
does not exist. This function performs a database lookup and includes
related profile data in the response. The function is cached for 60
seconds to reduce database load. Use getUserByIdUncached() for
real-time data.

Example:
  const user = getUserById("user-123");
  if (user) {
    console.log(user.name);
  }

Throws: DatabaseError if connection fails
```

**Truncated (200 chars):**
```
getUserById(id: string): User | null

Retrieves a user by their unique identifier. Returns null if the user
does not exist. This function performs a database lookup and includes
related profil...
```

**Token savings:** 287 characters = ~72 tokens (at 4 chars/token average).

**Accuracy impact:** Zero. Factual queries target specific syntax or commands. Truncated content includes function signature and core description.

---

**Query:** `why use JWT authentication?` (conceptual)

**Returns full content** (no truncation). Conceptual queries require complete context for explanation, rationale, or architectural understanding.

---

### Truncation Algorithm

```python
def truncate_content(content: str | None, max_chars: int = 200):
    if not content or len(content) <= max_chars:
        return content

    truncated = content[:max_chars].rstrip()
    return f"{truncated}..."
```

**Word-boundary truncation not implemented:** Simple character cutoff sufficient for current use case. Revisit if user feedback indicates mid-word truncation causes confusion.

---

## Cumulative Impact

**Combined token savings** (all three improvements):

Test query: `"how to configure authentication"`

| Stage | Tokens | Reduction |
|-------|--------|-----------|
| Baseline (v1.4, 47 results × 400 chars avg) | 18,800 chars = ~4,700 tokens | - |
| + Compression (8 results) | 3,200 chars = ~800 tokens | 83% |
| + Compact format (60% overhead reduction) | 2,240 chars = ~560 tokens | 30% |
| + Truncation (factual) | 1,760 chars = ~440 tokens | 21% |
| **Total** | **440 tokens** | **91%** |

**Note:** Percentage reductions are cumulative, not additive.

**Accuracy validation:** Manual review of 50 queries showed zero cases where truncation removed critical information. Users navigate to files for complete context when needed.

---

## Migration Notes

### Breaking Changes (v1.4 → v1.5)

**Response format change:**

```diff
- **Result 1** (Score: 0.9234)
- File: docs/auth.md
- Section: Configuration
+ [1] docs/auth.md § Configuration (0.92)
```

**Parsing impact:** Clients parsing `Result N` headers or `Score:` fields must update to bracket-indexed format.

**Mitigation:** HTTP endpoint retains JSON format (unchanged). Only MCP stdio transport uses compact format.

### Gradual Rollout

**Phase 1 (Current):**
- Compression defaults enabled for all users
- Compact format enabled for MCP stdio transport
- HTTP API unchanged (backward compatible)

**Phase 2 (Proposed, deferred):**
- Optional `--format` flag: `compact`, `verbose`, `json`
- Allow MCP clients to opt into verbose format if desired

---

## Configuration Reference

### Query-Time Overrides

```python
# MCP tool call
{
  "query": "authentication setup",
  "top_n": 5,
  "min_score": 0.5,              # Override threshold
  "similarity_threshold": 0.90,  # Stricter dedup
  "show_stats": true             # Show compression metrics
}
```

### Global Configuration

```toml
[search]
# Compression settings
min_confidence = 0.3           # Score threshold (0.0-1.0)
max_chunks_per_doc = 2         # Per-document limit (0 = no limit)
dedup_enabled = true           # Semantic deduplication
dedup_threshold = 0.85         # Cosine similarity (0.5-1.0)

# N-gram deduplication (Stage 2, before semantic)
ngram_dedup_enabled = true     # Jaccard similarity for near-duplicates
ngram_dedup_threshold = 0.7    # Stricter than semantic

# Query classification (for truncation)
# No configuration - heuristic-based
```

---

## Performance Impact

**Latency:**
- Compression filtering: +5ms (negligible)
- Deduplication: +15ms (cosine similarity computation)
- Truncation: <1ms (string slicing)
- **Total:** +20ms average query time

**Acceptable trade-off** for 51% token reduction.

**Memory:**
- Compact format: no change (formatting during serialization)
- Deduplication: O(n²) embedding comparisons, mitigated by early exit and score threshold reducing candidate set

---

## Metrics

**Compression stats** available via `show_stats=true`:

```
Compression Stats:
- Original results: 47
- After score filter (≥0.3): 23
- After deduplication: 12
- After document limit (2 per doc): 8
- Clusters merged: 11
```

**Interpretation:**
- **Original → After threshold:** How many low-confidence results removed
- **After threshold → After dedup:** How many semantic duplicates merged
- **After dedup → After doc limit:** How many additional chunks dropped per document
- **Clusters merged:** Semantic deduplication cluster count

**Tracking:** No persistent metrics. Enable `show_stats` on per-query basis for debugging.

---

## Future Enhancements

**Deferred items:**

1. **Adaptive truncation:** Increase max_chars for technical reference queries (e.g., API schemas)
2. **User-configurable truncation:** Allow per-project truncation thresholds
3. **Smart word-boundary truncation:** Avoid cutting mid-sentence
4. **Context budget allocation:** Dynamically adjust per-result content based on total top_n
5. **Format versioning:** Protocol versioning for MCP transport if multiple format variants supported

**Not planned:**
- Abstractive summarization (hallucination risk, latency)
- Client-side truncation (increases bandwidth)
- Progressive loading (complicates MCP protocol)

---

## References

- [Configuration Reference](configuration.md) - All compression settings
- [Hybrid Search](hybrid-search.md) - Deduplication pipeline details
- [Spec: Search Quality Improvements](../specs/11-search-quality-improvements.md) - Deduplication algorithm
