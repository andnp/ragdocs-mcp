# Troubleshooting Guide

This guide helps diagnose and fix common issues with search quality, performance, and configuration using the CLI debug mode and other diagnostic tools.

## Using Debug Mode

The `--debug` flag displays search internals for understanding result quality and tuning configuration:

```zsh
uv run mcp-markdown-ragdocs query "your query" --debug
```

Debug mode shows two tables:

### Search Strategy Results

Shows result counts from each search strategy before fusion:

```
┌─ Search Strategy Results ──────────┐
│ Strategy          │ Count          │
├───────────────────┼────────────────┤
│ Vector (Semantic) │ 15             │
│ Keyword (BM25)    │ 8              │
│ Graph (PageRank)  │ 3              │
└───────────────────┴────────────────┘
```

**What to look for:**
- **Zero counts**: Strategy not contributing (check if indices built successfully)
- **Imbalanced counts**: One strategy dominating (adjust weights in config)
- **Low semantic count**: Query may need more specific terms or embeddings may be stale

### Compression Pipeline

Shows filtering stages and items removed at each step:

```
┌─ Compression Pipeline ───────────────────┐
│ Stage                      │ Count │ Removed│
├────────────────────────────┼───────┼────────┤
│ Original (RRF Fusion)      │ 26    │ -      │
│ After Confidence Filter    │ 20    │ 6      │
│ After Content Dedup        │ 18    │ 2      │
│ After N-gram Dedup         │ 16    │ 2      │
│ After Semantic Dedup       │ 12    │ 4      │
│ After Doc Limit            │ 5     │ 7      │
└────────────────────────────┴───────┴────────┘
```

**What to look for:**
- **High confidence filter removal**: Lower `min_confidence` threshold if results too sparse
- **High semantic dedup removal**: Lower `dedup_similarity_threshold` if losing distinct results
- **High doc limit removal**: Increase `max_chunks_per_doc` if need more context per document

## Common Issues

### 1. No Results or Too Few Results

**Symptoms:**
- Empty result set
- Fewer results than expected

**Diagnosis with debug mode:**
```zsh
uv run mcp-markdown-ragdocs query "your query" --debug --top-n 20
```

**Check:**
1. **Strategy counts**: Are all strategies returning zero?
   - Vector=0, Keyword=0, Graph=0 → Index may be empty or corrupted
   - Run: `uv run mcp-markdown-ragdocs rebuild-index`

2. **Confidence filter removal**: High removal at confidence stage?
   - **Cause**: `min_confidence` threshold too high
   - **Fix**: Lower threshold in config:
     ```toml
     [search]
     min_confidence = 0.1  # Default: 0.3
     ```

3. **Query too specific**: Semantic search may miss exact terms
   - **Try**: More general terms or use keyword-focused query
   - **Adjust**: Increase keyword weight:
     ```toml
     [search]
     keyword_weight = 2.0  # Default: 1.0
     semantic_weight = 1.0
     ```

### 2. Low-Quality Results (Irrelevant Content)

**Symptoms:**
- Results don't match query intent
- Top results have low relevance

**Diagnosis with debug mode:**
```zsh
uv run mcp-markdown-ragdocs query "your query" --debug
```

**Check:**
1. **Strategy balance**: Is one strategy dominating?
   - Semantic >> Keyword → Query may be too vague, keyword search not matching
   - Keyword >> Semantic → Query has exact terms but semantic context missing
   - **Fix**: Adjust weights to balance strategies:
     ```toml
     [search]
     semantic_weight = 1.0
     keyword_weight = 1.0
     ```

2. **Calibration threshold**: Are low-score results passing through?
   - **Cause**: `score_calibration_threshold` too low
   - **Fix**: Raise threshold for stricter filtering:
     ```toml
     [search]
     score_calibration_threshold = 0.5  # Default: 0.3
     ```

3. **Deduplication too aggressive**: Losing distinct results?
   - Check semantic dedup removal count
   - **Fix**: Raise similarity threshold (only dedupe near-duplicates):
     ```toml
     [search]
     dedup_similarity_threshold = 0.90  # Default: 0.85
     ```

### 3. Duplicate or Near-Duplicate Results

**Symptoms:**
- Multiple results with similar content
- Same information repeated across chunks

**Diagnosis with debug mode:**
```zsh
uv run mcp-markdown-ragdocs query "your query" --debug
```

**Check:**
1. **Dedup stages**: Low removal in Content/N-gram/Semantic Dedup?
   - **Cause**: Deduplication disabled or thresholds too high
   - **Fix**: Enable and tune deduplication:
     ```toml
     [search]
     dedup_enabled = true
     dedup_similarity_threshold = 0.80  # Lower = more aggressive dedup
     ```

2. **Per-document limit**: Multiple chunks from same document?
   - Check "After Doc Limit" removal count
   - **Fix**: Lower per-document chunk limit:
     ```toml
     [search]
     max_chunks_per_doc = 1  # Default: 2
     ```

### 4. Missing Recent Content

**Symptoms:**
- Recently added/updated documents not appearing
- Stale results

**Diagnosis:**
```zsh
# Check index status
uv run mcp-markdown-ragdocs check-config

# Force rebuild
uv run mcp-markdown-ragdocs rebuild-index
```

**Check:**
1. **File watcher**: Is automatic indexing working?
   - MCP server watches files in background
   - HTTP server requires manual restart or rebuild

2. **Exclude patterns**: Are files being filtered out?
   - Check `[indexing].exclude` patterns in config
   - Hidden files (`.md`) excluded by default if `exclude_hidden_dirs = true`

3. **Recency bias**: Increase weight for recent documents:
   ```toml
   [search]
   recency_bias = 1.0  # Default: 0.5 (higher = stronger recency boost)
   ```

### 5. Performance Issues (Slow Queries)

**Symptoms:**
- Query takes >2 seconds
- High latency in MCP responses

**Diagnosis with debug mode:**
```zsh
time uv run mcp-markdown-ragdocs query "your query" --debug
```

**Check:**
1. **Strategy counts**: Are counts very high (>100)?
   - **Cause**: Large corpus, inefficient retrieval
   - **Fix**: Reduce top_k during search:
     ```toml
     [search]
     # In code, top_k defaults to max(20, top_n * 4)
     # Consider lowering multiplier in src/search/orchestrator.py
     ```

2. **Re-ranking overhead**: Re-ranking enabled with many results?
   - ~50ms per 10 documents
   - **Fix**: Disable or reduce re-ranking scope:
     ```toml
     [search]
     rerank_enabled = false
     # OR
     rerank_top_n = 10  # Only re-rank top 10
     ```

3. **Query expansion**: Is query expansion enabled?
   - Adds extra embedding lookups
   - **Fix**: Disable if not needed:
     ```toml
     [search]
     query_expansion_enabled = false
     ```

4. **Index size**: Very large index (>10k documents)?
   - FAISS may benefit from different index type
   - Consider filtering corpus or splitting into projects

## Configuration Tuning Examples

### Precision-Focused (High Quality, Fewer Results)

```toml
[search]
min_confidence = 0.5                # Strict score threshold
score_calibration_threshold = 0.6   # High calibration bar
dedup_enabled = true
dedup_similarity_threshold = 0.75   # Aggressive dedup
max_chunks_per_doc = 1              # One chunk per doc
rerank_enabled = true               # Use cross-encoder
```

### Recall-Focused (More Results, Lower Bar)

```toml
[search]
min_confidence = 0.1                # Permissive threshold
score_calibration_threshold = 0.2   # Low calibration bar
dedup_enabled = false               # Keep all results
max_chunks_per_doc = 3              # Multiple chunks per doc
keyword_weight = 2.0                # Boost keyword matching
```

### Balanced (Default with Tweaks)

```toml
[search]
min_confidence = 0.3
score_calibration_threshold = 0.3
dedup_enabled = true
dedup_similarity_threshold = 0.85
max_chunks_per_doc = 2
semantic_weight = 1.0
keyword_weight = 1.0
recency_bias = 0.5
```

## Interpreting Debug Statistics

### Strategy Result Patterns

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Semantic: High, Keyword: Low | Query is conceptual/broad | Increase semantic weight |
| Semantic: Low, Keyword: High | Query has exact terms | Increase keyword weight |
| Graph: Low/Zero | Weak document connectivity | Check wikilink usage or disable graph |
| All: Low | Small corpus or very specific query | Broaden query or add documents |

### Compression Pipeline Patterns

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| High confidence removal (>50%) | Threshold too strict | Lower `min_confidence` |
| Low/zero dedup removal | Dedup not working or needed | Check `dedup_enabled` setting |
| High doc limit removal | Many chunks per document | Adjust `max_chunks_per_doc` |
| Original count << top_k | Not enough candidate results | Increase corpus or broaden query |

## Advanced Diagnostics

### Check Index Health

```zsh
# Validate configuration
uv run mcp-markdown-ragdocs check-config

# Check manifest
cat .index_data/index.manifest.json | jq
```

**Look for:**
- Document count matches expected
- Chunk count reasonable (typically 5-20x document count)
- No failed files in manifest

### Inspect Individual Indices

```python
# In Python REPL
from src.context import ApplicationContext

ctx = ApplicationContext.create()
ctx.index_manager.load()

# Check vector index
print(f"Vector index size: {ctx.index_manager.vector.size()}")
print(f"Embedding dim: {ctx.index_manager.vector.dimension}")

# Check keyword index
print(f"Keyword index docs: {ctx.index_manager.keyword.size()}")

# Check graph
print(f"Graph nodes: {len(ctx.index_manager.graph._graph.nodes)}")
print(f"Graph edges: {len(ctx.index_manager.graph._graph.edges)}")
```

### Compare MCP vs. CLI Results

MCP responses may differ from CLI due to formatting:

```zsh
# CLI (formatted output)
uv run mcp-markdown-ragdocs query "test" --debug

# CLI (JSON output, matches MCP structure)
uv run mcp-markdown-ragdocs query "test" --json
```

## Getting Help

If issues persist after troubleshooting:

1. **Collect diagnostics:**
   ```zsh
   # Config validation
   uv run mcp-markdown-ragdocs check-config > diagnostics.txt

   # Debug query output
   uv run mcp-markdown-ragdocs query "problematic query" --debug >> diagnostics.txt

   # Index manifest
   cat .index_data/index.manifest.json >> diagnostics.txt
   ```

2. **Check logs:**
   - MCP server: `~/.local/share/mcp-markdown-ragdocs/logs/`
   - HTTP server: Console output

3. **File issue:**
   - Include diagnostics output
   - Describe expected vs. actual behavior
   - Note corpus size and configuration
