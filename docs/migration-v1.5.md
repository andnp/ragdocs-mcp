# Migration Guide: v1.5 (Removal of Synthesized Answer Feature)

## Overview

Version 1.5 removes the synthesized answer feature. The tool now returns ranked document chunks instead of generating synthesized answers. This change refocuses the tool on its core strength: identifying relevant documentation sections using hybrid semantic search.

## Breaking Changes

### Removed: `answer` Field from Tool Response

**Before (v1.4):**

```json
{
  "answer": "Authentication is configured via the auth.toml file...",
  "results": [
    {
      "content": "...",
      "file_path": "docs/authentication.md",
      "header_path": ["Configuration"],
      "score": 1.0
    }
  ]
}
```

**After (v1.5):**

```json
{
  "results": [
    {
      "chunk_id": "authentication_0",
      "content": "...",
      "file_path": "docs/authentication.md",
      "header_path": ["Configuration"],
      "score": 1.0
    }
  ]
}
```

### Added: `chunk_id` Field

Each result now includes a `chunk_id` field for unique identification of document chunks.

## Rationale

### Why Remove Synthesis?

1. **Focus on Discovery:** The tool's strength is identifying which sections of documentation are relevant, not generating natural language summaries.

2. **Resource Efficiency:** Loading many files into LLM context is expensive and can poison the context window. This tool provides efficient search to identify specific sections.

3. **Better Workflow:** The revised usage pattern encourages a more effective workflow:
   - Search → identify relevant files/sections → read full documentation
   - This provides better context than synthesized summaries

4. **Clear Responsibility:** The tool is responsible for discovery and ranking. The LLM using the tool is responsible for reading and reasoning about the identified sections.

## Migration Steps

### For MCP Clients (VS Code, Claude Desktop)

MCP clients receive tool responses automatically. No configuration changes required, but usage patterns may need adjustment:

**Before (v1.4):**
```
Query: "How do I configure authentication?"
Response: "Authentication is configured via the auth.toml file..."
→ Client receives synthesized answer directly
```

**After (v1.5):**
```
Query: "How do I configure authentication?"
Response: {
  "results": [
    {"chunk_id": "auth_0", "file_path": "docs/auth.md", ...}
  ]
}
→ Client receives ranked chunks, reads file_path for full context
```

**Recommended Workflow:**
1. Call `query_documents` to identify relevant sections
2. Review the `file_path` and `header_path` fields
3. Use file reading tools (native to your MCP client) to access full documentation

### For HTTP API Clients

Update response parsing to handle the new structure:

**Before (v1.4):**

```python
response = requests.post(
    "http://localhost:8000/query_documents",
    json={"query": "authentication config"}
)
data = response.json()
answer = data["answer"]  # ← No longer exists
print(answer)
```

**After (v1.5):**

```python
response = requests.post(
    "http://localhost:8000/query_documents",
    json={"query": "authentication config"}
)
data = response.json()
results = data["results"]

# Display ranked results
for result in results:
    print(f"File: {result['file_path']}")
    print(f"Section: {' > '.join(result['header_path'])}")
    print(f"Score: {result['score']:.2f}")
    print(f"Preview: {result['content'][:200]}...")
    print()

# Read full file for complete context
target_file = results[0]["file_path"]
with open(target_file) as f:
    full_content = f.read()
```

### For CLI Users

The `query` command output format changes:

**Before (v1.4):**

```zsh
$ uv run mcp-markdown-ragdocs query "authentication"

Answer:
Authentication is configured via the auth.toml file...

Sources:
- docs/auth.md
- docs/security.md
```

**After (v1.5):**

```zsh
$ uv run mcp-markdown-ragdocs query "authentication"

Found 3 results:

╭─ #1 Score: 0.92 ──────────────────────────────╮
│ Document: authentication                      │
│ Section: Configuration > OAuth                │
│ File: docs/auth.md                           │
│                                               │
│ To configure authentication, set the auth     │
│ section in config.toml...                    │
╰───────────────────────────────────────────────╯
```

## Configuration Changes

### Deprecated: `llm_provider`

The `[llm]` section's `llm_provider` field is now deprecated:

```toml
[llm]
# llm_provider = null  # ← DEPRECATED: No longer used, can be removed
```

The field is retained for backward compatibility but has no effect. Remove it from your configuration.

### No Other Breaking Changes

All other configuration options remain unchanged:
- `[search]` section: All options preserved
- `[indexing]` section: All options preserved
- `[chunking]` section: All options preserved

## Tool Parameter Changes

### `query_documents` Tool

**Added Optional Parameters:**

```python
{
  "query": "string (required)",
  "top_n": "integer (optional, default: 5)",
  "min_score": "number (optional, default: 0.0)",  # NEW
  "similarity_threshold": "number (optional, default: 0.85)"  # NEW
}
```

- `min_score`: Filter results below this confidence threshold (0.0-1.0)
- `similarity_threshold`: Semantic deduplication threshold (0.5-1.0)

**Example:**

```json
{
  "query": "How do I configure authentication?",
  "top_n": 5,
  "min_score": 0.3,
  "similarity_threshold": 0.85
}
```

## Testing Your Migration

### Verify Tool Response Structure

```python
import json

# Call query_documents tool
response = query_documents(query="test query")

# Verify structure
assert "results" in response
assert "answer" not in response  # Removed field

# Verify chunk structure
for result in response["results"]:
    assert "chunk_id" in result  # New field
    assert "content" in result
    assert "file_path" in result
    assert "header_path" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
```

### Verify Backward Compatibility

Existing indices do not require rebuild. The v1.5 server can read v1.4 indices without issues.

```zsh
# Check server version
uv run mcp-markdown-ragdocs --version

# Verify index compatibility
uv run mcp-markdown-ragdocs check-config
```

## Rollback Instructions

If you need to revert to v1.4:

```zsh
# Using uv
uv pip install 'mcp-markdown-ragdocs==1.4.0'

# Or using git tag
git checkout v1.4.0
uv sync
```

Configuration files from v1.5 are fully compatible with v1.4 (the deprecated `llm_provider` field will be ignored).

## FAQ

### Q: Why not keep both modes (synthesis and raw chunks)?

**A:** Maintaining two response formats increases complexity and testing burden. The discovery-focused workflow (search → read) provides better results than synthesis.

### Q: How should LLMs use this tool effectively?

**A:**
1. Call `query_documents` to identify relevant sections
2. Parse the `file_path` and `header_path` fields to locate content
3. Use file reading capabilities to access full documentation
4. Synthesize your own response based on the full context

### Q: Will indices need to be rebuilt?

**A:** No. Indices from v1.4 are fully compatible with v1.5. The change affects response format only.

### Q: Can I still get synthesized answers?

**A:** No. The synthesis feature has been removed to focus the tool on efficient document discovery. LLMs consuming this tool should perform their own synthesis after reading identified sections.

## Support

For issues or questions:
- File an issue: [GitHub Issues](https://github.com/yourusername/mcp-markdown-ragdocs/issues)
- Check documentation: [docs/](./README.md)
- Review examples: [examples/](../examples/)

## Summary

**Key Changes:**
- ✅ Removed `answer` field from responses
- ✅ Added `chunk_id` field to results
- ✅ Deprecated `llm_provider` configuration
- ✅ Added optional `min_score` and `similarity_threshold` parameters
- ✅ Updated CLI output format
- ✅ Refined tool purpose: discovery-focused hybrid semantic search

**Migration Effort:** Low (response parsing updates only)

**Backward Compatibility:** Index format unchanged, configuration fully compatible
