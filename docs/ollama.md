# Ollama embeddings

ragdocs uses Ollama for embeddings by default. The model runs in the Ollama
daemon, so the MCP process and indexing worker do not each load a copy of the
model into Python memory.

Install and start Ollama, then pull the configured model:

```sh
ollama serve
ollama pull nomic-embed-text:latest
```

The default configuration is equivalent to:

```toml
[embedding]
provider = "ollama"
model_name = "nomic-embed-text:latest"
base_url = "http://localhost:11434"
auto_pull = true
timeout_seconds = 60.0
pull_timeout_seconds = 600.0
```

Set `base_url` when Ollama runs on another host. `auto_pull` can be disabled
in production to make a missing model fail fast instead of downloading it.

The canonical backend stores records, keywords, vectors, and graph edges in
the configured `index_path` SQLite database. Changing the embedding model or
dimension creates a new semantic namespace; rebuild the index after changing
either value:

```sh
uv run mcp-markdown-ragdocs rebuild-index
```

For offline tests, set `MCP_RAGDOCS_TEST_FAKE_EMBEDDINGS=1` to use the
deterministic test provider without contacting Ollama.
