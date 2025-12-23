# Configuration Examples

This directory contains example configurations for common use cases.

## Available Examples

### [config-minimal.toml](config-minimal.toml)

Bare minimum configuration to get started. Uses defaults for everything except documents path.

**Use when:**
- Getting started quickly
- Testing the server
- Default behavior is sufficient

### [config-obsidian.toml](config-obsidian.toml)

Configuration optimized for Obsidian vaults with heavy wikilink usage.

**Use when:**
- Indexing an Obsidian vault
- Working with interconnected notes
- Prefer semantic connections over exact matches
- Recent notes more important

### [config-documentation.toml](config-documentation.toml)

Configuration for technical documentation sites with multiple file types.

**Use when:**
- Indexing project documentation
- Exact API/function names matter (higher keyword weight)
- Documentation changes less frequently (lower recency bias)
- Need predictable exact-match behavior

### [config-development.toml](config-development.toml)

Configuration for development and testing with debug-friendly settings.

**Use when:**
- Developing new features
- Testing locally
- Need fast indexing and rebuilds
- Debugging search behavior

### [docker-compose.example.yml](docker-compose.example.yml)

Docker Compose configuration with detailed comments.

**Use when:**
- Deploying in containers
- Need persistent storage
- Running as a service
- Multi-environment deployments

## Using an Example

Copy the example you want to use:

```zsh
cp examples/config-minimal.toml config.toml
```

Edit paths to match your setup:

```toml
[indexing]
documents_path = "/path/to/your/documents"
```

Run the server:

```zsh
uv run mcp-markdown-ragdocs run
```

## Configuration Discovery

The server searches for `config.toml` in:

1. Current directory (`./config.toml`)
2. User config directory (`~/.config/mcp-markdown-ragdocs/config.toml`)

If no config file is found, all options use defaults (documents in current directory).

## Customization Tips

### Adjust Search Weights

Balance semantic vs keyword search:

```toml
[search]
semantic_weight = 1.2  # Favor conceptual similarity
keyword_weight = 0.8   # De-emphasize exact matches
```

Or reverse for technical docs where exact terms matter:

```toml
[search]
semantic_weight = 0.8
keyword_weight = 1.2   # Favor exact matches
```

### Tune Recency Bias

Control how much recent modifications matter:

```toml
[search]
recency_bias = 0.0   # Ignore modification time
recency_bias = 0.5   # Moderate boost (default)
recency_bias = 1.0   # Strong preference for recent docs
```

Recency tiers:
- Last 7 days: highest boost
- Last 30 days: medium boost
- Over 30 days: no boost

### Index Storage Location

Local project storage:

```toml
[indexing]
index_path = ".ragdocs-index/"
```

User-global cache:

```toml
[indexing]
index_path = "~/.cache/mcp-ragdocs-indices"
```

### Server Binding

Listen only on localhost (default):

```toml
[server]
host = "127.0.0.1"
port = 8000
```

Listen on all interfaces (LAN access):

```toml
[server]
host = "0.0.0.0"
port = 8080
```

## Validation

Check your configuration for errors:

```zsh
uv run mcp-markdown-ragdocs check-config
```

Shows resolved paths and all effective values.

## Complete Reference

See [docs/configuration.md](../docs/configuration.md) for exhaustive documentation of all options.
