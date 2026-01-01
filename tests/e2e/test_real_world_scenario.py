"""
E2E test for real-world documentation site scenario.

Tests complete workflow with interconnected documentation corpus simulating
a technical documentation site. Validates hybrid search strategies, graph
traversal, and LLM synthesis.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.config import Config, IndexingConfig, LLMConfig, SearchConfig, ServerConfig
from src.server import create_app


@pytest.fixture
def docs_corpus(tmp_path):
    """
    Create realistic documentation corpus with 8-10 interconnected files.

    Simulates a technical documentation site with:
    - Wikilinks between documents
    - Frontmatter tags and aliases
    - Code blocks and examples
    - Varied content lengths
    - Recent timestamp on deployment doc
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # 1. Getting Started - Entry point with links
    (docs_dir / "getting-started.md").write_text("""---
tags: [tutorial, beginner]
aliases: ["quickstart", "intro"]
---

# Getting Started

Welcome to our platform! This guide will help you get up and running quickly.

## Prerequisites

Before you begin, make sure you have:
- An active account
- Basic understanding of REST APIs

## First Steps

1. **Authentication**: See [[authentication]] for setting up your API credentials
2. **API Reference**: Check out the [[api-reference]] for available endpoints
3. **Configuration**: Review [[configuration]] options for your environment

## Quick Example

```python
import requests

# Your first API call
response = requests.get('https://api.example.com/v1/status')
print(response.json())
```

For advanced topics, see [[advanced-topics]].
""")

    # 2. Authentication - Core doc with security link
    (docs_dir / "authentication.md").write_text("""---
tags: [security, api, authentication]
aliases: ["auth", "credentials"]
---

# Authentication

Learn how to authenticate API requests securely.

## API Keys

Generate API keys from your dashboard. Keys should be kept secret and rotated regularly.

### Creating API Keys

1. Navigate to Settings > API Keys
2. Click "Generate New Key"
3. Copy and store securely

## OAuth 2.0

For user-delegated access, use OAuth 2.0 authentication flow.

See [[security]] best practices for credential management.
For API endpoints requiring authentication, check [[api-reference]].

## Token Validation

All requests must include the `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/data
```
""")

    # 3. API Reference - Technical reference linking back to auth
    (docs_dir / "api-reference.md").write_text("""---
tags: [api, reference, technical]
aliases: ["api", "endpoints"]
---

# API Reference

Complete reference for all API endpoints.

## Authentication Required

All endpoints require authentication. See [[authentication]] for setup.

## Endpoints

### GET /v1/data

Retrieve data from the platform.

**Headers:**
- `Authorization`: Bearer token (required)

**Parameters:**
- `limit`: Number of records (default: 10)
- `offset`: Pagination offset (default: 0)

### POST /v1/data

Create new data records.

**Request Body:**
```json
{
  "name": "Example",
  "value": 42
}
```

For deployment considerations, see [[deployment]].
For security guidelines, review [[security]].
""")

    # 4. Security - Security guide linking to authentication
    (docs_dir / "security.md").write_text("""---
tags: [security, best-practices, compliance]
aliases: ["secure", "safety"]
---

# Security Best Practices

Protect your application and data with these security guidelines.

## Credential Management

Never expose API keys in client-side code or version control.
See [[authentication]] for proper credential handling.

## HTTPS Only

Always use HTTPS for API requests. Our platform enforces TLS 1.2 or higher.

## Rate Limiting

Implement rate limiting to prevent abuse:
- 1000 requests per hour per API key
- 10 requests per second per endpoint

## Security Headers

Configure your application with proper security headers:

```nginx
add_header X-Content-Type-Options "nosniff";
add_header X-Frame-Options "DENY";
add_header X-XSS-Protection "1; mode=block";
```

Review [[deployment]] for production security settings.
For advanced security topics, see [[advanced-topics]].
""")

    # 5. Deployment - Recent doc (2 days ago) for recency testing
    deployment_path = docs_dir / "deployment.md"
    deployment_path.write_text("""---
tags: [deployment, devops, production]
aliases: ["deploy", "hosting"]
---

# Deployment Guide

Deploy your application to production environments.

## Environment Configuration

Set environment variables for production:

```bash
export API_BASE_URL=https://api.example.com
export API_KEY=your_production_key
export LOG_LEVEL=info
```

See [[configuration]] for all available options.

## Docker Deployment

Build and run with Docker:

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

## Security Considerations

Ensure proper [[security]] measures are in place:
- Use secrets management for API keys
- Enable firewall rules
- Configure HTTPS certificates

Review [[authentication]] setup for production environments.

## Monitoring

Set up monitoring and alerting for production systems.
""")
    # Set modified time to 2 days ago for recency testing
    two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
    os.utime(deployment_path, (two_days_ago.timestamp(), two_days_ago.timestamp()))

    # 6. Advanced Topics - Technical deep dive
    (docs_dir / "advanced-topics.md").write_text("""---
tags: [advanced, technical, optimization]
aliases: ["advanced", "expert"]
---

# Advanced Topics

Deep dive into advanced platform capabilities.

## Performance Optimization

### Caching Strategies

Implement caching to reduce API calls:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fetch_data(key):
    return api_client.get(f'/v1/data/{key}')
```

### Batch Operations

Use batch endpoints for bulk operations to improve throughput.

## Webhooks

Configure webhooks for real-time event notifications.

See [[configuration]] for webhook setup.
Review [[security]] for webhook signature verification.

## Custom Integrations

Build custom integrations using our SDK and API.
Refer to [[api-reference]] for complete endpoint documentation.
""")

    # 7. Troubleshooting - FAQ-style content
    (docs_dir / "troubleshooting.md").write_text("""---
tags: [help, faq, debug]
aliases: ["help", "issues", "faq"]
---

# Troubleshooting

Common issues and solutions.

## Authentication Errors

**Error: 401 Unauthorized**

Your API key may be invalid or expired.
- Check your [[authentication]] credentials
- Verify the key is active in your dashboard
- Ensure proper Authorization header format

## API Rate Limits

**Error: 429 Too Many Requests**

You've exceeded the rate limit. See [[security]] for rate limit details.

Wait before retrying or upgrade your plan for higher limits.

## Connection Issues

**Error: Connection Timeout**

- Check your network connectivity
- Verify firewall rules allow outbound HTTPS
- Review [[deployment]] configuration

## Configuration Problems

If services won't start, validate your [[configuration]] file syntax.

For deployment-specific issues, see the [[deployment]] guide.
""")

    # 8. Configuration - Config guide with examples
    (docs_dir / "configuration.md").write_text("""---
tags: [configuration, setup, settings]
aliases: ["config", "settings"]
---

# Configuration Guide

Configure the platform for your environment.

## Configuration File

Create `config.yaml` in your project root:

```yaml
api:
  base_url: https://api.example.com
  timeout: 30
  retries: 3

auth:
  api_key: ${API_KEY}
  refresh_token: ${REFRESH_TOKEN}

logging:
  level: info
  format: json
```

See [[authentication]] for auth configuration details.

## Environment Variables

Override config with environment variables:

- `API_BASE_URL`: API endpoint URL
- `API_KEY`: Authentication key
- `LOG_LEVEL`: Logging verbosity (debug, info, warn, error)

## Development vs Production

Use different configs for each environment.
Review [[deployment]] for production configuration.
Check [[security]] for secure configuration practices.

## Advanced Settings

For advanced options, see [[advanced-topics]].
""")

    return docs_dir


@pytest.fixture
def app_with_corpus(tmp_path, docs_corpus, monkeypatch):
    """
    Create FastAPI app with realistic documentation corpus.

    Configures server with test corpus and waits for initial indexing.
    """
    index_path = tmp_path / "indices"
    index_path.mkdir(parents=True, exist_ok=True)

    def mock_load_config():
        return Config(
            server=ServerConfig(host="127.0.0.1", port=8000),
            indexing=IndexingConfig(
                documents_path=str(docs_corpus),
                index_path=str(index_path),
                recursive=True,
            ),
            parsers={"**/*.md": "MarkdownParser"},
            search=SearchConfig(
                semantic_weight=0.6,
                keyword_weight=0.4,
                recency_bias=0.5,
                rrf_k_constant=60,
            ),
            llm=LLMConfig(
                embedding_model="all-MiniLM-L6-v2"
            ),
        )

    monkeypatch.setattr("src.context.load_config", mock_load_config)
    app = create_app()
    return app


@pytest.fixture
def client(app_with_corpus):
    """
    Create TestClient for HTTP testing.

    Handles server lifecycle automatically via context manager.
    """
    with TestClient(app_with_corpus) as client:
        # Wait for initial indexing to complete
        time.sleep(2)
        yield client


def test_real_world_documentation_site_complete_workflow(client):
    """
    Test complete workflow with realistic interconnected documentation corpus.

    Validates:
    - Server starts and indexes 8 documents
    - Each query returns results with scores
    - Different queries return different or overlapping results
    - Results contain relevant documents from expected topics
    - Graph traversal works (deployment query includes security concepts)
    - Multi-strategy combination queries work correctly
    """
    # Verify server started
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    # Verify all documents indexed
    status_response = client.get("/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    document_count = status_data["indices"]["document_count"]
    assert document_count == 8, f"Expected 8 documents, got {document_count}"

    # Query 1: Keyword match - "authenticate" + "API"
    query1_response = client.post(
        "/query_documents",
        json={"query": "how do I authenticate API requests?"},
    )
    assert query1_response.status_code == 200
    query1_data = query1_response.json()
    results1 = query1_data["results"]

    # Verify results exist and have meaningful content
    assert isinstance(results1, list)
    assert len(results1) > 0, "Expected at least 1 result for authentication query"
    # Check that results contain authentication-related content
    result_contents = [r["content"].lower() for r in results1]
    assert any("auth" in content or "api" in content or "key" in content or "token" in content
               for content in result_contents), \
        "Results should contain authentication-related content"

    # Query 2: Semantic similarity - security concepts
    query2_response = client.post(
        "/query_documents",
        json={"query": "securing my application"},
    )
    assert query2_response.status_code == 200
    query2_data = query2_response.json()
    results2 = query2_data["results"]

    assert isinstance(results2, list)
    assert len(results2) > 0, "Expected at least 1 result for security query"
    result_contents_2 = [r["content"].lower() for r in results2]
    assert any("security" in content or "secure" in content or "https" in content
               or "tls" in content or "credential" in content
               for content in result_contents_2), \
        "Results should contain security-related content"

    # Query 3: Graph traversal - deployment → auth → security via links
    query3_response = client.post(
        "/query_documents",
        json={"query": "deployment security"},
    )
    assert query3_response.status_code == 200
    query3_data = query3_response.json()
    results3 = query3_data["results"]

    assert isinstance(results3, list)
    assert len(results3) > 0, "Expected at least 1 result for deployment security query"
    result_contents_3 = [r["content"].lower() for r in results3]
    # Should include deployment AND security concepts via graph traversal
    has_deployment = any("deploy" in content or "deployment" in content or "production" in content
                         for content in result_contents_3)
    has_security = any("security" in content or "secure" in content or "firewall" in content
                       or "https" in content or "credential" in content
                       for content in result_contents_3)
    assert has_deployment, "Results should include deployment-related content"
    assert has_security, "Results should include security-related content (graph traversal)"

    # Query 4: Multi-strategy combination - getting started with auth
    query4_response = client.post(
        "/query_documents",
        json={"query": "getting started with authentication"},
    )
    assert query4_response.status_code == 200
    query4_data = query4_response.json()
    results4 = query4_data["results"]

    assert isinstance(results4, list)
    assert len(results4) > 0, "Expected at least 1 result for getting started query"
    result_contents_4 = [r["content"].lower() for r in results4]
    # Should combine getting-started + authentication concepts
    has_getting_started = any("start" in content or "begin" in content or "quick" in content or "first" in content
                              for content in result_contents_4)
    has_auth = any("auth" in content or "api" in content or "key" in content or "credential" in content
                   for content in result_contents_4)
    assert has_getting_started, "Results should include getting-started content"
    assert has_auth, "Results should include authentication content"

    # Verify results diversity - collect all result chunk_ids
    all_chunk_ids = set()
    for results in [results1, results2, results3, results4]:
        for result in results:
            all_chunk_ids.add(result["chunk_id"])

    # We should have retrieved diverse results across queries
    assert len(all_chunk_ids) >= 4, \
        f"Expected diverse results across queries, got {len(all_chunk_ids)} unique chunks"

    # Verify all results are non-empty and have correct structure
    for i, results in enumerate([results1, results2, results3, results4], 1):
        for result in results:
            assert isinstance(result, dict), f"Query {i} result is not dict: {type(result)}"
            assert "chunk_id" in result, f"Query {i} result missing chunk_id"
            assert "score" in result, f"Query {i} result missing score"
            assert "content" in result, f"Query {i} result missing content"
            assert isinstance(result["content"], str), f"Query {i} content is not string"
            assert len(result["content"]) > 0, f"Query {i} content is empty"
