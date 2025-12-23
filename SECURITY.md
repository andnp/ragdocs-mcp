# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately.

**DO NOT** open a public issue for security vulnerabilities.

### How to Report

Send details to the maintainers via one of these methods:

1. **GitHub Security Advisory**: Use the "Security" tab in this repository to create a private security advisory
2. **Email**: Contact the maintainers directly (if email addresses are provided in the project)

### What to Include

Your report should include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)
- Your contact information for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with assessment and planned response
- **Fix Timeline**: Depends on severity:
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Within 90 days

### Disclosure Policy

After a fix is released, we will:

1. Credit the reporter (unless anonymity is requested)
2. Publish a security advisory detailing the vulnerability
3. Update the changelog with security fix information

## Security Considerations

### Local Deployment

This server is designed for local or trusted network deployment. Security considerations include:

**File Access**
- The server reads markdown files from configured directories
- Avoid pointing `documents_path` to sensitive system directories
- Restrict `documents_path` to specific documentation folders
- Use appropriate filesystem permissions on indexed directories

**Network Exposure**
- Default binding is `127.0.0.1` (localhost only)
- Only bind to `0.0.0.0` on trusted networks
- Use reverse proxy (nginx, Caddy) with authentication for remote access
- Consider TLS termination at the reverse proxy layer

**Index Storage**
- Vector indices and metadata are stored in `index_path`
- Ensure appropriate filesystem permissions on index directories
- Index files may contain document content in embedded form

**Dependencies**
- Keep dependencies updated: `uv sync --upgrade`
- Monitor security advisories for llama-index, FastAPI, and other dependencies
- Run with minimal required permissions

### Production Deployment

For production deployments:

1. **Use systemd service** with hardening options (see `deployment/mcp-ragdocs.service`)
2. **Dedicated user account** with minimal privileges
3. **Filesystem restrictions**:
   - ReadOnly access to document directories
   - ReadWrite access only to index storage
4. **Network isolation**:
   - Bind to localhost by default
   - Use reverse proxy for remote access
   - Implement authentication at proxy layer
5. **Resource limits**:
   - Set memory and CPU limits via systemd or container orchestration
6. **Monitoring**:
   - Monitor `/health` and `/status` endpoints
   - Log security-relevant events
   - Set up alerts for service failures

### Known Limitations

- No built-in authentication mechanism
- No rate limiting on query endpoints
- No input sanitization for file paths (relies on filesystem permissions)
- Embedding model runs locally (memory requirements)

For questions about security, open a discussion in the repository's Discussions tab.
