---
name: Feature Request
about: Suggest a new feature or enhancement
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

## Problem

A clear description of the problem this feature would solve.

Example: "I need to search across multiple document collections but currently can only index one directory."

## Proposed Solution

Describe your proposed solution.

Example: "Add support for multiple `documents_path` entries in config.toml, each with its own namespace."

## Alternatives Considered

What alternatives have you considered?

Example: "Running multiple server instances, but this requires managing multiple processes and ports."

## Use Case

How would you use this feature?

```toml
# Example configuration or usage
[indexing]
collections = [
    { name = "project-docs", path = "./docs" },
    { name = "personal-notes", path = "~/notes" }
]
```

## Additional Context

- Would this be a breaking change?
- Are there similar features in other tools?
- Any implementation considerations?
- Related issues or discussions?
