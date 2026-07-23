"""Golden set schema and loader for evaluation.

A golden set is a collection of queries with their ground-truth relevant result IDs.
Used to measure retrieval quality against a benchmark.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoldenEntry:
    """Single evaluation entry: a query with its relevant result IDs."""

    query: str
    """The search query text."""

    relevant_ids: list[str]
    """List of result IDs that are relevant to this query."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "query": self.query,
            "relevant_ids": self.relevant_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenEntry":
        """Deserialize from a dictionary."""
        return cls(
            query=data["query"],
            relevant_ids=data["relevant_ids"],
        )


@dataclass
class GoldenSet:
    """A collection of golden entries for evaluation."""

    entries: list[GoldenEntry]
    """List of evaluation entries."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenSet":
        """Deserialize from a dictionary."""
        entries = [GoldenEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries=entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)


def load_golden(path: str | Path) -> GoldenSet:
    """Load a golden set from a JSON file.

    Expected JSON format:
    ```json
    {
      "entries": [
        {
          "query": "search query text",
          "relevant_ids": ["result_id_1", "result_id_2"]
        },
        ...
      ]
    }
    ```

    Args:
        path: Path to the JSON file.

    Returns:
        A GoldenSet instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("Golden set JSON must contain an 'entries' key with a list value")

    return GoldenSet.from_dict(data)


def save_golden(golden_set: GoldenSet, path: str | Path) -> None:
    """Save a golden set to a JSON file.

    Args:
        golden_set: The GoldenSet to save.
        path: Path to write the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(golden_set.to_dict(), f, indent=2)
