import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class IndexManifest:
    spec_version: str
    embedding_model: str
    parsers: dict[str, str]
    chunking_config: dict[str, Any]
    indexed_files: dict[str, str] | None = None  # doc_id -> relative_file_path


def save_manifest(path: Path, manifest: IndexManifest) -> None:
    manifest_path = path / "index.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "spec_version": manifest.spec_version,
        "embedding_model": manifest.embedding_model,
        "parsers": manifest.parsers,
        "chunking_config": manifest.chunking_config,
        "indexed_files": manifest.indexed_files or {},
    }

    with manifest_path.open("w") as f:
        json.dump(data, f, indent=2)


def load_manifest(path: Path):
    manifest_path = path / "index.manifest.json"

    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r") as f:
            data = json.load(f)

        return IndexManifest(
            spec_version=data["spec_version"],
            embedding_model=data["embedding_model"],
            parsers=data["parsers"],
            chunking_config=data.get("chunking_config", {}),
            indexed_files=data.get("indexed_files"),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def should_rebuild(current: IndexManifest, saved: Optional[IndexManifest]):
    if saved is None:
        return True

    # Missing indexed_files triggers a one-time rebuild to populate it
    if saved.indexed_files is None:
        return True

    return (
        current.spec_version != saved.spec_version
        or current.embedding_model != saved.embedding_model
        or current.chunking_config != saved.chunking_config
        or current.parsers != saved.parsers
    )
