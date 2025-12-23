import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IndexManifest:
    spec_version: str
    embedding_model: str
    parsers: dict[str, str]


def save_manifest(path: Path, manifest: IndexManifest) -> None:
    manifest_path = path / "index.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "spec_version": manifest.spec_version,
        "embedding_model": manifest.embedding_model,
        "parsers": manifest.parsers,
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
        )
    except (json.JSONDecodeError, KeyError):
        return None


def should_rebuild(current: IndexManifest, saved: Optional[IndexManifest]):
    if saved is None:
        return True

    return (
        current.spec_version != saved.spec_version
        or current.embedding_model != saved.embedding_model
        or current.parsers != saved.parsers
    )
