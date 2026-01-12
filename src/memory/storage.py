from pathlib import Path


def ensure_memory_dirs(memory_path: Path) -> None:
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "indices").mkdir(exist_ok=True)
    (memory_path / ".trash").mkdir(exist_ok=True)


def get_memory_file_path(memory_path: Path, filename: str) -> Path:
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    return memory_path / filename


def get_trash_path(memory_path: Path) -> Path:
    return memory_path / ".trash"


def get_indices_path(memory_path: Path) -> Path:
    return memory_path / "indices"


def list_memory_files(memory_path: Path) -> list[Path]:
    if not memory_path.exists():
        return []
    return [
        f for f in memory_path.glob("*.md")
        if f.is_file() and not f.name.startswith(".")
    ]


def compute_memory_id(memory_path: Path, file_path: Path) -> str:
    try:
        rel_path = file_path.relative_to(memory_path)
        return f"memory:{rel_path.with_suffix('')}"
    except ValueError:
        return f"memory:{file_path.stem}"
