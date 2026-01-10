from pathlib import Path


def normalize_path(file_path: str, docs_root: Path):
    path = Path(file_path)

    if path.is_absolute():
        try:
            path = path.relative_to(docs_root)
        except ValueError:
            pass

    return str(path.with_suffix(""))


def matches_any_excluded(file_path: str, excluded_files: set[str], docs_root: Path):
    normalized = normalize_path(file_path, docs_root)

    if normalized in excluded_files:
        return True

    filename = Path(normalized).name
    if filename in excluded_files:
        return True

    return False
