import fnmatch
from pathlib import Path

from src.utils.similarity import cosine_similarity, cosine_similarity_lists


def should_include_file(
    file_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
    documents_roots: list[str | Path] | None = None,
):
    normalized_path = file_path.replace("\\", "/")
    match_path = normalized_path

    if documents_roots:
        resolved_path = Path(file_path).resolve()
        for root in documents_roots:
            try:
                match_path = str(
                    resolved_path.relative_to(Path(root).expanduser().resolve())
                ).replace("\\", "/")
                break
            except ValueError:
                continue

    if exclude_hidden_dirs:
        path_parts = match_path.split("/")
        for part in path_parts:
            if part and part.startswith("."):
                return False

    match_candidates = {normalized_path, match_path}
    if match_path and not match_path.startswith("/"):
        match_candidates.add(f"/{match_path}")

    included = False
    for pattern in include_patterns:
        if any(fnmatch.fnmatch(candidate, pattern) for candidate in match_candidates):
            included = True
            break

    if not included:
        return False

    for pattern in exclude_patterns:
        if any(fnmatch.fnmatch(candidate, pattern) for candidate in match_candidates):
            return False

    return True


__all__ = [
    "should_include_file",
    "cosine_similarity",
    "cosine_similarity_lists",
]
