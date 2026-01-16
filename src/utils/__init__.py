import fnmatch

from src.utils.similarity import cosine_similarity, cosine_similarity_lists


def should_include_file(
    file_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    exclude_hidden_dirs: bool = True,
):
    normalized_path = file_path.replace("\\", "/")

    if exclude_hidden_dirs:
        path_parts = normalized_path.split("/")
        for part in path_parts:
            if part and part.startswith("."):
                return False

    included = False
    for pattern in include_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            included = True
            break

    if not included:
        return False

    for pattern in exclude_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            return False

    return True


__all__ = [
    "should_include_file",
    "cosine_similarity",
    "cosine_similarity_lists",
]
