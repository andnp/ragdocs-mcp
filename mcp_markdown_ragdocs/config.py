import logging
import importlib.util
import math
import os
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from searchkernel.api import (
    TEST_FAKE_EMBEDDING_MODEL_NAME,
    should_use_test_fake_embeddings,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = ".index_data/"
DEFAULT_GDRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def _default_gdrive_credentials_path() -> str:
    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return str(
            Path(config_home)
            / "mcp-markdown-ragdocs"
            / "gdrive"
            / "authorized-user.json"
        )
    return str(
        Path.home()
        / ".config"
        / "mcp-markdown-ragdocs"
        / "gdrive"
        / "authorized-user.json"
    )


@dataclass
class ProjectConfig:
    name: str
    path: str

    def __post_init__(self):
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.name):
            raise ValueError(
                f"Invalid project name '{self.name}': "
                "must contain only alphanumeric characters, hyphens, and underscores"
            )

        path_obj = Path(self.path).expanduser()
        if not path_obj.is_absolute():
            raise ValueError(f"Project path '{self.path}' must be absolute")

        self.path = str(path_obj.resolve())


@dataclass
class IndexingConfig:
    documents_path: str = "."
    index_path: str = ".index_data/"
    include: list[str] = field(default_factory=lambda: ["**/*"])
    exclude: list[str] = field(
        default_factory=lambda: [
            "**/.venv/**",
            "**/venv/**",
            "**/build/**",
            "**/dist/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/.codanna/**",
            "**/*-egg-info/**",
            "**/.mcp-markdown-ragdocs/**",
            "**/.stversions/**",
            "**/.worktree/**",
            "**/.worktrees/**",
        ]
    )
    exclude_hidden_dirs: bool = True
    reconciliation_interval_seconds: int = 3600  # 1 hour, 0 to disable
    torch_num_threads: int = 4
    debounce_window_seconds: float = 0.5
    task_backpressure_limit: int = 100
    rebuild_checkpoint_interval: int = 25
    delta_full_reindex_threshold: float = 0.5
    move_detection_threshold: float = 0.8


@dataclass
class SearchConfig:
    """Search settings with an explicit compatibility boundary.

    Only the ranking weights affect canonical search. The remaining fields
    remain loadable for configuration-file compatibility, but are deprecated
    because canonical search applies request policy explicitly.
    """

    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    recency_bias: float = field(
        default=0.5,
        metadata={"deprecated": "canonical search has no recency policy"},
    )
    min_confidence: float = field(
        default=0.3,
        metadata={"deprecated": "use request min_score"},
    )
    abstention_threshold: float | None = field(
        default=None,
        metadata={
            "description": (
                "Optional raw RRF score floor for abstention; calibrate against "
                "labeled queries before enabling."
            )
        },
    )
    max_chunks_per_doc: int = field(
        default=1,
        metadata={"deprecated": "use request uniqueness_mode"},
    )
    dedup_threshold: float = field(
        default=0.80,
        metadata={"deprecated": "canonical search owns deduplication"},
    )
    reranking_enabled: bool = field(
        default=True,
        metadata={"deprecated": "canonical search does not rerank"},
    )
    rerank_top_n: int = field(
        default=10,
        metadata={"deprecated": "canonical search does not rerank"},
    )
    reranker_model: str | None = None
    rerank_budget: int = 0
    project_uplift_multiplier: float = field(
        default=1.2,
        metadata={"deprecated": "project scope is an explicit filter"},
    )

    @classmethod
    def deprecated_policy_fields(cls) -> frozenset[str]:
        return frozenset(
            value.name for value in fields(cls)
            if value.metadata.get("deprecated")
        )

    def __post_init__(self) -> None:
        if not math.isfinite(self.project_uplift_multiplier):
            raise ValueError("search.project_uplift_multiplier must be a finite number")
        if self.project_uplift_multiplier <= 0:
            raise ValueError(
                "search.project_uplift_multiplier must be greater than 0"
            )
        if self.abstention_threshold is not None and not (
            math.isfinite(self.abstention_threshold)
            and 0.0 <= self.abstention_threshold <= 1.0
        ):
            raise ValueError(
                "search.abstention_threshold must be between 0 and 1"
            )
        if self.rerank_budget < 0:
            raise ValueError("search.rerank_budget must be non-negative")
        if self.reranker_model is not None and not self.reranker_model.strip():
            raise ValueError("search.reranker_model must not be empty")
        if self.rerank_budget or self.reranker_model is not None:
            if self.reranker_model is None or self.rerank_budget < 1:
                raise ValueError(
                    "search.reranker_model and search.rerank_budget must be "
                    "configured together"
                )
            if importlib.util.find_spec("sentence_transformers") is None:
                raise ValueError(
                    "search reranking requires the optional "
                    "'sentence-transformers' dependency"
                )


@dataclass
class LLMConfig:
    embedding_model: str = "local"

    DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

    @property
    def resolved_embedding_model(self) -> str:
        """Return actual embedding model name, resolving 'local' to default.

        This centralizes the embedding model resolution logic.
        """
        if self.embedding_model == "local":
            if should_use_test_fake_embeddings():
                return TEST_FAKE_EMBEDDING_MODEL_NAME
            return self.DEFAULT_LOCAL_MODEL
        return self.embedding_model


def resolve_embedding_model(config: "Config") -> str:
    """Resolve embedding model name from config with fallback.

    This function provides a robust way to get the embedding model name,
    handling edge cases where the LLMConfig.resolved_embedding_model property
    might not be accessible (e.g., in subprocess environments with module
    loading edge cases).

    Use this function instead of accessing config.llm.resolved_embedding_model
    directly in contexts where module loading may be unreliable (subprocess,
    worker processes).
    """
    try:
        return config.llm.resolved_embedding_model
    except AttributeError:
        # Fallback: resolve manually if property not accessible
        model = config.llm.embedding_model
        if model == "local":
            if should_use_test_fake_embeddings():
                return TEST_FAKE_EMBEDDING_MODEL_NAME
            return LLMConfig.DEFAULT_LOCAL_MODEL
        return model


@dataclass
class ChunkingConfig:
    strategy: str = "header_based"
    min_chunk_chars: int = 1000
    max_chunk_chars: int = 3000
    overlap_chars: int = 200
    parent_chunk_min_chars: int = 1500
    parent_chunk_max_chars: int = 4000


@dataclass
class GitIndexingConfig:
    enabled: bool = True
    watch_enabled: bool = True
    poll_interval_seconds: float = 300.0


@dataclass
class GoogleDriveConfig:
    enabled: bool = False
    credentials_path: str = field(default_factory=_default_gdrive_credentials_path)
    workspace_id: str = "default"
    shared_drive_ids: tuple[str, ...] = ()
    scopes: tuple[str, ...] = (DEFAULT_GDRIVE_SCOPE,)
    index_generation: str = "gdrive-v1"
    page_size: int = 1000
    batch_size: int = 100
    max_download_bytes: int = 25 * 1024 * 1024
    max_text_bytes: int = 4 * 1024 * 1024
    max_items: int = 100_000
    max_pages: int = 500
    max_seconds: float = 10.0
    request_min_interval_seconds: float = 0.2
    request_max_concurrent: int = 4
    push_enabled: bool = False
    push_address: str = ""
    watch_renewal_seconds: int = 3600

    def __post_init__(self) -> None:
        self.shared_drive_ids = tuple(self.shared_drive_ids)
        self.scopes = tuple(self.scopes)


@dataclass
class FederationConfig:
    deployment_token: str = ""
    drive_workspace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.drive_workspace_ids = tuple(self.drive_workspace_ids)


@dataclass
class StoreConfig:
    backend: str = "local"  # canonical local record stores
    pg_dsn: str = ""  # retained for configuration-file compatibility


@dataclass
class EmbeddingConfig:
    # Embeddings are served by Ollama so worker and query processes do not
    # each load a copy of the model into Python.
    provider: str = "ollama"
    model_name: str = "nomic-embed-text-v2-moe:latest"
    base_url: str = "http://localhost:11434"
    dimension: int | None = None
    timeout_seconds: float = 60.0
    auto_pull: bool = True
    pull_timeout_seconds: float = 600.0
    truncate_dim: int | None = None
    batch_size: int = 64


@dataclass
class LoggingConfig:
    max_bytes: int = 50 * 1024 * 1024
    backup_count: int = 5

    def __post_init__(self) -> None:
        if self.max_bytes < 1:
            raise ValueError("logging.max_bytes must be positive")
        if self.backup_count < 1:
            raise ValueError("logging.backup_count must be positive")


@dataclass
class Config:
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    git_indexing: GitIndexingConfig = field(default_factory=GitIndexingConfig)
    gdrive: GoogleDriveConfig = field(default_factory=GoogleDriveConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    projects: list[ProjectConfig] = field(default_factory=list)
    detected_project: str | None = None
    config_warnings: list[str] = field(default_factory=list)

    def snapshot(self) -> "Config":
        """Return an independent copy for runtime-owned normalization."""
        return deepcopy(self)


@dataclass(frozen=True)
class GitWorktreeCandidate:
    root: Path
    branch: str | None = None


def _expand_path(path_str: str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_gitdir_pointer(pointer: str, base_dir: Path) -> Path | None:
    if not pointer.lower().startswith("gitdir:"):
        return None

    raw_path = pointer.partition(":")[2].strip()
    if not raw_path:
        return None

    git_dir = Path(raw_path).expanduser()
    if not git_dir.is_absolute():
        git_dir = (base_dir / git_dir).resolve()
    else:
        git_dir = git_dir.resolve()

    return git_dir


def _resolve_git_dir(repo_root: Path) -> Path | None:
    git_entry = repo_root / ".git"
    if git_entry.is_dir():
        return git_entry.resolve()

    if not git_entry.is_file():
        return None

    pointer = _read_text_file(git_entry)
    if pointer is None:
        return None

    return _parse_gitdir_pointer(pointer, repo_root)


def _resolve_git_common_dir_from_git_dir(git_dir: Path) -> Path:
    commondir_file = git_dir / "commondir"
    commondir = _read_text_file(commondir_file)
    if commondir is None:
        return git_dir.resolve()

    common_dir = Path(commondir).expanduser()
    if not common_dir.is_absolute():
        common_dir = (git_dir / common_dir).resolve()
    else:
        common_dir = common_dir.resolve()

    return common_dir


def _resolve_git_common_dir(repo_root: Path) -> Path | None:
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None

    return _resolve_git_common_dir_from_git_dir(git_dir)


def _read_symbolic_branch(head_path: Path) -> str | None:
    head_value = _read_text_file(head_path)
    if head_value is None or not head_value.startswith("ref:"):
        return None

    ref = head_value.partition(":")[2].strip()
    if not ref:
        return None

    return Path(ref).name


def _git_dir_is_bare(git_dir: Path) -> bool:
    config_text = _read_text_file(git_dir / "config")
    if config_text is None:
        return False

    return re.search(r"(?mi)^\s*bare\s*=\s*true\s*$", config_text) is not None


def _discover_main_worktree(common_git_dir: Path) -> GitWorktreeCandidate | None:
    if _git_dir_is_bare(common_git_dir):
        return None

    main_root = common_git_dir.parent.resolve()
    if _resolve_git_dir(main_root) != common_git_dir.resolve():
        return None

    return GitWorktreeCandidate(
        root=main_root,
        branch=_read_symbolic_branch(common_git_dir / "HEAD"),
    )


def _discover_linked_worktrees(common_git_dir: Path) -> list[GitWorktreeCandidate]:
    worktrees_dir = common_git_dir / "worktrees"
    if not worktrees_dir.is_dir():
        return []

    candidates: list[GitWorktreeCandidate] = []
    for worktree_dir in sorted(worktrees_dir.iterdir(), key=lambda path: path.name):
        if not worktree_dir.is_dir():
            continue

        gitdir_pointer = _read_text_file(worktree_dir / "gitdir")
        if gitdir_pointer is None:
            continue

        worktree_git_file = Path(gitdir_pointer).expanduser()
        if not worktree_git_file.is_absolute():
            worktree_git_file = (worktree_dir / worktree_git_file).resolve()
        else:
            worktree_git_file = worktree_git_file.resolve()

        candidates.append(
            GitWorktreeCandidate(
                root=worktree_git_file.parent.resolve(),
                branch=_read_symbolic_branch(worktree_dir / "HEAD"),
            )
        )

    return candidates


def _list_git_worktrees(repo_root: Path) -> tuple[Path | None, list[GitWorktreeCandidate]]:
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None, []

    common_git_dir = _resolve_git_common_dir_from_git_dir(git_dir)
    deduped_candidates: dict[Path, GitWorktreeCandidate] = {}

    main_worktree = _discover_main_worktree(common_git_dir)
    if main_worktree is not None:
        deduped_candidates[main_worktree.root] = main_worktree

    for candidate in _discover_linked_worktrees(common_git_dir):
        deduped_candidates[candidate.root] = candidate

    if not deduped_candidates:
        deduped_candidates[repo_root.resolve()] = GitWorktreeCandidate(
            root=repo_root.resolve(),
            branch=_read_symbolic_branch(git_dir / "HEAD"),
        )

    return common_git_dir, sorted(
        deduped_candidates.values(),
        key=lambda candidate: str(candidate.root),
    )


def _select_canonical_worktree(repo_root: Path) -> Path | None:
    common_git_dir, worktrees = _list_git_worktrees(repo_root)
    if common_git_dir is None or not worktrees:
        return None

    preferred_branch = _read_symbolic_branch(common_git_dir / "HEAD")
    preferred_names = {"main", "master"}
    current_root = repo_root.resolve()

    def _sort_key(candidate: GitWorktreeCandidate):
        return (
            0 if candidate.root.name in preferred_names else 1,
            0 if candidate.branch in preferred_names else 1,
            0
            if preferred_branch is not None
            and (
                candidate.branch == preferred_branch
                or candidate.root.name == preferred_branch
            )
            else 1,
            0 if candidate.root == current_root else 1,
            str(candidate.root),
        )

    return min(worktrees, key=_sort_key).root


def _load_dataclass_from_dict[T](
    cls: type[T], data: dict[str, Any], path_fields: set[str] | None = None
) -> T:
    if path_fields is None:
        path_fields = set()

    kwargs: dict[str, Any] = {}
    for f in fields(cast(type, cls)):
        if f.name not in data:
            continue

        value = data[f.name]

        if (
            is_dataclass(f.type)
            and isinstance(f.type, type)
            and isinstance(value, dict)
        ):
            value = _load_dataclass_from_dict(f.type, value)
        elif f.name in path_fields and isinstance(value, str):
            value = _expand_path(value)

        kwargs[f.name] = value

    return cls(**kwargs)


def _find_project_config():
    current = Path.cwd()

    while True:
        config_path = current / ".mcp-markdown-ragdocs" / "config.toml"
        if config_path.exists():
            return config_path

        parent = current.parent
        if parent == current:
            return None

        current = parent


def _global_config_path() -> Path:
    return Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"


def _load_projects_from_data(projects_data: list[dict[str, Any]]) -> list[ProjectConfig]:
    projects: list[ProjectConfig] = []
    for proj_data in projects_data:
        try:
            projects.append(
                ProjectConfig(name=proj_data["name"], path=proj_data["path"])
            )
        except (KeyError, ValueError) as e:
            logger.warning(
                f"Skipping invalid project config: {e}. Project data: {proj_data}"
            )
    return projects


def _load_global_projects() -> list[ProjectConfig]:
    global_config_path = _global_config_path()
    if not global_config_path.exists():
        return []

    with open(global_config_path, "rb") as f:
        config_data = tomllib.load(f)

    projects_data = config_data.get("projects", [])
    if not isinstance(projects_data, list):
        return []

    return _load_projects_from_data(projects_data)


def _find_nearest_git_root_candidate(cwd: Path) -> Path | None:
    current = cwd.resolve()

    while True:
        git_path = current / ".git"
        if git_path.exists():
            return current

        parent = current.parent
        if parent == current:
            return None

        current = parent


def _detect_project_from_related_git_repo(
    cwd: Path,
    projects: list[ProjectConfig],
) -> str | None:
    repo_root = _find_nearest_git_root_candidate(cwd)
    if repo_root is None:
        return None

    common_git_dir = _resolve_git_common_dir(repo_root)
    if common_git_dir is None:
        return None

    related_projects = [
        project
        for project in projects
        if _resolve_git_common_dir(Path(project.path).resolve()) == common_git_dir
    ]
    if not related_projects:
        return None

    canonical_root = _select_canonical_worktree(repo_root)
    if canonical_root is not None:
        for project in related_projects:
            if Path(project.path).resolve() == canonical_root:
                logger.info(
                    "Detected project %s via shared git worktree identity",
                    project.name,
                )
                return project.name

    fallback_project = related_projects[0]
    logger.info(
        "Detected project %s via fallback shared git worktree identity",
        fallback_project.name,
    )
    return fallback_project.name


def load_config():
    config_locations = []

    project_config = _find_project_config()
    if project_config:
        config_locations.append(project_config)

    config_locations.append(
        Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    )

    config_data: dict[str, Any] = {}
    for config_path in config_locations:
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
            break

    indexing = _load_dataclass_from_dict(
        IndexingConfig,
        config_data.get("indexing", {}),
        path_fields={"documents_path", "index_path"},
    )
    # Always expand paths (defaults may be relative)
    indexing.documents_path = _expand_path(indexing.documents_path)
    indexing.index_path = _expand_path(indexing.index_path)

    search = _load_dataclass_from_dict(SearchConfig, config_data.get("search", {}))
    llm = _load_dataclass_from_dict(LLMConfig, config_data.get("llm", {}))
    git_indexing = _load_dataclass_from_dict(
        GitIndexingConfig, config_data.get("git_indexing", {})
    )
    gdrive = _load_dataclass_from_dict(
        GoogleDriveConfig,
        config_data.get("gdrive", {}),
        path_fields={"credentials_path"},
    )
    gdrive.credentials_path = _expand_path(gdrive.credentials_path)
    federation = _load_dataclass_from_dict(
        FederationConfig, config_data.get("federation", {})
    )
    if not federation.deployment_token:
        federation.deployment_token = os.environ.get("RAGDOCS_FEDERATION_TOKEN", "")

    chunking_data = config_data.get(
        "chunking", config_data.get("chunking_documents", {})
    )
    if not isinstance(chunking_data, dict):
        chunking_data = {}
    chunking = _load_dataclass_from_dict(ChunkingConfig, chunking_data)

    # Load store config; pg_dsn defaults to env var if not in config
    store_data = config_data.get("store", {})
    store = _load_dataclass_from_dict(StoreConfig, store_data)
    if not store.pg_dsn:
        store.pg_dsn = os.environ.get("SEARCHKERNEL_PG_DSN", "")

    embedding = _load_dataclass_from_dict(
        EmbeddingConfig, config_data.get("embedding", {})
    )
    logging_config = _load_dataclass_from_dict(
        LoggingConfig, config_data.get("logging", {})
    )

    projects_data = config_data.get("projects", [])
    projects = []
    if isinstance(projects_data, list) and projects_data:
        projects = _load_projects_from_data(projects_data)

    _validate_projects(projects)
    config_warnings = get_project_root_warnings(projects)
    for warning in config_warnings:
        logger.warning(f"Configuration warning: {warning}")

    return Config(
        indexing=indexing,
        git_indexing=git_indexing,
        gdrive=gdrive,
        federation=federation,
        search=search,
        llm=llm,
        chunking=chunking,
        store=store,
        embedding=embedding,
        logging=logging_config,
        projects=projects,
        config_warnings=config_warnings,
    )


def _validate_projects(projects: list[ProjectConfig]):
    names = [p.name for p in projects]
    if len(names) != len(set(names)):
        dupes = [name for name in names if names.count(name) > 1]
        raise ValueError(
            f"Duplicate project names found: {', '.join(set(dupes))}. "
            "Each project must have a unique name."
        )

    paths = [p.path for p in projects]
    if len(paths) != len(set(paths)):
        dupes = [path for path in paths if paths.count(path) > 1]
        raise ValueError(
            f"Duplicate project paths found: {', '.join(set(dupes))}. "
            "Each project must have a unique path."
        )


def get_project_root_warnings(projects: list[ProjectConfig]):
    warnings: list[str] = []
    home_path = Path.home().resolve()
    resolved_paths = {
        project.name: Path(project.path).resolve() for project in projects
    }

    for project in projects:
        project_path = resolved_paths[project.name]

        if project_path == home_path:
            warnings.append(
                f"Project '{project.name}' path '{project.path}' is the current user's home directory."
            )

        if project_path.parent == project_path:
            warnings.append(
                f"Project '{project.name}' path '{project.path}' is the filesystem root."
            )

        contained_projects: list[str] = []
        for other_project in projects:
            if other_project.name == project.name:
                continue

            other_path = resolved_paths[other_project.name]
            try:
                other_path.relative_to(project_path)
            except ValueError:
                continue

            if other_path != project_path:
                contained_projects.append(other_project.name)

        if contained_projects:
            child_projects = ", ".join(sorted(contained_projects))
            warnings.append(
                f"Project '{project.name}' path '{project.path}' contains other registered project roots: {child_projects}."
            )

    projects_by_repo_identity: dict[Path, list[ProjectConfig]] = {}
    for project in projects:
        common_git_dir = _resolve_git_common_dir(resolved_paths[project.name])
        if common_git_dir is None:
            continue
        projects_by_repo_identity.setdefault(common_git_dir, []).append(project)

    for common_git_dir, related_projects in sorted(
        projects_by_repo_identity.items(), key=lambda item: str(item[0])
    ):
        if len(related_projects) < 2:
            continue

        related_names = ", ".join(sorted(project.name for project in related_projects))
        warnings.append(
            f"Registered projects {related_names} point to git worktrees from the same repository identity '{common_git_dir}'."
        )

    return warnings


def _generate_unique_project_name(base_name: str, existing_names: list[str]):
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", base_name)
    name = re.sub(r"-+", "-", name).strip("-")

    if not name or not re.match(r"^[a-zA-Z0-9_-]+$", name):
        name = "project"

    if name not in existing_names:
        return name

    counter = 2
    while f"{name}-{counter}" in existing_names:
        counter += 1

    return f"{name}-{counter}"


def detect_project(
    cwd: Path | None = None,
    projects: list[ProjectConfig] | None = None,
    project_override: str | None = None,
):
    if project_override:
        if projects is None:
            projects = _load_global_projects()

        if projects:
            for project in projects:
                if project.name == project_override:
                    logger.info(
                        f"Using project from --project flag: {project.name} (path: {project.path})"
                    )
                    return project.name

            project_path = Path(project_override).expanduser().resolve()
            for project in projects:
                if Path(project.path).resolve() == project_path:
                    logger.info(
                        f"Using project from --project flag (matched by path): {project.name}"
                    )
                    return project.name

            # Check if project_override path is a subdirectory of a known project (deepest-match-wins)
            projects_sorted = sorted(
                projects, key=lambda p: len(Path(p.path).parts), reverse=True
            )
            for project in projects_sorted:
                project_path_resolved = Path(project.path).resolve()
                try:
                    project_path.relative_to(project_path_resolved)
                    logger.info(
                        f"Using project from --project flag (subdirectory of '{project.name}'): {project.path}"
                    )
                    return project.name
                except ValueError:
                    continue

        project_path = Path(project_override).expanduser().resolve()
        if project_path.exists():
            logger.info(
                f"Using transient path from --project flag without persisting: {project_path}"
            )

            existing_names = [p.name for p in (projects or [])]
            return _generate_unique_project_name(project_path.name, existing_names)

        logger.warning(
            f"Project override '{project_override}' not found in registry and is not a valid path"
        )
        return None

    if cwd is None:
        cwd = Path.cwd()

    if projects is None:
        projects = _load_global_projects()

    if not projects:
        projects = []

    cwd_resolved = cwd.resolve()

    projects_sorted = sorted(
        projects, key=lambda p: len(Path(p.path).parts), reverse=True
    )

    for project in projects_sorted:
        project_path = Path(project.path).resolve()

        try:
            cwd_resolved.relative_to(project_path)
            logger.info(f"Detected project: {project.name} (path: {project.path})")
            return project.name
        except ValueError:
            continue

    related_git_project = _detect_project_from_related_git_repo(cwd_resolved, projects)
    if related_git_project is not None:
        return related_git_project

    logger.debug(f"No project match for CWD: {cwd_resolved}")
    return None


def resolve_index_path(config: Config):
    index_path_str = config.indexing.index_path

    expanded = Path(index_path_str).expanduser()
    if not expanded.is_absolute():
        expanded = expanded.resolve()

    default_resolved = Path(DEFAULT_INDEX_PATH).resolve()
    if expanded != default_resolved:
        logger.info(f"Using explicit index path from config: {expanded}")
        return expanded

    data_home = os.getenv("XDG_DATA_HOME")
    if data_home:
        base_dir = Path(data_home)
    else:
        base_dir = Path.home() / ".local" / "share"

    index_path = base_dir / "mcp-markdown-ragdocs"
    logger.info(f"Using global data directory: {index_path}")
    return index_path


def resolve_documents_path(
    config: Config,
    detected_project: str | None = None,
    projects: list[ProjectConfig] | None = None,
) -> str:
    documents_path_str = config.indexing.documents_path
    documents_path = Path(documents_path_str).expanduser()

    # If already absolute, use as-is
    if documents_path.is_absolute():
        logger.info(f"Using explicit absolute documents path: {documents_path}")
        return str(documents_path)

    # Otherwise resolve relative to CWD
    resolved_path = documents_path.resolve()
    logger.info(f"Using documents path relative to CWD: {resolved_path}")
    return str(resolved_path)


def resolve_project_id_for_path(file_path: Path, config: Config) -> str | None:
    resolved_file_path = file_path.expanduser().resolve()

    projects_sorted = sorted(
        config.projects,
        key=lambda project: len(Path(project.path).parts),
        reverse=True,
    )
    for project in projects_sorted:
        project_path = Path(project.path).resolve()
        try:
            resolved_file_path.relative_to(project_path)
            return project.name
        except ValueError:
            continue

    if config.detected_project:
        documents_path = Path(config.indexing.documents_path).expanduser().resolve()
        try:
            resolved_file_path.relative_to(documents_path)
            return config.detected_project
        except ValueError:
            pass

    return None
