from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast
import os
import re
import logging
import tomllib
import tomlkit
import tempfile

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = ".index_data/"


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
    embedding_workers: int = 4
    torch_num_threads: int = 4
    debounce_window_seconds: float = 0.5
    task_backpressure_limit: int = 100
    rebuild_checkpoint_interval: int = 25
    delta_full_reindex_threshold: float = 0.5
    move_detection_threshold: float = 0.8


@dataclass
class SearchConfig:
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    recency_bias: float = 0.5
    min_confidence: float = 0.3
    max_chunks_per_doc: int = 2
    dedup_threshold: float = 0.80
    reranking_enabled: bool = True
    rerank_top_n: int = 10


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
    poll_interval_seconds: float = 30.0


@dataclass
class Config:
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    git_indexing: GitIndexingConfig = field(default_factory=GitIndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    projects: list[ProjectConfig] = field(default_factory=list)
    detected_project: str | None = None
    config_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AutoRegistrationResult:
    changed: bool
    project_name: str | None = None
    project_path: str | None = None
    reason: str | None = None


def _expand_path(path_str: str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


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


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _find_nearest_project_config_root(cwd: Path) -> Path | None:
    current = cwd.resolve()

    while True:
        config_path = current / ".mcp-markdown-ragdocs" / "config.toml"
        if config_path.exists():
            return current

        parent = current.parent
        if parent == current:
            return None

        current = parent


def _find_nearest_git_root(cwd: Path) -> Path | None:
    current = cwd.resolve()

    while True:
        git_path = current / ".git"
        if git_path.exists():
            return current

        parent = current.parent
        if parent == current:
            return None

        current = parent


def derive_auto_registration_root(cwd: Path | None = None) -> Path:
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()

    config_root = _find_nearest_project_config_root(resolved_cwd)
    if config_root is not None:
        return config_root

    git_root = _find_nearest_git_root(resolved_cwd)
    if git_root is not None:
        return git_root

    return resolved_cwd


def _is_unsafe_auto_registration_root(root: Path) -> bool:
    resolved_root = root.resolve()
    home_path = Path.home().resolve()

    forbidden_roots = {
        Path("/").resolve(),
        home_path,
        Path(tempfile.gettempdir()).resolve(),
        Path("/etc").resolve(),
        Path("/run").resolve(),
        Path("/var/run").resolve(),
        Path("/var/tmp").resolve(),
        Path(os.environ.get("XDG_CONFIG_HOME", home_path / ".config")).resolve(),
        Path(
            os.environ.get("XDG_STATE_HOME", home_path / ".local" / "state")
        ).resolve(),
        _global_config_path().parent.resolve(),
    }

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        forbidden_roots.add(Path(xdg_runtime_dir).resolve())

    return resolved_root in forbidden_roots


def ensure_runtime_project_registered(
    cwd: Path | None = None,
    project_override: str | None = None,
) -> AutoRegistrationResult:
    if project_override:
        return AutoRegistrationResult(
            changed=False,
            reason="explicit_project_override",
        )

    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()
    registered_projects = _load_global_projects()

    detected_registered_project = detect_project(
        cwd=resolved_cwd,
        projects=registered_projects,
    )
    if detected_registered_project is not None:
        return AutoRegistrationResult(
            changed=False,
            project_name=detected_registered_project,
            reason="already_registered",
        )

    candidate_root = derive_auto_registration_root(resolved_cwd).resolve()
    if _is_unsafe_auto_registration_root(candidate_root):
        return AutoRegistrationResult(
            changed=False,
            project_path=str(candidate_root),
            reason="unsafe_root",
        )

    for project in registered_projects:
        project_path = Path(project.path).resolve()
        if project_path == candidate_root:
            return AutoRegistrationResult(
                changed=False,
                project_name=project.name,
                project_path=str(candidate_root),
                reason="already_registered",
            )

        if _path_is_relative_to(candidate_root, project_path):
            return AutoRegistrationResult(
                changed=False,
                project_name=project.name,
                project_path=str(project_path),
                reason="inside_registered_project",
            )

        if project_path != candidate_root and _path_is_relative_to(
            project_path, candidate_root
        ):
            return AutoRegistrationResult(
                changed=False,
                project_path=str(candidate_root),
                reason="contains_registered_project",
            )

    project_name = _generate_unique_project_name(
        candidate_root.name,
        [project.name for project in registered_projects],
    )
    persist_project_to_config(project_name, str(candidate_root))
    return AutoRegistrationResult(
        changed=True,
        project_name=project_name,
        project_path=str(candidate_root),
        reason="registered",
    )


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

    chunking_data = config_data.get(
        "chunking", config_data.get("chunking_documents", {})
    )
    chunking = _load_dataclass_from_dict(ChunkingConfig, chunking_data)

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
        search=search,
        llm=llm,
        chunking=chunking,
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


def persist_project_to_config(project_name: str, project_path: str):
    global_config_path = _global_config_path()

    global_config_path.parent.mkdir(parents=True, exist_ok=True)

    doc: Any
    if global_config_path.exists():
        with open(global_config_path, "r") as f:
            doc = tomlkit.load(f)
    else:
        doc = tomlkit.document()

    if "projects" not in doc:
        doc["projects"] = tomlkit.aot()

    projects_array: Any = doc["projects"]
    if not isinstance(projects_array, list):
        from tomlkit.items import AoT

        projects_array = AoT([])
        doc["projects"] = projects_array

    # Cast to avoid type checker issues with tomlkit types
    projects_list = cast(list[Any], projects_array)

    for proj_item in projects_list:
        proj = cast(dict[str, Any], proj_item)
        if proj.get("name") == project_name:
            logger.debug(f"Project '{project_name}' already exists in config")
            return
        if proj.get("path") == project_path:
            logger.debug(f"Project path '{project_path}' already registered")
            return

    new_project: Any = tomlkit.table()
    new_project["name"] = project_name
    new_project["path"] = project_path
    projects_list.append(new_project)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=global_config_path.parent, delete=False, suffix=".tmp"
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tomlkit.dump(doc, tmp_file)

    tmp_path.replace(global_config_path)
    logger.info(f"Persisted project '{project_name}' to config: {project_path}")


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
