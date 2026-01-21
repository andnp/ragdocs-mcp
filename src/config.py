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
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class IndexingConfig:
    documents_path: str = "."
    index_path: str = ".index_data/"
    recursive: bool = True
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
    coordination_mode: str = "file_lock"
    lock_timeout_seconds: float = 5.0


@dataclass
class SearchConfig:
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    recency_bias: float = 0.5
    rrf_k_constant: int = 60
    min_confidence: float = 0.3
    max_chunks_per_doc: int = 2
    dedup_enabled: bool = True
    dedup_similarity_threshold: float = 0.80
    ngram_dedup_enabled: bool = True
    ngram_dedup_threshold: float = 0.7
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10
    adaptive_weights_enabled: bool = True
    code_search_enabled: bool = False
    code_search_weight: float = 1.0
    query_expansion_enabled: bool = True
    query_expansion_max_terms: int = 2000
    query_expansion_min_frequency: int = 3
    community_detection_enabled: bool = True
    community_boost_factor: float = 1.1
    dynamic_weights_enabled: bool = True
    variance_threshold: float = 0.1
    min_weight_factor: float = 0.5
    hyde_enabled: bool = True
    tag_expansion_enabled: bool = True
    tag_expansion_max_tags: int = 5
    tag_expansion_depth: int = 2
    score_calibration_threshold: float = 0.035
    score_calibration_steepness: float = 150.0


@dataclass
class LLMConfig:
    embedding_model: str = "local"


@dataclass
class ChunkingConfig:
    strategy: str = "header_based"
    min_chunk_chars: int = 1000
    max_chunk_chars: int = 3000
    overlap_chars: int = 200
    include_parent_headers: bool = True
    parent_retrieval_enabled: bool = True
    parent_chunk_min_chars: int = 1500
    parent_chunk_max_chars: int = 4000


@dataclass
class GitIndexingConfig:
    enabled: bool = True
    delta_max_lines: int = 200
    batch_size: int = 100
    watch_enabled: bool = True
    watch_cooldown: float = 5.0
    parallel_workers: int = 4
    embed_batch_size: int = 32


@dataclass
class MemoryDecayConfig:
    decay_rate: float
    floor_multiplier: float

    def __post_init__(self):
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0.0, 1.0], got {self.decay_rate}")
        if not (0.0 <= self.floor_multiplier <= 1.0):
            raise ValueError(
                f"floor_multiplier must be in [0.0, 1.0], got {self.floor_multiplier}"
            )


@dataclass
class MemoryConfig:
    enabled: bool = True
    storage_strategy: str = "user"
    score_threshold: float = 0.2
    decay_journal: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(0.90, 0.1)
    )
    decay_plan: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(0.85, 0.1)
    )
    decay_fact: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(0.98, 0.2)
    )
    decay_observation: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(0.92, 0.15)
    )
    decay_reflection: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(0.95, 0.2)
    )
    # Deprecated fields (for backward compatibility)
    recency_boost_days: int | None = None
    recency_boost_factor: float | None = None

    def __post_init__(self):
        if self.storage_strategy not in ("project", "user"):
            raise ValueError(
                f"Invalid storage_strategy '{self.storage_strategy}': "
                "must be 'project' or 'user'"
            )

        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError(
                f"score_threshold must be in [0.0, 1.0], got {self.score_threshold}"
            )

        # Warn about deprecated fields
        if self.recency_boost_days is not None:
            logger.warning(
                "Config field 'recency_boost_days' is deprecated. "
                "Use per-type decay configs instead (e.g., decay_journal)."
            )
        if self.recency_boost_factor is not None:
            logger.warning(
                "Config field 'recency_boost_factor' is deprecated. "
                "Use per-type decay configs instead (e.g., decay_journal)."
            )

    def get_decay_config(self, memory_type: str):
        decay_configs = {
            "journal": self.decay_journal,
            "plan": self.decay_plan,
            "fact": self.decay_fact,
            "observation": self.decay_observation,
            "reflection": self.decay_reflection,
        }
        if memory_type in decay_configs:
            return decay_configs[memory_type]
        logger.debug(f"No decay config for type '{memory_type}', using decay_journal")
        return self.decay_journal


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    git_indexing: GitIndexingConfig = field(default_factory=GitIndexingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    parsers: dict[str, str] = field(
        default_factory=lambda: {
            "**/*.md": "MarkdownParser",
            "**/*.markdown": "MarkdownParser",
            "**/*.txt": "PlainTextParser",
        }
    )
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    document_chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    memory_chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    projects: list[ProjectConfig] = field(default_factory=list)


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


def _load_memory_config(data: dict[str, Any]):
    kwargs: dict[str, Any] = {}

    simple_fields = {
        "enabled",
        "storage_strategy",
        "score_threshold",
        "recency_boost_days",
        "recency_boost_factor",
    }
    decay_fields = {
        "decay_journal",
        "decay_plan",
        "decay_fact",
        "decay_observation",
        "decay_reflection",
    }

    for key in simple_fields:
        if key in data:
            kwargs[key] = data[key]

    for key in decay_fields:
        if key in data and isinstance(data[key], dict):
            kwargs[key] = _load_dataclass_from_dict(MemoryDecayConfig, data[key])

    return MemoryConfig(**kwargs)


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

    server = _load_dataclass_from_dict(ServerConfig, config_data.get("server", {}))

    indexing = _load_dataclass_from_dict(
        IndexingConfig,
        config_data.get("indexing", {}),
        path_fields={"documents_path", "index_path"},
    )
    # Always expand paths (defaults may be relative)
    indexing.documents_path = _expand_path(indexing.documents_path)
    indexing.index_path = _expand_path(indexing.index_path)

    parsers = config_data.get(
        "parsers",
        {
            "**/*.md": "MarkdownParser",
            "**/*.markdown": "MarkdownParser",
            "**/*.txt": "PlainTextParser",
        },
    )

    search = _load_dataclass_from_dict(SearchConfig, config_data.get("search", {}))
    llm = _load_dataclass_from_dict(LLMConfig, config_data.get("llm", {}))
    git_indexing = _load_dataclass_from_dict(
        GitIndexingConfig, config_data.get("git_indexing", {})
    )
    memory = _load_memory_config(config_data.get("memory", {}))

    # Backward compatibility: if [chunking] exists, use it for both document and memory
    # Otherwise, load separate configs
    if "chunking" in config_data:
        # Legacy config: single [chunking] section
        legacy_chunking = _load_dataclass_from_dict(
            ChunkingConfig, config_data["chunking"]
        )
        document_chunking = legacy_chunking
        memory_chunking = legacy_chunking
        logger.info("Using legacy [chunking] config for both documents and memories")
    else:
        # New config: separate sections
        document_chunking = _load_dataclass_from_dict(
            ChunkingConfig, config_data.get("chunking_documents", {})
        )
        memory_chunking = _load_dataclass_from_dict(
            ChunkingConfig, config_data.get("chunking_memories", {})
        )

    projects_data = config_data.get("projects", [])
    projects = []
    if projects_data:
        for proj_data in projects_data:
            try:
                projects.append(
                    ProjectConfig(name=proj_data["name"], path=proj_data["path"])
                )
            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Skipping invalid project config: {e}. Project data: {proj_data}"
                )

    _validate_projects(projects)

    return Config(
        server=server,
        indexing=indexing,
        git_indexing=git_indexing,
        memory=memory,
        parsers=parsers,
        search=search,
        llm=llm,
        document_chunking=document_chunking,
        memory_chunking=memory_chunking,
        projects=projects,
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
    global_config_path = (
        Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    )

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
            global_config_path = (
                Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
            )
            if global_config_path.exists():
                with open(global_config_path, "rb") as f:
                    config_data = tomllib.load(f)
                projects_data = config_data.get("projects", [])
                projects = []
                for proj_data in projects_data:
                    try:
                        projects.append(
                            ProjectConfig(
                                name=proj_data["name"], path=proj_data["path"]
                            )
                        )
                    except (KeyError, ValueError):
                        continue

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
            logger.info(f"Using arbitrary path from --project flag: {project_path}")

            existing_names = [p.name for p in (projects or [])]
            project_name = _generate_unique_project_name(
                project_path.name, existing_names
            )

            try:
                persist_project_to_config(project_name, str(project_path))
            except Exception as e:
                logger.warning(f"Failed to persist project to config: {e}")

            return project_name

        logger.warning(
            f"Project override '{project_override}' not found in registry and is not a valid path"
        )
        return None

    if cwd is None:
        cwd = Path.cwd()

    if projects is None:
        global_config_path = (
            Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
        )
        if not global_config_path.exists():
            projects = []
        else:
            with open(global_config_path, "rb") as f:
                config_data = tomllib.load(f)

            projects_data = config_data.get("projects", [])
            projects = []
            for proj_data in projects_data:
                try:
                    projects.append(
                        ProjectConfig(name=proj_data["name"], path=proj_data["path"])
                    )
                except (KeyError, ValueError):
                    continue

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

    if cwd_resolved.exists():
        logger.info(f"Auto-registering CWD as new project: {cwd_resolved}")

        existing_names = [p.name for p in projects]
        project_name = _generate_unique_project_name(cwd_resolved.name, existing_names)

        try:
            persist_project_to_config(project_name, str(cwd_resolved))
            logger.info(
                f"Successfully persisted CWD project '{project_name}': {cwd_resolved}"
            )
            return project_name
        except Exception as e:
            logger.warning(f"Failed to persist CWD project to config: {e}")
            return None

    return None


def resolve_index_path(config: Config, detected_project: str | None = None):
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

    if detected_project:
        safe_project_name = detected_project.replace("/", "_").replace("\\", "_")
        index_path = base_dir / "mcp-markdown-ragdocs" / safe_project_name
        logger.info(
            f"Using global data directory for project '{detected_project}': {index_path}"
        )
        return index_path

    cwd = Path.cwd()
    cwd_name = cwd.name
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "-", cwd_name)
    sanitized_name = re.sub(r"-+", "-", sanitized_name).strip("-")

    if not sanitized_name:
        sanitized_name = "default"

    fallback_name = f"local-{sanitized_name}"
    index_path = base_dir / "mcp-markdown-ragdocs" / fallback_name
    logger.info(
        f"No project detected, using global data directory with fallback: {index_path}"
    )
    return index_path


def resolve_documents_path(
    config: Config,
    detected_project: str | None = None,
    projects: list[ProjectConfig] | None = None,
) -> str:
    # If project detected, use the project's path (ignore config.indexing.documents_path)
    if detected_project and projects:
        for project in projects:
            if project.name == detected_project:
                project_path = Path(project.path)
                logger.info(
                    f"Using project path as documents root for '{detected_project}': {project_path}"
                )
                return str(project_path)

    # No project: use documents_path from config
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


def resolve_memory_path(
    config: Config,
    detected_project: str | None = None,
    projects: list[ProjectConfig] | None = None,
) -> Path:
    strategy = config.memory.storage_strategy

    if strategy == "project":
        if detected_project and projects:
            for project in projects:
                if project.name == detected_project:
                    memory_path = Path(project.path) / ".memories"
                    logger.info(
                        f"Using project memory path for '{detected_project}': {memory_path}"
                    )
                    return memory_path

        cwd = Path.cwd()
        memory_path = cwd / ".memories"
        logger.info(f"Using CWD memory path: {memory_path}")
        return memory_path

    data_home = os.getenv("XDG_DATA_HOME")
    if data_home:
        base_dir = Path(data_home)
    else:
        base_dir = Path.home() / ".local" / "share"

    if detected_project:
        safe_project_name = detected_project.replace("/", "_").replace("\\", "_")
        memory_path = base_dir / "mcp-markdown-ragdocs" / safe_project_name / "memories"
        logger.info(
            f"Using user memory path for project '{detected_project}': {memory_path}"
        )
        return memory_path

    cwd = Path.cwd()
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "-", cwd.name)
    sanitized_name = re.sub(r"-+", "-", sanitized_name).strip("-") or "default"

    memory_path = (
        base_dir / "mcp-markdown-ragdocs" / f"local-{sanitized_name}" / "memories"
    )
    logger.info(f"Using fallback user memory path: {memory_path}")
    return memory_path
