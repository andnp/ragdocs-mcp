from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.name):
            raise ValueError(
                f"Invalid project name '{self.name}': "
                "must contain only alphanumeric characters, hyphens, and underscores"
            )

        path_obj = Path(self.path).expanduser()
        if not path_obj.is_absolute():
            raise ValueError(
                f"Project path '{self.path}' must be absolute"
            )

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
    exclude: list[str] = field(default_factory=lambda: [
        "**/.venv/**",
        "**/venv/**",
        "**/build/**",
        "**/dist/**",
        "**/.git/**",
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/.pytest_cache/**"
    ])
    exclude_hidden_dirs: bool = True
    reconciliation_interval_seconds: int = 3600  # 1 hour, 0 to disable


@dataclass
class SearchConfig:
    semantic_weight: float = 1.0
    keyword_weight: float = 0.8
    recency_bias: float = 0.5
    rrf_k_constant: int = 60
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 2
    dedup_enabled: bool = False
    dedup_similarity_threshold: float = 0.80
    ngram_dedup_enabled: bool = True
    ngram_dedup_threshold: float = 0.7
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10
    adaptive_weights_enabled: bool = False
    code_search_enabled: bool = False
    code_search_weight: float = 1.0


@dataclass
class LLMConfig:
    embedding_model: str = "local"
    llm_provider: str | None = None


@dataclass
class ChunkingConfig:
    strategy: str = "header_based"
    min_chunk_chars: int = 200
    max_chunk_chars: int = 2000
    overlap_chars: int = 100
    include_parent_headers: bool = True
    parent_retrieval_enabled: bool = False
    parent_chunk_min_chars: int = 1500
    parent_chunk_max_chars: int = 2000


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    parsers: dict[str, str] = field(default_factory=lambda: {
        "**/*.md": "MarkdownParser",
        "**/*.markdown": "MarkdownParser"
    })
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    projects: list[ProjectConfig] = field(default_factory=list)


def _expand_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


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

    config_data = {}
    for config_path in config_locations:
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
            break

    server_data = config_data.get("server", {})
    server = ServerConfig(
        host=server_data.get("host", "127.0.0.1"),
        port=server_data.get("port", 8000)
    )

    indexing_data = config_data.get("indexing", {})
    indexing = IndexingConfig(
        documents_path=_expand_path(indexing_data.get("documents_path", ".")),
        index_path=_expand_path(indexing_data.get("index_path", ".index_data/")),
        recursive=indexing_data.get("recursive", True),
        include=indexing_data.get("include", ["**/*"]),
        exclude=indexing_data.get("exclude", [
            "**/.venv/**",
            "**/venv/**",
            "**/build/**",
            "**/dist/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**"
        ]),
        exclude_hidden_dirs=indexing_data.get("exclude_hidden_dirs", True),
        reconciliation_interval_seconds=indexing_data.get("reconciliation_interval_seconds", 3600)
    )

    parsers = config_data.get("parsers", {
        "**/*.md": "MarkdownParser",
        "**/*.markdown": "MarkdownParser"
    })

    search_data = config_data.get("search", {})
    search = SearchConfig(
        semantic_weight=search_data.get("semantic_weight", 1.0),
        keyword_weight=search_data.get("keyword_weight", 1.0),
        recency_bias=search_data.get("recency_bias", 0.5),
        rrf_k_constant=search_data.get("rrf_k_constant", 60),
        min_confidence=search_data.get("min_confidence", 0.0),
        max_chunks_per_doc=search_data.get("max_chunks_per_doc", 0),
        dedup_enabled=search_data.get("dedup_enabled", False),
        dedup_similarity_threshold=search_data.get("dedup_similarity_threshold", 0.85),
        ngram_dedup_enabled=search_data.get("ngram_dedup_enabled", True),
        ngram_dedup_threshold=search_data.get("ngram_dedup_threshold", 0.7),
        mmr_enabled=search_data.get("mmr_enabled", False),
        mmr_lambda=search_data.get("mmr_lambda", 0.7),
        rerank_enabled=search_data.get("rerank_enabled", False),
        rerank_model=search_data.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_top_n=search_data.get("rerank_top_n", 10),
        adaptive_weights_enabled=search_data.get("adaptive_weights_enabled", False),
        code_search_enabled=search_data.get("code_search_enabled", False),
        code_search_weight=search_data.get("code_search_weight", 1.0),
    )

    llm_data = config_data.get("llm", {})
    llm = LLMConfig(
        embedding_model=llm_data.get("embedding_model", "local"),
        llm_provider=llm_data.get("llm_provider")
    )

    chunking_data = config_data.get("chunking", {})
    chunking = ChunkingConfig(
        strategy=chunking_data.get("strategy", "header_based"),
        min_chunk_chars=chunking_data.get("min_chunk_chars", 200),
        max_chunk_chars=chunking_data.get("max_chunk_chars", 1500),
        overlap_chars=chunking_data.get("overlap_chars", 100),
        include_parent_headers=chunking_data.get("include_parent_headers", True),
        parent_retrieval_enabled=chunking_data.get("parent_retrieval_enabled", False),
        parent_chunk_min_chars=chunking_data.get("parent_chunk_min_chars", 1500),
        parent_chunk_max_chars=chunking_data.get("parent_chunk_max_chars", 2000),
    )

    projects_data = config_data.get("projects", [])
    projects = []
    if projects_data:
        for proj_data in projects_data:
            try:
                projects.append(ProjectConfig(
                    name=proj_data["name"],
                    path=proj_data["path"]
                ))
            except (KeyError, ValueError) as e:
                # REVIEW [MED] Logging: Warning log doesn't identify which project entry
                # failed (index in array or partial data). Include proj_data in log.
                logger.warning(f"Skipping invalid project config: {e}")

    _validate_projects(projects)

    return Config(
        server=server,
        indexing=indexing,
        parsers=parsers,
        search=search,
        llm=llm,
        chunking=chunking,
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
    name = re.sub(r'[^a-zA-Z0-9_-]', '-', base_name)
    name = re.sub(r'-+', '-', name).strip('-')

    if not name or not re.match(r'^[a-zA-Z0-9_-]+$', name):
        name = "project"

    if name not in existing_names:
        return name

    counter = 2
    while f"{name}-{counter}" in existing_names:
        counter += 1

    return f"{name}-{counter}"


def persist_project_to_config(project_name: str, project_path: str):
    global_config_path = Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"

    global_config_path.parent.mkdir(parents=True, exist_ok=True)

    # REVIEW [LOW] Type Safety: Any used for tomlkit document. tomlkit.TOMLDocument
    # could be used but lacks good type stubs. Acceptable for dynamic TOML manipulation.
    doc: Any
    if global_config_path.exists():
        with open(global_config_path, "r") as f:
            doc = tomlkit.load(f)
    else:
        doc = tomlkit.document()

    if "projects" not in doc:
        doc["projects"] = tomlkit.aot()

    projects_array: Any = doc["projects"]
    for proj in projects_array:
        if proj.get("name") == project_name:
            logger.debug(f"Project '{project_name}' already exists in config")
            return
        if proj.get("path") == project_path:
            logger.debug(f"Project path '{project_path}' already registered")
            return

    new_project: Any = tomlkit.table()
    new_project["name"] = project_name
    new_project["path"] = project_path
    projects_array.append(new_project)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=global_config_path.parent,
        delete=False,
        suffix=".tmp"
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tomlkit.dump(doc, tmp_file)

    tmp_path.replace(global_config_path)
    logger.info(f"Persisted project '{project_name}' to config: {project_path}")


def detect_project(cwd: Path | None = None, projects: list[ProjectConfig] | None = None, project_override: str | None = None):
    if project_override:
        if projects is None:
            global_config_path = Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
            if global_config_path.exists():
                with open(global_config_path, "rb") as f:
                    config_data = tomllib.load(f)
                projects_data = config_data.get("projects", [])
                projects = []
                for proj_data in projects_data:
                    try:
                        projects.append(ProjectConfig(
                            name=proj_data["name"],
                            path=proj_data["path"]
                        ))
                    except (KeyError, ValueError):
                        continue

        if projects:
            for project in projects:
                if project.name == project_override:
                    logger.info(f"Using project from --project flag: {project.name} (path: {project.path})")
                    return project.name

            project_path = Path(project_override).expanduser().resolve()
            for project in projects:
                if Path(project.path).resolve() == project_path:
                    logger.info(f"Using project from --project flag (matched by path): {project.name}")
                    return project.name

            # Check if project_override path is a subdirectory of a known project (deepest-match-wins)
            projects_sorted = sorted(
                projects,
                key=lambda p: len(Path(p.path).parts),
                reverse=True
            )
            for project in projects_sorted:
                project_path_resolved = Path(project.path).resolve()
                try:
                    project_path.relative_to(project_path_resolved)
                    logger.info(f"Using project from --project flag (subdirectory of '{project.name}'): {project.path}")
                    return project.name
                except ValueError:
                    continue

        project_path = Path(project_override).expanduser().resolve()
        if project_path.exists():
            logger.info(f"Using arbitrary path from --project flag: {project_path}")

            existing_names = [p.name for p in (projects or [])]
            project_name = _generate_unique_project_name(project_path.name, existing_names)

            try:
                persist_project_to_config(project_name, str(project_path))
            except Exception as e:
                logger.warning(f"Failed to persist project to config: {e}")

            return project_name

        logger.warning(f"Project override '{project_override}' not found in registry and is not a valid path")
        return None

    if cwd is None:
        cwd = Path.cwd()

    if projects is None:
        global_config_path = Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
        if not global_config_path.exists():
            return None

        with open(global_config_path, "rb") as f:
            config_data = tomllib.load(f)

        projects_data = config_data.get("projects", [])
        projects = []
        for proj_data in projects_data:
            try:
                projects.append(ProjectConfig(
                    name=proj_data["name"],
                    path=proj_data["path"]
                ))
            except (KeyError, ValueError):
                continue

    if not projects:
        return None

    cwd_resolved = cwd.resolve()

    projects_sorted = sorted(
        projects,
        key=lambda p: len(Path(p.path).parts),
        reverse=True
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
        logger.info(f"Using global data directory for project '{detected_project}': {index_path}")
        return index_path

    cwd = Path.cwd()
    cwd_name = cwd.name
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '-', cwd_name)
    sanitized_name = re.sub(r'-+', '-', sanitized_name).strip('-')

    if not sanitized_name:
        sanitized_name = "default"

    fallback_name = f"local-{sanitized_name}"
    index_path = base_dir / "mcp-markdown-ragdocs" / fallback_name
    logger.info(f"No project detected, using global data directory with fallback: {index_path}")
    return index_path


def resolve_documents_path(config: Config, detected_project: str | None = None, projects: list[ProjectConfig] | None = None) -> str:
    # If project detected, use the project's path (ignore config.indexing.documents_path)
    if detected_project and projects:
        for project in projects:
            if project.name == detected_project:
                project_path = Path(project.path)
                logger.info(f"Using project path as documents root for '{detected_project}': {project_path}")
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
