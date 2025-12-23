from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import tomllib


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class IndexingConfig:
    documents_path: str = "."
    index_path: str = ".index_data/"
    recursive: bool = True


@dataclass
class SearchConfig:
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    recency_bias: float = 0.5
    rrf_k_constant: int = 60


@dataclass
class LLMConfig:
    embedding_model: str = "local"
    llm_provider: Optional[str] = None


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


def _expand_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


def load_config():
    config_locations = [
        Path("./config.toml"),
        Path.home() / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    ]

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
        recursive=indexing_data.get("recursive", True)
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
        rrf_k_constant=search_data.get("rrf_k_constant", 60)
    )

    llm_data = config_data.get("llm", {})
    llm = LLMConfig(
        embedding_model=llm_data.get("embedding_model", "local"),
        llm_provider=llm_data.get("llm_provider")
    )

    return Config(
        server=server,
        indexing=indexing,
        parsers=parsers,
        search=search,
        llm=llm
    )
