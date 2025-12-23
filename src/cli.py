import glob
import json
import logging
import sys
from pathlib import Path

import click
import uvicorn

from src.config import load_config
from src.indexing.manager import IndexManager
from src.indexing.manifest import IndexManifest, save_manifest
from src.indices.graph import GraphStore
from src.indices.keyword import KeywordIndex
from src.indices.vector import VectorIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--host", default=None, help="Override host from config")
@click.option("--port", default=None, type=int, help="Override port from config")
def run(host: str | None, port: int | None):
    try:
        config = load_config()

        server_host = host or config.server.host
        server_port = port or config.server.port

        logger.info(f"Starting server on {server_host}:{server_port}")
        uvicorn.run(
            "src.server:create_app",
            host=server_host,
            port=server_port,
            factory=True,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


@cli.command("rebuild-index")
def rebuild_index_cmd():
    try:
        logger.info("Loading configuration")
        config = load_config()

        logger.info("Initializing indices")
        vector = VectorIndex()
        keyword = KeywordIndex()
        graph = GraphStore()

        manager = IndexManager(config, vector, keyword, graph)

        docs_path = Path(config.indexing.documents_path)
        index_path = Path(config.indexing.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Rebuilding index for documents in: {docs_path}")

        pattern = str(docs_path / "**" / "*.md")
        file_count = 0
        for file_path in glob.glob(pattern, recursive=config.indexing.recursive):
            logger.info(f"Indexing: {file_path}")
            manager.index_document(file_path)
            file_count += 1

        logger.info(f"Persisting indices ({file_count} documents processed)")
        manager.persist()

        current_manifest = IndexManifest(
            spec_version="1.0.0",
            embedding_model=config.llm.embedding_model,
            parsers=config.parsers,
        )
        save_manifest(index_path, current_manifest)

        logger.info("Index rebuild complete")
        click.echo(f"Successfully rebuilt index: {file_count} documents indexed")

    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("check-config")
def check_config_cmd():
    try:
        logger.info("Loading configuration")
        config = load_config()

        config_dict = {
            "server": {
                "host": config.server.host,
                "port": config.server.port,
            },
            "indexing": {
                "documents_path": config.indexing.documents_path,
                "index_path": config.indexing.index_path,
                "recursive": config.indexing.recursive,
            },
            "search": {
                "semantic_weight": config.search.semantic_weight,
                "keyword_weight": config.search.keyword_weight,
                "recency_bias": config.search.recency_bias,
                "rrf_k_constant": config.search.rrf_k_constant,
            },
            "llm": {
                "embedding_model": config.llm.embedding_model,
                "llm_provider": config.llm.llm_provider,
            },
            "parsers": config.parsers,
        }

        click.echo("Configuration loaded successfully:")
        click.echo(json.dumps(config_dict, indent=2))

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()
