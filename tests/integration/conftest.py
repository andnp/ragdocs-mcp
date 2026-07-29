"""Integration test fixtures with xdist worker isolation.

pytest.mark.xdist_group (used elsewhere in this suite for "serial" tests) only
has an effect under `--dist loadgroup` -- it is a no-op under this repo's
default `--dist worksteal` (see tests/integration/test_pgvector_index.py's
docstring). So pgvector integration tests cannot rely on grouping to avoid
running concurrently with other workers against the shared Postgres
container; they need real per-worker isolation instead.
"""


def pg_worker_schema(config) -> str:
    """A stable, valid Postgres schema name unique to this xdist worker.

    Returns e.g. "pgtest_gw0", or "pgtest_master" when not running under xdist.
    """
    worker_id = "master"
    if hasattr(config, "workerinput"):
        worker_id = config.workerinput.get("workerid", "master")
    return f"pgtest_{worker_id}"


def pg_dsn_for_schema(base_dsn: str, schema: str) -> str:
    """Return base_dsn with a libpq `options` param pinning search_path to schema.

    Table DDL/DML in this suite is never schema-qualified, so pinning
    search_path is sufficient to give each xdist worker its own private set
    of `records`/`vector_tables`/`graph_edges`/`cache_store`/`index_epoch`
    tables and per-model vector tables, without touching production code.
    """
    from urllib.parse import quote

    options = quote(f"-c search_path={schema},public", safe="")
    separator = "&" if "?" in base_dsn else "?"
    return f"{base_dsn}{separator}options={options}"
