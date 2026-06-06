#!/usr/bin/env python3
"""
MCP server for retrieval-only codebase search via DirToRAG.

Runs as a stdio process. Claude Code calls search_codebase / get_index_stats,
and the server returns relevant chunks without any LLM generation.

Optionally watches a directory for file changes and auto-reindexes
(set DIRTORAG_WATCH_DIR env var to the indexed directory).

For debugging:
    python mcp_search.py           # stdio mode
    mcp dev mcp_search.py          # MCP Inspector (web UI)
"""

import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Any

# Disable system proxies (as in server.py and cli.py)
for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from agents.pg_agent import PostgresSearchAgent, set_search_table, get_search_table, get_db_connection_string
from models_loader import load_app_config
from chunker import chunk_text
from embedder import get_embeddings
from cli import (
    ALLOWED_EXT,
    INDEXED_LOG_FILENAME,
    delete_chunks_for_file,
    insert_to_postgres,
    append_indexed_file,
    load_indexed_files,
    get_postgres_connection_string,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_search")

DEFAULT_TABLE = os.environ.get("DIRTORAG_TABLE", "documents")
WATCH_DIR = os.environ.get("DIRTORAG_WATCH_DIR", "")

mcp = FastMCP(name="DirToRAG Search")

_search_agent: PostgresSearchAgent | None = None

# Debounce state for file watcher
_debounce_timers: dict[str, threading.Timer] = {}
_debounce_lock = threading.Lock()
DEBOUNCE_SECONDS = 2.0


def _get_agent() -> PostgresSearchAgent:
    """Lazily create PostgresSearchAgent with config from config.yaml."""
    global _search_agent
    if _search_agent is None:
        table = get_search_table()
        if table == "documents":
            set_search_table(DEFAULT_TABLE)
            table = DEFAULT_TABLE
        logger.info("Initializing PostgresSearchAgent for table: %s", table)
        _search_agent = PostgresSearchAgent(
            config={
                "table_name": table,
                "limit": 50,
            }
        )
    return _search_agent


def _is_indexable(file_path: str) -> bool:
    """Check if a file should be indexed based on its extension."""
    return Path(file_path).suffix.lower() in ALLOWED_EXT


def _reindex_single_file(abs_path: str, table_name: str, watch_dir: str) -> None:
    """Reindex a single file: chunk, embed, delete old chunks, insert new."""
    rel_path = str(Path(abs_path).relative_to(watch_dir))

    try:
        text = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", abs_path, e)
        return

    chunks = chunk_text(text)
    conn_str = get_postgres_connection_string()
    mtime = Path(abs_path).stat().st_mtime

    # Delete old chunks
    delete_chunks_for_file(conn_str, table_name, rel_path)

    if not chunks:
        log_path = Path(watch_dir) / INDEXED_LOG_FILENAME
        append_indexed_file(log_path, rel_path, mtime)
        logger.info("Reindexed (empty chunks): %s", rel_path)
        return

    # Get embeddings
    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", rel_path, e)
        return

    if len(embeddings) < len(chunks):
        chunks = chunks[:len(embeddings)]

    if not embeddings:
        log_path = Path(watch_dir) / INDEXED_LOG_FILENAME
        append_indexed_file(log_path, rel_path, mtime)
        return

    records = [
        {"content": chunk, "embedding": emb, "metadata": {"path": rel_path}}
        for chunk, emb in zip(chunks, embeddings)
    ]

    try:
        insert_to_postgres(records, conn_str, table_name)
    except Exception as e:
        logger.error("Insert failed for %s: %s", rel_path, e)
        return

    log_path = Path(watch_dir) / INDEXED_LOG_FILENAME
    append_indexed_file(log_path, rel_path, mtime)
    logger.info("Reindexed: %s (%d chunks)", rel_path, len(chunks))


def _delete_single_file(rel_path: str, table_name: str, watch_dir: str) -> None:
    """Delete all chunks for a removed file and update the log."""
    conn_str = get_postgres_connection_string()
    n = delete_chunks_for_file(conn_str, table_name, rel_path)
    logger.info("Deleted %d chunks for removed file: %s", n, rel_path)

    # Remove from log
    log_path = Path(watch_dir) / INDEXED_LOG_FILENAME
    indexed = load_indexed_files(log_path)
    indexed.pop(rel_path, None)
    from cli import _rewrite_indexed_log
    _rewrite_indexed_log(log_path, indexed)


def _schedule_reindex(abs_path: str, table_name: str, watch_dir: str) -> None:
    """Debounce file change events — schedule reindex after DEBOUNCE_SECONDS."""
    with _debounce_lock:
        existing = _debounce_timers.pop(abs_path, None)
        if existing is not None:
            existing.cancel()

        timer = threading.Timer(
            DEBOUNCE_SECONDS,
            _debounced_reindex,
            args=(abs_path, table_name, watch_dir),
        )
        timer.daemon = True
        _debounce_timers[abs_path] = timer
        timer.start()


def _debounced_reindex(abs_path: str, table_name: str, watch_dir: str) -> None:
    """Called by timer after debounce period. Checks file still exists."""
    with _debounce_lock:
        _debounce_timers.pop(abs_path, None)

    if not os.path.isfile(abs_path):
        # File was deleted during debounce window
        _delete_single_file(
            str(Path(abs_path).relative_to(watch_dir)), table_name, watch_dir
        )
        return

    _reindex_single_file(abs_path, table_name, watch_dir)


def _start_watcher(table_name: str, watch_dir: str) -> None:
    """Start a background file system watcher for the indexed directory."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class IndexEventHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                logger.debug("File created: %s", event.src_path)
                _schedule_reindex(event.src_path, table_name, watch_dir)

        def on_modified(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                logger.debug("File modified: %s", event.src_path)
                _schedule_reindex(event.src_path, table_name, watch_dir)

        def on_deleted(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                rel_path = str(Path(event.src_path).relative_to(watch_dir))
                logger.debug("File deleted: %s", rel_path)
                # Deletion doesn't need debounce — file is gone
                _delete_single_file(rel_path, table_name, watch_dir)

        def on_moved(self, event):
            if not event.is_directory:
                if _is_indexable(event.src_path):
                    logger.debug("File moved in: %s", event.src_path)
                    _schedule_reindex(event.src_path, table_name, watch_dir)
                if _is_indexable(event.dest_path):
                    # If moved from an indexable file, delete old entry
                    rel_old = str(Path(event.src_path).relative_to(watch_dir))
                    conn_str = get_postgres_connection_string()
                    delete_chunks_for_file(conn_str, table_name, rel_old)
                    log_path = Path(watch_dir) / INDEXED_LOG_FILENAME
                    indexed = load_indexed_files(log_path)
                    indexed.pop(rel_old, None)
                    from cli import _rewrite_indexed_log
                    _rewrite_indexed_log(log_path, indexed)
                    logger.debug("Deleted old entry for moved file: %s", rel_old)

    observer = Observer()
    observer.schedule(IndexEventHandler(), watch_dir, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info("File watcher started for: %s (table: %s)", watch_dir, table_name)


@mcp.tool()
def search_codebase(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the indexed codebase using hybrid search (full-text + semantic vector search).

    Use this tool to find relevant code snippets, documentation, and configuration
    files related to a natural language query. The search combines keyword matching
    (PostgreSQL full-text search) with semantic similarity (pgvector cosine distance)
    using Reciprocal Rank Fusion.

    Args:
        query: Natural language search query (e.g., "How does authentication work?")
        limit: Maximum number of results to return (default: 10, max: 50)

    Returns:
        List of dictionaries, each containing:
        - content: The matched text chunk
        - file_path: Relative path to the source file
        - rank: Result rank (1 = most relevant)
        - score: Relevance score (higher = more relevant)
    """
    agent = _get_agent()
    results = agent.search_raw(query, limit=min(limit, 50))

    if not results:
        return []

    output: list[dict[str, Any]] = []
    for i, row in enumerate(results, 1):
        metadata = row.get("metadata", {})
        content = row.get("content", "")
        path = metadata.get("path", "unknown") if metadata else "unknown"
        score = row.get("score", None)
        output.append({
            "content": content,
            "file_path": path,
            "rank": i,
            "score": score,
        })

    return output


@mcp.tool()
def get_index_stats() -> dict[str, Any]:
    """
    Get statistics about the indexed codebase.

    Returns:
        Dictionary with:
        - table_name: Name of the search table
        - total_chunks: Total number of indexed chunks
        - total_files: Number of unique files indexed
        - embedding_model: Embedding model used
    """
    import psycopg

    table_name = get_search_table()
    if table_name == "documents":
        table_name = DEFAULT_TABLE

    conn_str = get_db_connection_string()
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_chunks = cur.fetchone()[0]

                cur.execute(
                    f"SELECT COUNT(DISTINCT metadata->>'path') FROM {table_name}"
                )
                total_files = cur.fetchone()[0]

        cfg = load_app_config()
        emb_model = cfg.get("embedding", {}).get("model", "unknown")

        return {
            "table_name": table_name,
            "total_chunks": total_chunks,
            "total_files": total_files,
            "embedding_model": emb_model,
        }
    except Exception as e:
        logger.exception("Stats failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
def get_debug_info() -> dict[str, Any]:
    """
    Get detailed debug information about the indexed codebase and database.

    Returns:
        Dictionary with:
        - table_name: Search table name
        - total_chunks, total_files, embedding_model
        - db_size_mb: Total PostgreSQL database size on disk
        - table_size_mb: Search table size (data + indexes)
        - index_size_mb: Size of all indexes on the search table
        - chunks_per_extension: Breakdown of chunk count by file extension
        - avg_chunks_per_file: Average number of chunks per file
        - max_chunks_per_file: Path and count of the file with most chunks
        - embedding_dim: Actual embedding vector dimension from the DB
    """
    import psycopg

    table_name = get_search_table()
    if table_name == "documents":
        table_name = DEFAULT_TABLE

    conn_str = get_db_connection_string()
    result: dict[str, Any] = {"table_name": table_name}

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Total chunks
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                result["total_chunks"] = cur.fetchone()[0]

                # Total files
                cur.execute(
                    f"SELECT COUNT(DISTINCT metadata->>'path') FROM {table_name}"
                )
                result["total_files"] = cur.fetchone()[0]

                # DB size
                cur.execute(
                    "SELECT pg_database_size(current_database()) / (1024*1024.0)"
                )
                result["db_size_mb"] = round(cur.fetchone()[0], 2)

                # Table size (data)
                cur.execute(
                    f"SELECT pg_table_size('{table_name}') / (1024*1024.0)"
                )
                result["table_size_mb"] = round(cur.fetchone()[0], 2)

                # Index size
                cur.execute(
                    f"SELECT pg_indexes_size('{table_name}') / (1024*1024.0)"
                )
                result["index_size_mb"] = round(cur.fetchone()[0], 2)

                # Chunks per extension
                cur.execute(f"""
                    SELECT
                        split_part(metadata->>'path', '.', -1) AS ext,
                        COUNT(*) AS cnt
                    FROM {table_name}
                    GROUP BY ext
                    ORDER BY cnt DESC
                """)
                result["chunks_per_extension"] = {
                    f".{row[0]}": row[1] for row in cur.fetchall()
                }

                # Average chunks per file
                cur.execute(f"""
                    SELECT AVG(cnt)::numeric(10,2), MAX(cnt), MAX(path)
                    FROM (
                        SELECT metadata->>'path' AS path, COUNT(*) AS cnt
                        FROM {table_name}
                        GROUP BY metadata->>'path'
                    ) sub
                """)
                row = cur.fetchone()
                result["avg_chunks_per_file"] = float(row[0]) if row[0] else 0
                result["max_chunks_per_file"] = {
                    "path": row[2] or "N/A",
                    "chunks": row[1] or 0,
                }

                # Embedding dimension (from actual vectors in the table)
                cur.execute(
                    f"SELECT extensions.vector_dims(embedding) FROM {table_name} "
                    "WHERE embedding IS NOT NULL LIMIT 1"
                )
                row = cur.fetchone()
                result["embedding_dim"] = row[0] if row else "unknown"

        cfg = load_app_config()
        result["embedding_model"] = cfg.get("embedding", {}).get("model", "unknown")

        return result

    except Exception as e:
        logger.exception("Debug info failed: %s", e)
        return {"error": str(e)}


def main():
    """Entry point for MCP server."""
    if len(sys.argv) > 1:
        table = sys.argv[1]
        set_search_table(table)
        logger.info("Table set from command line: %s", table)
    else:
        set_search_table(DEFAULT_TABLE)
        logger.info("Using default table: %s", DEFAULT_TABLE)

    # Determine watch directory
    watch_dir = WATCH_DIR
    if len(sys.argv) > 2:
        watch_dir = sys.argv[2]
        logger.info("Watch directory from args: %s", watch_dir)

    if watch_dir:
        if not os.path.isdir(watch_dir):
            logger.error("Watch directory not found: %s", watch_dir)
        else:
            table = get_search_table()
            if table == "documents":
                table = DEFAULT_TABLE
            _start_watcher(table, watch_dir)

    logger.info("Starting DirToRAG MCP server (stdio mode)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
