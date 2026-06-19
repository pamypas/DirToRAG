#!/usr/bin/env python3
"""
MCP server for retrieval-only codebase search via DirToRAG.

Runs as a stdio process. Claude Code calls search_codebase / get_index_stats,
and the server returns relevant chunks without any LLM generation.

Auto-detects the project from CWD (via project_resolver).
Starts a web dashboard alongside the MCP stdio transport.
"""

import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Any

# Force unbuffered stdout — required for MCP stdio transport
sys.stdout.reconfigure(line_buffering=True)

# Disable system proxies
for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from agents.pg_agent import PostgresSearchAgent, set_search_table, get_search_table
from models_loader import load_app_config
from chunker import chunk_text
from embedder import get_embeddings
from cli import (
    ALLOWED_EXT,
    SKIP_DIRS,
    delete_chunks_for_file,
    insert_to_postgres,
    get_postgres_connection_string,
    init_database,
    index_directory,
)
from state_db import (
    CHUNKS_TABLE,
    FILES_TABLE,
    ensure_registry,
    ensure_project_db,
    get_or_create_project,
    get_project_by_table,
    upsert_indexed_file,
    delete_indexed_file,
    set_project_paused,
    update_project_indexed_at,
    migrate_file_state,
    table_exists_in_project_db,
    _project_conn_str,
    scan_files_to_db,
)
from project_resolver import resolve_project

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_search")

DEFAULT_TABLE = os.environ.get("DIRTORAG_TABLE", "documents")

mcp = FastMCP(name="DirToRAG Search")

_search_agent: PostgresSearchAgent | None = None
_project_id: int | None = None
_db_name: str = ""
_indexing_paused = False
_dashboard_state = None

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
            config={"table_name": table, "limit": 50}
        )
    return _search_agent


def _is_indexable(file_path: str) -> bool:
    if Path(file_path).suffix.lower() not in ALLOWED_EXT:
        return False
    # Skip files in excluded directories (venv, node_modules, etc.)
    parts = Path(file_path).parts
    for part in parts:
        if part.startswith(".") or part in SKIP_DIRS:
            return False
    return True


def _reindex_single_file(abs_path: str, db_name: str, watch_dir: str) -> None:
    """Reindex a single file: chunk, embed, delete old chunks, insert new."""
    rel_path = str(Path(abs_path).relative_to(watch_dir))

    try:
        text = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", abs_path, e)
        return

    chunks = chunk_text(text)
    conn_str = _project_conn_str(db_name)
    mtime = Path(abs_path).stat().st_mtime
    file_size = Path(abs_path).stat().st_size

    delete_chunks_for_file(conn_str, rel_path)

    if not chunks:
        upsert_indexed_file(db_name, rel_path, mtime, chunk_count=0, size_bytes=file_size, status="empty")
        logger.info("Reindexed (empty chunks): %s", rel_path)
        return

    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", rel_path, e)
        upsert_indexed_file(db_name, rel_path, mtime, chunk_count=0, size_bytes=file_size, status="failed", error=str(e))
        return

    if len(embeddings) < len(chunks):
        chunks = chunks[:len(embeddings)]

    if not embeddings:
        upsert_indexed_file(db_name, rel_path, mtime, chunk_count=0, size_bytes=file_size, status="empty")
        return

    records = [
        {"content": chunk, "embedding": emb, "metadata": {"path": rel_path}}
        for chunk, emb in zip(chunks, embeddings)
    ]

    try:
        insert_to_postgres(records, conn_str)
    except Exception as e:
        logger.error("Insert failed for %s: %s", rel_path, e)
        upsert_indexed_file(db_name, rel_path, mtime, chunk_count=0, size_bytes=file_size, status="failed", error=str(e))
        return

    upsert_indexed_file(db_name, rel_path, mtime, chunk_count=len(chunks), size_bytes=file_size)
    logger.info("Reindexed: %s (%d chunks)", rel_path, len(chunks))


def _delete_single_file(rel_path: str, db_name: str) -> None:
    """Delete all chunks for a removed file and update the DB state."""
    conn_str = _project_conn_str(db_name)
    n = delete_chunks_for_file(conn_str, rel_path)
    logger.info("Deleted %d chunks for removed file: %s", n, rel_path)
    delete_indexed_file(db_name, rel_path)


def _schedule_reindex(abs_path: str, db_name: str, watch_dir: str) -> None:
    """Debounce file change events — schedule reindex after DEBOUNCE_SECONDS."""
    paused = _indexing_paused
    if _dashboard_state is not None:
        paused = _dashboard_state.indexing_paused

    if paused:
        logger.debug("Indexing paused, skipping reindex for: %s", abs_path)
        return

    with _debounce_lock:
        existing = _debounce_timers.pop(abs_path, None)
        if existing is not None:
            existing.cancel()

        timer = threading.Timer(
            DEBOUNCE_SECONDS,
            _debounced_reindex,
            args=(abs_path, db_name, watch_dir),
        )
        timer.daemon = True
        _debounce_timers[abs_path] = timer
        timer.start()


def _debounced_reindex(abs_path: str, db_name: str, watch_dir: str) -> None:
    with _debounce_lock:
        _debounce_timers.pop(abs_path, None)

    if not os.path.isfile(abs_path):
        rel_path = str(Path(abs_path).relative_to(watch_dir))
        _delete_single_file(rel_path, db_name)
        return

    _reindex_single_file(abs_path, db_name, watch_dir)


def _start_watcher(db_name: str, watch_dir: str) -> None:
    """Start a background file system watcher for the indexed directory."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class IndexEventHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                logger.debug("File created: %s", event.src_path)
                _schedule_reindex(event.src_path, db_name, watch_dir)

        def on_modified(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                logger.debug("File modified: %s", event.src_path)
                _schedule_reindex(event.src_path, db_name, watch_dir)

        def on_deleted(self, event):
            if not event.is_directory and _is_indexable(event.src_path):
                rel_path = str(Path(event.src_path).relative_to(watch_dir))
                logger.debug("File deleted: %s", rel_path)
                _delete_single_file(rel_path, db_name)

        def on_moved(self, event):
            if not event.is_directory:
                if _is_indexable(event.dest_path):
                    logger.debug("File moved in: %s", event.dest_path)
                    _schedule_reindex(event.dest_path, db_name, watch_dir)
                if _is_indexable(event.src_path):
                    rel_old = str(Path(event.src_path).relative_to(watch_dir))
                    conn_str = _project_conn_str(db_name)
                    delete_chunks_for_file(conn_str, rel_old)
                    delete_indexed_file(db_name, rel_old)
                    logger.debug("Deleted old entry for moved file: %s", rel_old)

    observer = Observer()
    observer.schedule(IndexEventHandler(), watch_dir, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info("File watcher started for: %s (db: %s)", watch_dir, db_name)


@mcp.tool()
def search_codebase(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search the indexed codebase using hybrid search (full-text + semantic vector search).

    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 10, max: 50)

    Returns:
        List of dicts with content, file_path, rank, score
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
        output.append({"content": content, "file_path": path, "rank": i, "score": score})

    if _dashboard_state is not None:
        _dashboard_state.search_history.append({
            "ts": time.time(), "query": query, "n_results": len(output),
        })

    return output


@mcp.tool()
def get_index_stats() -> dict[str, Any]:
    """Get statistics about the indexed codebase."""
    import psycopg

    table_name = get_search_table()
    if table_name == "documents":
        table_name = DEFAULT_TABLE

    conn_str = _project_conn_str(table_name)
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {CHUNKS_TABLE}")
                total_chunks = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE}")
                total_files = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE} WHERE status = 'pending'")
                pending_files = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE} WHERE status = 'indexed'")
                indexed_files = cur.fetchone()[0]

        cfg = load_app_config()
        emb_model = cfg.get("embedding", {}).get("model", "unknown")

        return {
            "table_name": table_name,
            "total_chunks": total_chunks,
            "total_files": total_files,
            "indexed_files": indexed_files,
            "pending_files": pending_files,
            "embedding_model": emb_model,
        }
    except Exception as e:
        logger.exception("Stats failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
def get_debug_info() -> dict[str, Any]:
    """Get detailed debug information about the indexed codebase and database."""
    import psycopg

    table_name = get_search_table()
    if table_name == "documents":
        table_name = DEFAULT_TABLE

    conn_str = _project_conn_str(table_name)
    result: dict[str, Any] = {"table_name": table_name}

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {CHUNKS_TABLE}")
                result["total_chunks"] = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(DISTINCT metadata->>'path') FROM {CHUNKS_TABLE}")
                result["total_files"] = cur.fetchone()[0]

                cur.execute("SELECT pg_database_size(current_database()) / (1024*1024.0)")
                result["db_size_mb"] = round(cur.fetchone()[0], 2)

                cur.execute(f"SELECT pg_table_size('{CHUNKS_TABLE}') / (1024*1024.0)")
                result["table_size_mb"] = round(cur.fetchone()[0], 2)

                cur.execute(f"SELECT pg_indexes_size('{CHUNKS_TABLE}') / (1024*1024.0)")
                result["index_size_mb"] = round(cur.fetchone()[0], 2)

                cur.execute(f"""
                    SELECT split_part(metadata->>'path', '.', -1) AS ext, COUNT(*) AS cnt
                    FROM {CHUNKS_TABLE} GROUP BY ext ORDER BY cnt DESC
                """)
                result["chunks_per_extension"] = {f".{row[0]}": row[1] for row in cur.fetchall()}

                cur.execute(f"""
                    SELECT AVG(cnt)::numeric(10,2), MAX(cnt), MAX(path)
                    FROM (SELECT metadata->>'path' AS path, COUNT(*) AS cnt
                          FROM {CHUNKS_TABLE} GROUP BY metadata->>'path') sub
                """)
                row = cur.fetchone()
                result["avg_chunks_per_file"] = float(row[0]) if row[0] else 0
                result["max_chunks_per_file"] = {"path": row[2] or "N/A", "chunks": row[1] or 0}

                cur.execute(
                    f"SELECT extensions.vector_dims(embedding) FROM {CHUNKS_TABLE} "
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
    global _project_id, _indexing_paused, _db_name

    # Auto-detect project
    table_name, watch_dir, project_name = resolve_project()
    set_search_table(table_name)
    _db_name = table_name
    logger.info("Project: %s (db: %s, dir: %s)", project_name, table_name, watch_dir)

    # Ensure registry and project DB (creates DB + schema automatically)
    ensure_registry()
    ensure_project_db(table_name)
    project = get_or_create_project(table_name, watch_dir, project_name)
    _project_id = project["id"]

    # Migrate from file-based state if needed
    migrate_file_state(table_name, watch_dir)

    # Scan directory and register all indexable files as pending
    new_count = scan_files_to_db(table_name, watch_dir, ALLOWED_EXT)
    logger.info("File scan: %d new pending files", new_count)

    # Start file watcher
    if os.path.isdir(watch_dir):
        _start_watcher(table_name, watch_dir)

    # Load indexing_paused state
    _indexing_paused = project.get("indexing_paused", False)

    # Start web dashboard
    from dashboard import DashboardState, start_dashboard
    dash_state = DashboardState(
        project_id=_project_id,
        project_path=watch_dir,
        project_name=project_name,
        table_name=table_name,
        watch_dir=watch_dir,
        indexing_paused=_indexing_paused,
        db_name=table_name,
    )
    _dashboard_state = dash_state
    cfg = load_app_config()
    dash_cfg = cfg.get("dashboard", {})
    auto_open = dash_cfg.get("auto_open", True)
    start_dashboard(dash_state, auto_open=auto_open)

    logger.info("Starting DirToRAG MCP server (stdio mode)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
