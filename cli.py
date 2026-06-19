#!/usr/bin/env python3
"""
DirToRAG CLI - Unified entry point for database initialization, indexing, and server.

Usage:
    python cli.py init <table>                    - Initialize project database
    python cli.py index <table> <directory>       - Index a directory
    python cli.py index <table> <directory> --dry-run  - Dry-run mode
    python cli.py serve <table>                   - Start the LLM server
"""

import argparse
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Callable

import psycopg
from psycopg.types.json import Json

from models_loader import load_app_config
from embedder import get_embeddings
from chunker import chunk_text
from state_db import (
    CHUNKS_TABLE,
    FILES_TABLE,
    ensure_registry,
    ensure_project_db,
    get_or_create_project,
    load_indexed_files as db_load_indexed_files,
    upsert_indexed_file,
    delete_indexed_file,
    update_project_indexed_at,
    migrate_file_state,
    _project_conn_str,
)
from project_resolver import create_marker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable system proxy
for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

# Allowed file extensions for indexing
ALLOWED_EXT = {".pp", ".yaml", ".yml", ".erb", ".epp", ".md", ".txt"}


def get_embedding_config() -> tuple[int, int]:
    """Get batch_size and concurrency from config."""
    cfg = load_app_config()
    emb_cfg = cfg.get("embedding", {})
    batch_size = emb_cfg.get("batch_size", 10)
    concurrency = emb_cfg.get("concurrency", 48)
    return batch_size, concurrency


def get_postgres_connection_string(db_name: str | None = None) -> str:
    """Build PostgreSQL connection string. Uses project DB name if given, else central DB."""
    cfg = load_app_config()
    db_cfg = cfg.get("database", {})

    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 5432)
    dbname = db_name or db_cfg.get("name", "dirtoRAG")
    user = db_cfg.get("user", "postgres")
    password = db_cfg.get("password", "")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"


def init_database(table_name: str, project_path: str | None = None) -> None:
    """Initialize a project database with chunks table, files table, and search function."""
    ensure_registry()

    db_name = table_name
    logger.info("Initializing project database: %s", db_name)

    # Ensure the project DB exists with full schema
    ensure_project_db(db_name)

    # Register project in the central registry
    if project_path:
        project_name = Path(project_path).name
        get_or_create_project(table_name, project_path, project_name)

    logger.info("Project database '%s' initialized successfully!", db_name)


SKIP_DIRS = {"venv", "node_modules", "__pycache__", ".git", ".tox", "dist", "build", ".mypy_cache", ".pytest_cache"}

def iter_files(repo_path: Path) -> List[Path]:
    """Itererate over allowed files in directory."""
    for root, dirs, files in os.walk(repo_path):
        # Skip dot-dirs and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]

        for fname in files:
            # Skip files starting with dot
            if fname.startswith("."):
                continue
            p = Path(root) / fname
            if p.suffix.lower() in ALLOWED_EXT:
                yield p


def insert_to_postgres(records: List[Dict[str, Any]], conn_str: str) -> None:
    """Insert records into the chunks table."""
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for record in records:
                embedding = record.get("embedding")
                if isinstance(embedding, list):
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                else:
                    embedding_str = embedding

                cur.execute(
                    f"""
                    INSERT INTO {CHUNKS_TABLE} (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    """,
                    (record["content"], embedding_str, Json(record["metadata"]))
                )
        conn.commit()


def delete_chunks_for_file(conn_str: str, rel_path: str) -> int:
    """Delete all chunks for a given file path. Returns number of deleted rows."""
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {CHUNKS_TABLE} WHERE metadata->>'path' = %s",
                (rel_path,),
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted


def index_directory(
    table_name: str,
    directory: str,
    dry_run: bool = False,
    incremental: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """
    Index a directory into the project database.

    Args:
        table_name: Project identifier (used as DB name)
        directory: Path to directory to index
        dry_run: If True, print chunks without writing to DB
        incremental: If True, detect changed files (by mtime) and reindex only those.
        progress_callback: Optional callback(done, total, current_file) for progress reporting.
    """
    ensure_registry()

    repo_path = Path(directory).resolve()
    if not repo_path.is_dir():
        logger.error(f"Directory not found: {repo_path}")
        sys.exit(1)

    # SIGINT/SIGTERM handling for graceful interruption
    _interrupted = False

    def _signal_handler(signum, frame):
        nonlocal _interrupted
        if _interrupted:
            logger.warning("Second interrupt received, forcing exit...")
            sys.exit(1)
        _interrupted = True
        logger.info("Interrupt received. Finishing current file, then stopping...")

    prev_sigint = signal.signal(signal.SIGINT, _signal_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    db_name = table_name
    conn_str = get_postgres_connection_string(db_name) if not dry_run else ""

    # Get or create project in registry
    project_name = repo_path.name
    project = get_or_create_project(table_name, str(repo_path), project_name)

    # Ensure project DB and schema
    if not dry_run:
        ensure_project_db(db_name)

    # Migrate from file-based state if needed
    if not dry_run:
        migrate_file_state(db_name, str(repo_path))

    batch_size, concurrency = get_embedding_config()

    indexed_files = db_load_indexed_files(db_name) if not dry_run else {}

    all_files = list(iter_files(repo_path))
    total_files = len(all_files)
    if total_files == 0:
        logger.info("No files to index")
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        return

    current_file_paths: set[str] = set()
    files_to_index: list[tuple[Path, str, bool]] = []

    for fpath in all_files:
        rel_path = str(fpath.relative_to(repo_path))
        current_file_paths.add(rel_path)
        current_mtime = fpath.stat().st_mtime

        if rel_path not in indexed_files:
            files_to_index.append((fpath, rel_path, False))
        elif incremental:
            indexed_mtime = indexed_files[rel_path]
            if indexed_mtime == 0.0 or abs(current_mtime - indexed_mtime) > 1.0:
                files_to_index.append((fpath, rel_path, True))

    # Detect deleted files (only in incremental mode)
    deleted_files: list[str] = []
    if incremental and not dry_run:
        for rel_path in indexed_files:
            if rel_path not in current_file_paths:
                deleted_files.append(rel_path)
                n = delete_chunks_for_file(conn_str, rel_path)
                logger.info("Deleted %d chunks for removed file: %s", n, rel_path)
                delete_indexed_file(db_name, rel_path)

    if not files_to_index and not deleted_files:
        logger.info("No files to index (everything up to date)")
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        return

    changed_count = sum(1 for _, _, changed in files_to_index if changed)
    new_count = len(files_to_index) - changed_count
    logger.info(
        "Files to index: %d (%d new, %d changed)",
        len(files_to_index), new_count, changed_count,
    )

    executor = ThreadPoolExecutor(max_workers=concurrency)
    indexed_count = 0
    last_progress = -1

    for fpath, rel_path, is_changed in files_to_index:
        if _interrupted:
            logger.info("Indexing interrupted by user")
            break

        if is_changed and not dry_run:
            n = delete_chunks_for_file(conn_str, rel_path)
            if n > 0:
                logger.debug("Deleted %d old chunks for: %s", n, rel_path)

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read %s: %s", rel_path, e)
            continue

        chunks = chunk_text(text)

        if not chunks:
            if not dry_run:
                upsert_indexed_file(
                    db_name, rel_path, fpath.stat().st_mtime,
                    chunk_count=0, size_bytes=fpath.stat().st_size, status="empty",
                )
                indexed_count += 1
            continue

        batches = [chunks[i:i + batch_size]
                   for i in range(0, len(chunks), batch_size)]
        futures = [executor.submit(get_embeddings, batch) for batch in batches]

        all_embs = []
        file_failed = False
        for future in futures:
            try:
                batch_embs = future.result()
                all_embs.extend(batch_embs)
            except Exception as e:
                logger.error("Error processing %s: %s", rel_path, e)
                file_failed = True
                break

        if file_failed:
            if not dry_run:
                upsert_indexed_file(
                    db_name, rel_path, fpath.stat().st_mtime,
                    chunk_count=0, size_bytes=fpath.stat().st_size,
                    status="failed", error="embedding error",
                )
            continue

        if len(all_embs) < len(chunks):
            chunks = chunks[:len(all_embs)]

        if not all_embs:
            if not dry_run:
                upsert_indexed_file(
                    db_name, rel_path, fpath.stat().st_mtime,
                    chunk_count=0, size_bytes=fpath.stat().st_size, status="empty",
                )
                indexed_count += 1
            continue

        if dry_run:
            print(f"\n{'=' * 20} FILE: {rel_path} {'=' * 20}")
            for i, (chunk, emb) in enumerate(zip(chunks, all_embs)):
                print(f"\n--- Chunk {i + 1} ---")
                print("Content:")
                print(chunk)
                print(f"\nEmbedding (first 10 values): {emb[:10]}")
                print(f"Vector size: {len(emb)}")
            continue

        records_to_insert = [
            {"content": chunk, "embedding": emb, "metadata": {"path": rel_path}}
            for chunk, emb in zip(chunks, all_embs)
        ]

        max_retries = 3
        inserted = False
        for attempt in range(max_retries):
            try:
                insert_to_postgres(records_to_insert, conn_str)
                inserted = True
                break
            except Exception as e:
                logger.error("Insert error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2)

        if not inserted:
            logger.error("Failed to insert %s after %d attempts", rel_path, max_retries)
            upsert_indexed_file(
                db_name, rel_path, fpath.stat().st_mtime,
                chunk_count=0, size_bytes=fpath.stat().st_size,
                status="failed", error="insert failed",
            )
            continue

        upsert_indexed_file(
            db_name, rel_path, fpath.stat().st_mtime,
            chunk_count=len(chunks), size_bytes=fpath.stat().st_size,
        )
        indexed_count += 1

        total = len(files_to_index)
        progress = int(indexed_count * 100 / total)
        if progress != last_progress:
            logger.info("Progress: %d%% (%d/%d files)", progress, indexed_count, total)
            last_progress = progress
            if progress_callback:
                progress_callback(indexed_count, total, rel_path)

    executor.shutdown(wait=True)

    if not dry_run:
        update_project_indexed_at(project["id"])

    logger.info("Indexing finished: %d/%d files processed", indexed_count, len(files_to_index))
    if deleted_files:
        logger.info("Deleted chunks for %d removed files", len(deleted_files))

    signal.signal(signal.SIGINT, prev_sigint)
    signal.signal(signal.SIGTERM, prev_sigterm)


def run_server(table_name: str) -> None:
    """Start the LLM server with specified project."""
    import uvicorn
    from server import app, set_search_table

    set_search_table(table_name)

    cfg = load_app_config()
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    logger.info(f"Starting server on {host}:{port} with project '{table_name}'")
    uvicorn.run("server:app", host=host, port=port, reload=False, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="DirToRAG CLI - Unified entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py init my_repo           Initialize project database 'my_repo'
    python cli.py index my_repo ./src    Index ./src into project 'my_repo'
    python cli.py index my_repo ./src --dry-run   Dry-run mode
    python cli.py index my_repo ./src --incremental   Reindex changed files only
    python cli.py register ./src         Create .dirtoRAG.yaml marker in ./src
    python cli.py serve my_repo          Start LLM server with project 'my_repo'
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser("init", help="Initialize project database")
    init_parser.add_argument("table", help="Project name (used as DB name)")
    init_parser.add_argument("--project-path", help="Project directory path (for registry)")

    index_parser = subparsers.add_parser("index", help="Index a directory")
    index_parser.add_argument("table", help="Project name (used as DB name)")
    index_parser.add_argument("directory", help="Directory to index")
    index_parser.add_argument("--dry-run", action="store_true", help="Print chunks without writing to DB")
    index_parser.add_argument("--incremental", action="store_true",
        help="Detect changed files by mtime and reindex only those.")

    serve_parser = subparsers.add_parser("serve", help="Start the LLM server")
    serve_parser.add_argument("table", help="Project name")

    register_parser = subparsers.add_parser("register", help="Create .dirtoRAG.yaml marker in a directory")
    register_parser.add_argument("directory", help="Directory to register")
    register_parser.add_argument("--table", help="Override project name (auto-derived if not set)")
    register_parser.add_argument("--name", help="Override project display name")

    args = parser.parse_args()

    if args.command == "init":
        init_database(args.table, project_path=args.project_path)
    elif args.command == "index":
        index_directory(args.table, args.directory, args.dry_run, incremental=args.incremental)
    elif args.command == "serve":
        run_server(args.table)
    elif args.command == "register":
        marker_path = create_marker(args.directory, table_name=args.table, project_name=args.name)
        print(f"Created: {marker_path}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
