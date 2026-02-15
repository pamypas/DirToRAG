#!/usr/bin/env python3
"""
DirToRAG CLI - Unified entry point for database initialization, indexing, and server.

Usage:
    python cli.py init <table>                    - Initialize database table
    python cli.py index <table> <directory>       - Index a directory
    python cli.py index <table> <directory> --dry-run  - Dry-run mode (no DB writes)
    python cli.py serve <table>                   - Start the LLM server
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Set, Dict, Any

import psycopg
from psycopg.types.json import Json

from models_loader import load_app_config
from embedder import get_embeddings
from chunker import chunk_text

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

# Log file for indexed files
INDEXED_LOG_FILENAME = ".indexed_files.log"


def get_embedding_config() -> tuple[int, int]:
    """Get batch_size and concurrency from config."""
    cfg = load_app_config()
    emb_cfg = cfg.get("embedding", {})
    batch_size = emb_cfg.get("batch_size", 10)
    concurrency = emb_cfg.get("concurrency", 48)
    return batch_size, concurrency


def get_postgres_connection_string() -> str:
    """Build PostgreSQL connection string from config."""
    cfg = load_app_config()
    db_cfg = cfg.get("database", {})

    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 5432)
    dbname = db_cfg.get("name", "dirtoRAG")
    user = db_cfg.get("user", "postgres")
    password = db_cfg.get("password", "")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"


def init_database(table_name: str) -> None:
    """Initialize database with vector extension and specified table."""
    conn_str = get_postgres_connection_string()
    logger.info(f"Connecting to database: {conn_str.split('@')[-1]}")

    with psycopg.connect(conn_str, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Create extensions schema
            logger.info("Creating extensions schema...")
            cur.execute("CREATE SCHEMA IF NOT EXISTS extensions;")

            # Create vector extension
            logger.info("Creating vector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;")

            # Drop existing table if exists
            cur.execute(f"DROP TABLE IF EXISTS public.{table_name};")

            # Create table
            logger.info(f"Creating table '{table_name}'...")
            cur.execute(f"""
                CREATE TABLE public.{table_name} (
                    id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    content text,
                    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    embedding extensions.vector(1024),
                    metadata jsonb
                );
            """)

            # Create indexes
            logger.info("Creating indexes...")
            cur.execute(f"CREATE INDEX ON {table_name} USING gin(fts);")
            cur.execute(f"CREATE INDEX ON {table_name} USING hnsw (embedding extensions.vector_ip_ops);")

            # Create hybrid_search function for this table
            func_name = f"hybrid_search_{table_name}"
            logger.info(f"Creating hybrid_search function '{func_name}'...")
            cur.execute(f"""
                CREATE OR REPLACE FUNCTION public.{func_name}(
                    query_text text,
                    query_embedding extensions.vector(1024),
                    match_count int,
                    full_text_weight float = 1,
                    semantic_weight float = 1,
                    rrf_k int = 50
                )
                RETURNS TABLE (
                    id bigint,
                    content text,
                    fts tsvector,
                    embedding extensions.vector(1024),
                    metadata jsonb
                )
                LANGUAGE sql
                SET search_path = public, extensions
                AS $$
                WITH full_text AS (
                    SELECT
                        id,
                        row_number() OVER(ORDER BY ts_rank_cd(fts, websearch_to_tsquery(query_text)) DESC) AS rank_ix
                    FROM {table_name}
                    WHERE fts @@ websearch_to_tsquery(query_text)
                    ORDER BY rank_ix
                    LIMIT least(match_count, 30) * 2
                ),
                semantic AS (
                    SELECT
                        id,
                        row_number() OVER (ORDER BY embedding <#> query_embedding) AS rank_ix
                    FROM {table_name}
                    ORDER BY rank_ix
                    LIMIT least(match_count, 30) * 2
                )
                SELECT t.*
                FROM full_text
                FULL OUTER JOIN semantic ON full_text.id = semantic.id
                JOIN {table_name} t ON coalesce(full_text.id, semantic.id) = t.id
                ORDER BY
                    coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
                    coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight DESC
                LIMIT least(match_count, 30)
                $$;
            """)

    logger.info(f"Table '{table_name}' initialized successfully!")


def iter_files(repo_path: Path) -> List[Path]:
    """Iterate over allowed files in directory."""
    for root, dirs, files in os.walk(repo_path):
        # Skip directories starting with dot
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fname in files:
            # Skip files starting with dot
            if fname.startswith("."):
                continue
            p = Path(root) / fname
            if p.suffix.lower() in ALLOWED_EXT:
                yield p


def load_indexed_files(log_path: Path) -> Set[str]:
    """Load set of already indexed file paths."""
    if not log_path.exists():
        return set()
    indexed: Set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indexed.add(line)
    return indexed


def append_indexed_file(log_path: Path, rel_path: str) -> None:
    """Append a file path to the indexed log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(rel_path + "\n")


def insert_to_postgres(records: List[Dict[str, Any]], conn_str: str, table_name: str) -> None:
    """Insert records into PostgreSQL."""
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for record in records:
                # Convert embedding list to string format for pgvector
                embedding = record.get("embedding")
                if isinstance(embedding, list):
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                else:
                    embedding_str = embedding

                cur.execute(
                    f"""
                    INSERT INTO {table_name} (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    """,
                    (record["content"], embedding_str, Json(record["metadata"]))
                )
        conn.commit()


def index_directory(table_name: str, directory: str, dry_run: bool = False) -> None:
    """Index a directory into the database table."""
    repo_path = Path(directory).resolve()
    if not repo_path.is_dir():
        logger.error(f"Directory not found: {repo_path}")
        sys.exit(1)

    log_path = repo_path / INDEXED_LOG_FILENAME
    conn_str = get_postgres_connection_string() if not dry_run else ""

    # Get embedding config
    batch_size, concurrency = get_embedding_config()

    # Load already indexed files
    indexed_files_set = load_indexed_files(log_path)

    # Count total files
    all_files = list(iter_files(repo_path))
    total_files = len(all_files)
    if total_files == 0:
        logger.info("No files to index")
        return

    indexed_files = 0
    last_progress = -1

    # Thread pool for embedding requests
    executor = ThreadPoolExecutor(max_workers=concurrency)

    for fpath in all_files:
        rel_path = str(fpath.relative_to(repo_path))

        # Skip already indexed files (unless dry-run)
        if not dry_run and rel_path in indexed_files_set:
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {rel_path}: {e}")
            continue

        chunks = chunk_text(text)

        if not chunks:
            if not dry_run:
                append_indexed_file(log_path, rel_path)
                indexed_files += 1
            continue

        # Split into batches for parallel processing
        batches = [chunks[i:i + batch_size]
                   for i in range(0, len(chunks), batch_size)]

        # Send requests in parallel
        futures = [executor.submit(get_embeddings, batch) for batch in batches]

        all_embs = []
        file_failed = False
        for future in futures:
            try:
                batch_embs = future.result()
                all_embs.extend(batch_embs)
            except Exception as e:
                logger.error(f"Error processing {rel_path}: {e}")
                file_failed = True
                break

        if file_failed:
            continue

        # Adjust chunks if embeddings are fewer
        if len(all_embs) < len(chunks):
            chunks = chunks[:len(all_embs)]

        if not all_embs:
            continue

        # Dry-run mode: print to console
        if dry_run:
            print(f"\n{'=' * 20} FILE: {rel_path} {'=' * 20}")
            for i, (chunk, emb) in enumerate(zip(chunks, all_embs)):
                print(f"\n--- Chunk {i + 1} ---")
                print("Content:")
                print(chunk)
                print(f"\nEmbedding (first 10 values): {emb[:10]}")
                print(f"Vector size: {len(emb)}")
            continue

        # Prepare records for insertion
        records_to_insert = [
            {
                "content": chunk,
                "embedding": emb,
                "metadata": {"path": rel_path}
            }
            for chunk, emb in zip(chunks, all_embs)
        ]

        # Insert with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                insert_to_postgres(records_to_insert, conn_str, table_name)
                break
            except Exception as e:
                logger.error(f"Insert error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to insert {rel_path} after {max_retries} attempts")
                    continue

        # Mark as indexed
        append_indexed_file(log_path, rel_path)
        indexed_files_set.add(rel_path)
        indexed_files += 1

        # Progress
        progress = int(indexed_files * 100 / total_files)
        if progress != last_progress:
            logger.info(f"Progress: {progress}% ({indexed_files}/{total_files} files)")
            last_progress = progress

    executor.shutdown(wait=True)
    logger.info(f"Indexing finished: {indexed_files}/{total_files} files processed")


def run_server(table_name: str) -> None:
    """Start the LLM server with specified table."""
    import uvicorn
    from server import app, set_search_table

    # Set the table name for the server
    set_search_table(table_name)

    cfg = load_app_config()
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    logger.info(f"Starting server on {host}:{port} with table '{table_name}'")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(
        description="DirToRAG CLI - Unified entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py init my_repo           Initialize table 'my_repo'
    python cli.py index my_repo ./src    Index ./src into table 'my_repo'
    python cli.py index my_repo ./src --dry-run   Dry-run mode
    python cli.py serve my_repo          Start LLM server with table 'my_repo'
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database table")
    init_parser.add_argument("table", help="Table name to create")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a directory")
    index_parser.add_argument("table", help="Table name to index into")
    index_parser.add_argument("directory", help="Directory to index")
    index_parser.add_argument("--dry-run", action="store_true",
                              help="Print chunks without writing to DB")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the LLM server")
    serve_parser.add_argument("table", help="Table name to search")

    args = parser.parse_args()

    if args.command == "init":
        init_database(args.table)
    elif args.command == "index":
        index_directory(args.table, args.directory, args.dry_run)
    elif args.command == "serve":
        run_server(args.table)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
