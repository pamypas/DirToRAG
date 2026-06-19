"""
Database-backed state management for DirToRAG.

Architecture:
- Central DB (dirtoRAG): stores dirto_rag_projects registry
- Per-project DB (dirto_rag_<name>): stores chunks + files tables

Each project gets its own PostgreSQL database. Inside it:
  - chunks: vectors + FTS + metadata (the main RAG data)
  - files: per-file indexing state (replaces .indexed_files.log)
  - hybrid_search(): SQL function for RRF search
"""

import logging
import re
from pathlib import Path

import psycopg
from psycopg.types.json import Json

from models_loader import load_app_config

logger = logging.getLogger(__name__)

# Fixed table names inside every project DB
CHUNKS_TABLE = "chunks"
FILES_TABLE = "files"


def _build_conn_str(dbname: str | None = None) -> str:
    """Build a PostgreSQL connection string. Uses central DB name if dbname not given."""
    cfg = load_app_config()
    db = cfg["database"]
    host, port = db["host"], db["port"]
    name = dbname or db["name"]
    user, password = db["user"], db.get("password", "")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return f"postgresql://{user}@{host}:{port}/{name}"


def _central_conn_str() -> str:
    """Connection string for the central registry DB."""
    return _build_conn_str()


def _project_conn_str(db_name: str) -> str:
    """Connection string for a per-project DB."""
    return _build_conn_str(db_name)


def _db_name_from_table(table_name: str) -> str:
    """Derive a PostgreSQL database name from the project table identifier.

    Table names like 'dirto_rag_myproject' or 'work' map to DB names
    like 'dirto_rag_myproject'. The DB name IS the table_name field.
    """
    return table_name


def ensure_registry() -> None:
    """Create/upgrade central registry table. Called lazily at startup."""
    with psycopg.connect(_central_conn_str(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.dirto_rag_projects (
                    id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    table_name text UNIQUE NOT NULL,
                    db_name text,
                    project_path text NOT NULL,
                    project_name text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    last_seen_at timestamptz NOT NULL DEFAULT now(),
                    last_indexed_at timestamptz,
                    indexing_paused boolean NOT NULL DEFAULT false,
                    settings jsonb NOT NULL DEFAULT '{}'::jsonb
                );
            """)
            # Migrate: add db_name column if missing (upgrade from old schema)
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'dirto_rag_projects' AND column_name = 'db_name'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE public.dirto_rag_projects ADD COLUMN db_name text")
                cur.execute("UPDATE public.dirto_rag_projects SET db_name = table_name WHERE db_name IS NULL")
                logger.info("Migrated dirto_rag_projects: added db_name column")

            # Migrate: drop dirto_rag_files table if it exists (moved to per-project DBs)
            cur.execute("DROP TABLE IF EXISTS public.dirto_rag_files")
    logger.info("Registry ensured")


def ensure_project_db(db_name: str) -> None:
    """Create the per-project database and its schema if they don't exist."""
    # Connect to central DB to check/create the project DB
    with psycopg.connect(_central_conn_str(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not cur.fetchone():
                # DB name must be safe — only lowercase alphanumeric + underscore
                if not re.match(r'^[a-z_][a-z0-9_]*$', db_name):
                    raise ValueError(f"Invalid database name: {db_name}")
                cur.execute(f'CREATE DATABASE {db_name}')
                logger.info("Created project database: %s", db_name)

    # Create schema inside the project DB
    with psycopg.connect(_project_conn_str(db_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            # Extensions
            cur.execute("CREATE SCHEMA IF NOT EXISTS extensions;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;")

            # Files table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS public.{FILES_TABLE} (
                    id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                    rel_path text NOT NULL,
                    mtime double precision NOT NULL,
                    size_bytes bigint NOT NULL DEFAULT 0,
                    chunk_count int NOT NULL DEFAULT 0,
                    indexed_at timestamptz NOT NULL DEFAULT now(),
                    status text NOT NULL DEFAULT 'pending',
                    error text,
                    UNIQUE (rel_path)
                );
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_files_status ON public.{FILES_TABLE} (status);")

            # Check if chunks table exists (it may have been created by init_database)
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (CHUNKS_TABLE,),
            )
            if not cur.fetchone()[0]:
                # Create chunks table + indexes + search function
                cur.execute(f"""
                    CREATE TABLE public.{CHUNKS_TABLE} (
                        id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                        content text,
                        fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                        embedding extensions.vector(1024),
                        metadata jsonb
                    );
                """)
                cur.execute(f"CREATE INDEX ON {CHUNKS_TABLE} USING gin(fts);")
                cur.execute(f"CREATE INDEX ON {CHUNKS_TABLE} USING hnsw (embedding extensions.vector_ip_ops);")
                _create_hybrid_search_function(cur, CHUNKS_TABLE)
                logger.info("Created chunks table and indexes in project DB: %s", db_name)


def _create_hybrid_search_function(cur, table_name: str) -> None:
    """Create the hybrid_search SQL function for a given chunks table."""
    func_name = "hybrid_search"
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
            LIMIT match_count * 2
        ),
        semantic AS (
            SELECT
                id,
                row_number() OVER (ORDER BY embedding <#> query_embedding) AS rank_ix
            FROM {table_name}
            ORDER BY rank_ix
            LIMIT match_count * 2
        )
        SELECT t.*
        FROM full_text
        FULL OUTER JOIN semantic ON full_text.id = semantic.id
        JOIN {table_name} t ON coalesce(full_text.id, semantic.id) = t.id
        ORDER BY
            coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
            coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight DESC
        LIMIT match_count
        $$;
    """)


# ---------------------------------------------------------------------------
# Project registry CRUD (central DB)
# ---------------------------------------------------------------------------

def get_or_create_project(table_name: str, project_path: str, project_name: str) -> dict:
    """Get existing or create new project. Returns dict with id, db_name, etc."""
    db_name = _db_name_from_table(table_name)

    with psycopg.connect(_central_conn_str(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, table_name, db_name, project_path, project_name, indexing_paused, settings FROM public.dirto_rag_projects WHERE table_name = %s",
                (table_name,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE public.dirto_rag_projects SET last_seen_at = now(), project_path = %s, project_name = %s WHERE id = %s",
                    (str(project_path), project_name, row[0]),
                )
                return _project_row_to_dict(row)

            cur.execute(
                "INSERT INTO public.dirto_rag_projects (table_name, db_name, project_path, project_name) VALUES (%s, %s, %s, %s) RETURNING id, table_name, db_name, project_path, project_name, indexing_paused, settings",
                (table_name, db_name, str(project_path), project_name),
            )
            return _project_row_to_dict(cur.fetchone())


def get_project_by_table(table_name: str) -> dict | None:
    """Look up a project by its table name."""
    with psycopg.connect(_central_conn_str()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, table_name, db_name, project_path, project_name, indexing_paused, settings FROM public.dirto_rag_projects WHERE table_name = %s",
                (table_name,),
            )
            row = cur.fetchone()
            return _project_row_to_dict(row) if row else None


def get_project_by_path(project_path: str) -> dict | None:
    """Look up a project by its resolved filesystem path."""
    with psycopg.connect(_central_conn_str()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, table_name, db_name, project_path, project_name, indexing_paused, settings FROM public.dirto_rag_projects WHERE project_path = %s",
                (str(project_path),),
            )
            row = cur.fetchone()
            return _project_row_to_dict(row) if row else None


def list_projects() -> list[dict]:
    """List all known projects."""
    with psycopg.connect(_central_conn_str()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, table_name, db_name, project_path, project_name, last_indexed_at, indexing_paused
                FROM public.dirto_rag_projects ORDER BY last_seen_at DESC
            """)
            return [
                {
                    "id": row[0],
                    "table_name": row[1],
                    "db_name": row[2],
                    "project_path": row[3],
                    "project_name": row[4],
                    "last_indexed_at": row[5].isoformat() if row[5] else None,
                    "indexing_paused": row[6],
                }
                for row in cur.fetchall()
            ]


def update_project_indexed_at(project_id: int) -> None:
    with psycopg.connect(_central_conn_str(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.dirto_rag_projects SET last_indexed_at = now() WHERE id = %s",
                (project_id,),
            )


def set_project_paused(project_id: int, paused: bool) -> None:
    with psycopg.connect(_central_conn_str(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.dirto_rag_projects SET indexing_paused = %s WHERE id = %s",
                (paused, project_id),
            )


def get_chunks_conn_str(table_name: str) -> tuple[str, str]:
    """Get connection string and actual chunks table name for a project.

    Tries the per-project DB first (table_name = DB name, chunks table = 'chunks').
    Falls back to the central DB with the table_name as the chunks table name
    (legacy mode for projects created before per-project DBs).

    Returns: (conn_str, chunks_table_name)
    """
    # Try per-project DB
    try:
        conn_str = _project_conn_str(table_name)
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    (CHUNKS_TABLE,),
                )
                if cur.fetchone()[0]:
                    return conn_str, CHUNKS_TABLE
    except Exception:
        pass

    # Fallback: central DB with legacy table name
    conn_str = _central_conn_str()
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table_name,),
            )
            if cur.fetchone()[0]:
                return conn_str, table_name

def _project_row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "table_name": row[1],
        "db_name": row[2],
        "project_path": row[3],
        "project_name": row[4],
        "indexing_paused": row[5],
        "settings": row[6],
    }


# ---------------------------------------------------------------------------
# Files CRUD (per-project DB)
# ---------------------------------------------------------------------------

def load_indexed_files(db_name: str) -> dict[str, float]:
    """Load indexed files for a project. Returns {rel_path: mtime}.

    Only includes files with status in ('indexed', 'empty', 'failed').
    Pending files are excluded so that index_directory() treats them as new.
    """
    with psycopg.connect(_project_conn_str(db_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT rel_path, mtime FROM public.{FILES_TABLE} WHERE status != 'pending'"
            )
            return {row[0]: row[1] for row in cur.fetchall()}


def upsert_indexed_file(
    db_name: str,
    rel_path: str,
    mtime: float,
    chunk_count: int = 0,
    size_bytes: int = 0,
    status: str = "indexed",
    error: str | None = None,
) -> None:
    with psycopg.connect(_project_conn_str(db_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO public.{FILES_TABLE} (rel_path, mtime, size_bytes, chunk_count, status, error)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (rel_path) DO UPDATE SET
                    mtime = EXCLUDED.mtime,
                    size_bytes = EXCLUDED.size_bytes,
                    chunk_count = EXCLUDED.chunk_count,
                    indexed_at = now(),
                    status = EXCLUDED.status,
                    error = EXCLUDED.error
            """, (rel_path, mtime, size_bytes, chunk_count, status, error))


def delete_indexed_file(db_name: str, rel_path: str) -> None:
    with psycopg.connect(_project_conn_str(db_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM public.{FILES_TABLE} WHERE rel_path = %s", (rel_path,))


def delete_stale_files(db_name: str, current_paths: set[str]) -> int:
    with psycopg.connect(_project_conn_str(db_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM public.{FILES_TABLE} WHERE rel_path != ALL(%s)",
                (list(current_paths),),
            )
            return cur.rowcount


def get_file_stats(db_name: str) -> dict:
    with psycopg.connect(_project_conn_str(db_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT status, COUNT(*), SUM(size_bytes), SUM(chunk_count)
                FROM public.{FILES_TABLE}
                GROUP BY status
            """)
            stats = {}
            for row in cur.fetchall():
                stats[row[0]] = {
                    "count": row[1],
                    "total_bytes": row[2] or 0,
                    "total_chunks": row[3] or 0,
                }
            return stats


def get_files_page(
    db_name: str,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    search: str | None = None,
) -> tuple[list[dict], int]:
    with psycopg.connect(_project_conn_str(db_name)) as conn:
        with conn.cursor() as cur:
            where = "WHERE 1=1"
            params: list = []
            if status:
                where += " AND status = %s"
                params.append(status)
            if search:
                where += " AND rel_path ILIKE %s"
                params.append(f"%{search}%")

            cur.execute(f"SELECT COUNT(*) FROM public.{FILES_TABLE} {where}", params)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT rel_path, mtime, size_bytes, chunk_count, indexed_at, status, error
                FROM public.{FILES_TABLE}
                {where}
                ORDER BY rel_path
                LIMIT %s OFFSET %s
            """, params + [limit, offset])

            rows = [
                {
                    "rel_path": row[0],
                    "mtime": row[1],
                    "size_bytes": row[2],
                    "chunk_count": row[3],
                    "indexed_at": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "error": row[6],
                }
                for row in cur.fetchall()
            ]
            return rows, total


def scan_files_to_db(db_name: str, watch_dir: str, allowed_ext: set[str]) -> int:
    """Scan directory and upsert all indexable files into the files table.

    Files not yet in the table are inserted with status='pending'.
    Files already in the table with status in ('indexed', 'empty', 'failed') are
    left untouched. Pending files whose rel_path no longer exists on disk are deleted.

    Returns the number of newly inserted pending files.
    """
    import os
    from cli import SKIP_DIRS

    conn_str = _project_conn_str(db_name)
    new_count = 0

    # Load existing file states
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT rel_path, status FROM public.{FILES_TABLE}")
            existing = {row[0]: row[1] for row in cur.fetchall()}

    # Scan filesystem
    disk_paths: set[str] = set()
    pending_rows: list[tuple] = []

    for root, dirs, files in os.walk(watch_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for fname in files:
            if fname.startswith("."):
                continue
            p = Path(root) / fname
            if p.suffix.lower() not in allowed_ext:
                continue

            rel_path = str(p.relative_to(watch_dir))
            disk_paths.add(rel_path)

            if rel_path in existing:
                continue

            try:
                stat = p.stat()
                pending_rows.append((rel_path, stat.st_mtime, stat.st_size))

            except OSError:
                continue

    # Batch insert new pending files
    if pending_rows:
        with psycopg.connect(conn_str, autocommit=True) as conn:
            with conn.cursor() as cur:
                for rel_path, mtime, size_bytes in pending_rows:
                    cur.execute(f"""
                        INSERT INTO public.{FILES_TABLE} (rel_path, mtime, size_bytes, chunk_count, status)
                        VALUES (%s, %s, %s, 0, 'pending')
                        ON CONFLICT (rel_path) DO NOTHING
                    """, (rel_path, mtime, size_bytes))
                    if cur.rowcount > 0:
                        new_count += 1

    # Delete pending files no longer on disk
    stale_count = 0
    if existing:
        stale_pending = [
            rp for rp, st in existing.items()
            if st == "pending" and rp not in disk_paths
        ]
        if stale_pending:
            with psycopg.connect(conn_str, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM public.{FILES_TABLE} WHERE status = 'pending' AND rel_path = ANY(%s)",
                        (stale_pending,),
                    )
                    stale_count = cur.rowcount

    logger.info("File scan: %d new pending, %d on disk, %d stale removed", new_count, len(disk_paths), stale_count)
    return new_count


# ---------------------------------------------------------------------------
# Migration from file-based state
# ---------------------------------------------------------------------------

def migrate_file_state(db_name: str, watch_dir: str) -> int:
    """Migrate .indexed_files.log to the files table in the project DB."""
    log_path = Path(watch_dir) / ".indexed_files.log"
    if not log_path.exists():
        return 0

    entries: dict[str, float] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                entries[parts[0]] = float(parts[1])
            else:
                entries[parts[0]] = 0.0

    if not entries:
        log_path.rename(log_path.with_suffix(".log.migrated"))
        return 0

    with psycopg.connect(_project_conn_str(db_name), autocommit=True) as conn:
        with conn.cursor() as cur:
            # Get chunk counts from chunks table
            cur.execute(f"""
                SELECT metadata->>'path' AS path, COUNT(*) AS cnt
                FROM {CHUNKS_TABLE}
                WHERE metadata->>'path' = ANY(%s)
                GROUP BY metadata->>'path'
            """, (list(entries.keys()),))
            chunk_counts = {row[0]: row[1] for row in cur.fetchall()}

            for rel_path, mtime in entries.items():
                chunk_count = chunk_counts.get(rel_path, 0)
                cur.execute(f"""
                    INSERT INTO public.{FILES_TABLE} (rel_path, mtime, chunk_count, status)
                    VALUES (%s, %s, %s, 'indexed')
                    ON CONFLICT (rel_path) DO UPDATE SET
                        mtime = EXCLUDED.mtime,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = now()
                """, (rel_path, mtime, chunk_count))

    log_path.rename(log_path.with_suffix(".log.migrated"))
    logger.info("Migrated %d entries from %s to DB", len(entries), log_path)
    return len(entries)


def table_exists_in_project_db(db_name: str) -> bool:
    """Check if the chunks table exists in the project DB."""
    try:
        with psycopg.connect(_project_conn_str(db_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    (CHUNKS_TABLE,),
                )
                return cur.fetchone()[0]
    except Exception:
        return False
