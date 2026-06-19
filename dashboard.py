"""
DirToRAG Web Dashboard.

FastAPI application served alongside the MCP stdio server.
Provides a web UI for managing indexing, viewing files, and monitoring activity.
"""

import logging
import queue
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models_loader import load_app_config
from state_db import (
    CHUNKS_TABLE,
    FILES_TABLE,
    get_project_by_table,
    get_files_page,
    get_file_stats,
    set_project_paused,
    ensure_registry,
    _project_conn_str,
)
from cli import init_database, index_directory, delete_chunks_for_file

logger = logging.getLogger("dashboard")


@dataclass
class DashboardState:
    project_id: int = 0
    project_path: str = ""
    project_name: str = ""
    table_name: str = ""
    db_name: str = ""
    watch_dir: str = ""
    indexing_paused: bool = False
    indexing_status: dict = field(default_factory=lambda: {
        "running": False, "percent": 0, "current_file": "", "done": 0, "total": 0,
    })
    activity_log: deque = field(default_factory=lambda: deque(maxlen=500))
    search_history: deque = field(default_factory=lambda: deque(maxlen=100))
    ws_queue: queue.Queue = field(default_factory=queue.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class DashboardLogHandler(logging.Handler):
    """Mirrors Python logging records into DashboardState.activity_log and ws_queue."""

    def __init__(self, state: DashboardState):
        super().__init__()
        self.state = state

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        entry = {
            "ts": record.created,
            "level": record.levelname,
            "type": "log",
            "message": msg,
        }
        self.state.activity_log.append(entry)
        try:
            self.state.ws_queue.put_nowait(entry)
        except queue.Full:
            pass


def create_dashboard_app(state: DashboardState) -> FastAPI:
    app = FastAPI(title="DirToRAG Dashboard", docs_url=None, redoc_url=None)

    # NOTE: all handlers use `def` (not `async def`) so FastAPI runs them
    # in a threadpool and never blocks the uvicorn event loop with
    # synchronous psycopg calls.

    @app.get("/api/state")
    def api_state():
        import psycopg

        table_exists = False
        stats = {}
        db_name = state.db_name or state.table_name
        try:
            conn_str = _project_conn_str(db_name)
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (CHUNKS_TABLE,),
                    )
                    table_exists = cur.fetchone()[0]
                    if table_exists:
                        cur.execute(f"SELECT COUNT(*) FROM {CHUNKS_TABLE}")
                        total_chunks = cur.fetchone()[0]
                        cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE}")
                        total_files = cur.fetchone()[0]
                        cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE} WHERE status = 'pending'")
                        pending_files = cur.fetchone()[0]
                        cur.execute(f"SELECT COUNT(*) FROM {FILES_TABLE} WHERE status = 'indexed'")
                        indexed_files = cur.fetchone()[0]
                        cur.execute(f"SELECT pg_table_size('{CHUNKS_TABLE}') / (1024*1024.0)")
                        table_size_mb = round(cur.fetchone()[0], 2)
                        cur.execute(f"SELECT pg_indexes_size('{CHUNKS_TABLE}') / (1024*1024.0)")
                        index_size_mb = round(cur.fetchone()[0], 2)
                        cur.execute("SELECT pg_database_size(current_database()) / (1024*1024.0)")
                        db_size_mb = round(cur.fetchone()[0], 2)

                        cur.execute(f"""
                            SELECT split_part(metadata->>'path', '.', -1) AS ext, COUNT(*) AS cnt
                            FROM {CHUNKS_TABLE} GROUP BY ext ORDER BY cnt DESC
                        """)
                        chunks_per_extension = {f".{row[0]}": row[1] for row in cur.fetchall()}

                        cur.execute(f"""
                            SELECT AVG(cnt)::numeric(10,2), MAX(cnt), MAX(path)
                            FROM (SELECT metadata->>'path' AS path, COUNT(*) AS cnt
                                  FROM {CHUNKS_TABLE} GROUP BY metadata->>'path') sub
                        """)
                        row = cur.fetchone()
                        avg_chunks = float(row[0]) if row[0] else 0
                        max_chunks_file = {"path": row[2] or "N/A", "chunks": row[1] or 0}

                        cur.execute(
                            f"SELECT extensions.vector_dims(embedding) FROM {CHUNKS_TABLE} "
                            "WHERE embedding IS NOT NULL LIMIT 1"
                        )
                        dim_row = cur.fetchone()
                        embedding_dim = dim_row[0] if dim_row else None

                        stats = {
                            "total_chunks": total_chunks,
                            "total_files": total_files,
                            "pending_files": pending_files,
                            "indexed_files": indexed_files,
                            "table_size_mb": table_size_mb,
                            "index_size_mb": index_size_mb,
                            "db_size_mb": db_size_mb,
                            "chunks_per_extension": chunks_per_extension,
                            "avg_chunks_per_file": avg_chunks,
                            "max_chunks_per_file": max_chunks_file,
                            "embedding_dim": embedding_dim,
                        }
        except Exception as e:
            logger.warning("State check failed: %s", e)
            table_exists = False

        file_stats = {}
        try:
            if state.db_name and table_exists:
                file_stats = get_file_stats(state.db_name)
        except Exception:
            pass

        return {
            "project": {
                "path": state.project_path,
                "name": state.project_name,
                "table": state.table_name,
                "db": state.db_name,
            },
            "table_exists": table_exists,
            "indexing": state.indexing_status,
            "indexing_paused": state.indexing_paused,
            "stats": stats,
            "file_stats": file_stats,
        }

    @app.post("/api/indexing/pause")
    def api_pause():
        state.indexing_paused = True
        if state.project_id:
            set_project_paused(state.project_id, True)
        return {"paused": True}

    @app.post("/api/indexing/resume")
    def api_resume():
        state.indexing_paused = False
        if state.project_id:
            set_project_paused(state.project_id, False)
        return {"paused": False}

    @app.post("/api/reindex")
    def api_reindex(body: dict | None = None):
        body = body or {}
        scope = body.get("scope", "changed")
        specific_path = body.get("path")

        if state.indexing_status.get("running"):
            return JSONResponse({"error": "Indexing already running"}, status_code=409)

        def _run_reindex():
            import psycopg
            from chunker import chunk_text
            from embedder import get_embeddings
            from cli import insert_to_postgres
            from state_db import upsert_indexed_file

            state.indexing_status = {"running": True, "percent": 0, "current_file": "", "done": 0, "total": 0}
            state.cancel_event.clear()
            try:
                if specific_path:
                    abs_path = str(Path(state.watch_dir) / specific_path)
                    if not Path(abs_path).exists():
                        logger.error("File not found: %s", abs_path)
                        return
                    db_name = state.db_name
                    conn_str = _project_conn_str(db_name)
                    delete_chunks_for_file(conn_str, specific_path)
                    text = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
                    chunks = chunk_text(text)
                    if chunks:
                        embeddings = get_embeddings(chunks)
                        records = [
                            {"content": c, "embedding": e, "metadata": {"path": specific_path}}
                            for c, e in zip(chunks, embeddings)
                        ]
                        insert_to_postgres(records, conn_str)
                    upsert_indexed_file(
                        db_name, specific_path,
                        Path(abs_path).stat().st_mtime,
                        chunk_count=len(chunks) if chunks else 0,
                        size_bytes=Path(abs_path).stat().st_size,
                    )
                    logger.info("Reindexed single file: %s", specific_path)
                else:
                    if scope == "all":
                        try:
                            with psycopg.connect(_project_conn_str(state.db_name), autocommit=True) as conn:
                                with conn.cursor() as cur:
                                    cur.execute(f"TRUNCATE public.{FILES_TABLE}")
                        except Exception:
                            pass
                    index_directory(
                        state.table_name,
                        state.watch_dir,
                        incremental=True,
                        progress_callback=lambda done, total, f: _update_progress(state, done, total, f),
                    )
            except Exception as e:
                logger.exception("Reindex failed: %s", e)
            finally:
                state.indexing_status = {"running": False, "percent": 100, "current_file": "", "done": 0, "total": 0}

        thread = threading.Thread(target=_run_reindex, daemon=True)
        thread.start()
        return {"started": True, "scope": scope}

    @app.get("/api/files")
    def api_files(limit: int = 50, offset: int = 0, status: str | None = None, search: str | None = None):
        if not state.db_name:
            return {"rows": [], "total": 0}
        rows, total = get_files_page(state.db_name, limit, offset, status, search)
        return {"rows": rows, "total": total}

    @app.delete("/api/files/{rel_path:path}")
    def api_delete_file(rel_path: str):
        if not state.db_name:
            return JSONResponse({"error": "No project"}, status_code=400)
        conn_str = _project_conn_str(state.db_name)
        n = delete_chunks_for_file(conn_str, rel_path)
        from state_db import delete_indexed_file
        delete_indexed_file(state.db_name, rel_path)
        return {"deleted_chunks": n}

    @app.get("/api/files/{rel_path:path}/chunks")
    def api_file_chunks(rel_path: str):
        import psycopg
        conn_str = _project_conn_str(state.db_name)
        try:
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id, content, metadata FROM {CHUNKS_TABLE} WHERE metadata->>'path' = %s ORDER BY id",
                        (rel_path,),
                    )
                    return [{"id": r[0], "content": r[1], "metadata": r[2]} for r in cur.fetchall()]
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/logs")
    def api_logs(since: float = 0):
        entries = [e for e in state.activity_log if e["ts"] > since]
        return entries

    @app.get("/api/searches")
    def api_searches():
        return list(state.search_history)

    @app.get("/api/projects")
    def api_projects():
        from state_db import list_projects
        return list_projects()

    @app.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                try:
                    entry = state.ws_queue.get(timeout=1.0)
                    await websocket.send_json(entry)
                except queue.Empty:
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Serve static files or embedded HTML
    static_dir = Path(__file__).parent / "dashboard_static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def _update_progress(state: DashboardState, done: int, total: int, current_file: str):
    percent = int(done * 100 / total) if total > 0 else 0
    state.indexing_status = {
        "running": True,
        "percent": percent,
        "current_file": current_file,
        "done": done,
        "total": total,
    }


def start_dashboard(state: DashboardState, port: int | None = None, auto_open: bool = True) -> threading.Thread:
    """Start the dashboard HTTP server in a daemon thread."""
    import uvicorn

    cfg = load_app_config()
    dash_cfg = cfg.get("dashboard", {})
    host = dash_cfg.get("host", "127.0.0.1")
    if port is None:
        port = dash_cfg.get("port", 8889)

    app = create_dashboard_app(state)

    # Attach log handler
    log_handler = DashboardLogHandler(state)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(log_handler)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)

    def _run():
        server.run()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if auto_open:
        webbrowser.open(f"http://{host}:{port}")

    logger.info("Dashboard started at http://%s:%d", host, port)
    return thread
