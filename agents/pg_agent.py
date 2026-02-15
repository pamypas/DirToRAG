"""
PostgreSQL hybrid search agent for DirToRAG.
Uses full-text search + vector similarity.
"""

import logging
from typing import List, Dict, Any

import httpx
import psycopg
from psycopg.rows import dict_row

from models_loader import load_app_config

logger = logging.getLogger("uvicorn.error")

# Global table name (set by CLI before server starts)
_search_table: str = "documents"


def set_search_table(table_name: str) -> None:
    """Set the global search table name."""
    global _search_table
    _search_table = table_name
    logger.info(f"Search table set to: {table_name}")


def get_search_table() -> str:
    """Get the current search table name."""
    return _search_table


def get_db_connection_string() -> str:
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


class PostgresSearchAgent:
    """
    Agent that provides context via hybrid search in PostgreSQL.
    Combines full-text search and vector similarity.
    """

    def __init__(self, config: dict | None = None, limit: int | None = None):
        """
        config:
          limit: int - max results to return
          table_name: str - table to search (optional, uses global if not set)
          full_text_weight: float
          semantic_weight: float
        """
        config = config or {}

        self.limit = limit if limit is not None else config.get("limit", 10)
        self.table_name = config.get("table_name")  # None = use global
        self.full_text_weight = config.get("full_text_weight", 1.0)
        self.semantic_weight = config.get("semantic_weight", 1.0)

        # Embedding config
        cfg = load_app_config()
        emb_cfg = cfg.get("embedding", {})
        self.embedding_api_base = emb_cfg.get("api_base", "")
        self.embedding_model = emb_cfg.get("model", "")

    def _get_table_name(self) -> str:
        """Get the table name to search."""
        return self.table_name if self.table_name else get_search_table()

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text via API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.embedding_api_base}/embeddings",
                json={"model": self.embedding_model, "input": [text]}
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract embedding from response
        if isinstance(data, list) and len(data) > 0 and "embedding" in data[0]:
            nested = data[0]["embedding"]
            if isinstance(nested, list) and len(nested) > 0:
                if isinstance(nested[0], list):
                    return nested[0]
                return nested
        return data[0]

    def build_context(self, user_message: str) -> str:
        """
        Build context for LLM based on hybrid search.
        Returns string (may be empty if no results).
        """
        import asyncio

        try:
            # Run async search in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self._hybrid_search(user_message))
            finally:
                loop.close()

            if not results:
                return ""

            # Format results
            context_parts: List[str] = []
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                content = result.get("content", "")
                path = metadata.get("path", "unknown") if metadata else "unknown"
                context_parts.append(f"[DOC {i}] file: {path}\n{content}\n")

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.exception("PostgresSearchAgent failed: %s", e)
            return ""

    async def _hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute hybrid search in PostgreSQL."""
        try:
            # Get embedding
            query_embedding = await self._get_embedding(query)

            if len(query_embedding) != 1024:
                logger.warning(f"Embedding size {len(query_embedding)} != 1024")

            table_name = self._get_table_name()
            func_name = f"hybrid_search_{table_name}"
            conn_str = get_db_connection_string()
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            with psycopg.connect(conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT * FROM {func_name}(%s, %s::extensions.vector, %s, %s, %s)",
                        (query, embedding_str, self.limit,
                         self.full_text_weight, self.semantic_weight)
                    )
                    results = cur.fetchall()

            return results

        except Exception as e:
            logger.exception("Hybrid search failed: %s", e)
            return []
