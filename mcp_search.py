#!/usr/bin/env python3
"""
MCP server for retrieval-only codebase search via DirToRAG.

Runs as a stdio process. Claude Code calls search_codebase / get_index_stats,
and the server returns relevant chunks without any LLM generation.

For debugging:
    python mcp_search.py           # stdio mode
    mcp dev mcp_search.py          # MCP Inspector (web UI)
"""

import os
import sys
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_search")

DEFAULT_TABLE = os.environ.get("DIRTORAG_TABLE", "documents")

mcp = FastMCP(
    name="DirToRAG Search",
    description="Hybrid codebase search (full-text + semantic) via DirToRAG",
)

_search_agent: PostgresSearchAgent | None = None


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


def main():
    """Entry point for MCP server."""
    if len(sys.argv) > 1:
        table = sys.argv[1]
        set_search_table(table)
        logger.info("Table set from command line: %s", table)
    else:
        set_search_table(DEFAULT_TABLE)
        logger.info("Using default table: %s", DEFAULT_TABLE)

    logger.info("Starting DirToRAG MCP server (stdio mode)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
