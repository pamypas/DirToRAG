#!/usr/bin/env python3
"""
PreToolUse(Read) hook script — finds semantically related chunks for a file.

Called by Claude Code BEFORE each Read tool call. Takes a file path,
uses its content to find similar chunks from OTHER files via DirToRAG,
and returns them as additionalContext for transparent injection into
the system prompt.

Usage:
    python find_related.py <absolute_path_to_file> [--table work] [--limit 5]

Output (stdout): formatted context string with related chunks.
Output (stderr): debug/log messages.
Exit codes: 0 = success (with or without results), 1 = error.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("find_related")

# Extensions likely to be text (skip binaries)
TEXT_EXTENSIONS = {
    ".py", ".pp", ".yaml", ".yml", ".erb", ".epp", ".md", ".txt",
    ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish", ".sql", ".json", ".xml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".css", ".scss",
    ".html", ".vue", ".svelte", ".tf", ".proto",
}


def is_text_file(file_path: str) -> bool:
    """Check if the file has a text-like extension."""
    ext = Path(file_path).suffix.lower()
    return ext in TEXT_EXTENSIONS


def build_search_query(file_path: str, max_chars: int = 1500) -> str:
    """Read file and build a search query from its content."""
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", file_path, e)
        return ""

    if not content.strip():
        return ""

    return content[:max_chars].strip()


def format_related_chunks(
    file_path: str,
    results: list[dict],
    max_results: int = 5,
) -> str:
    """Format search results as context for injection."""
    current_filename = Path(file_path).name
    current_path = str(Path(file_path))

    filtered: list[dict] = []
    for r in results:
        result_path = r.get("file_path", "")
        if result_path in current_path or current_path.endswith(result_path):
            continue
        if Path(result_path).name == current_filename:
            continue
        filtered.append(r)

    if not filtered:
        return ""

    lines: list[str] = []
    lines.append(f"--- Related code (semantically similar to {current_filename}) ---")
    lines.append("")

    for i, r in enumerate(filtered[:max_results], 1):
        file_path_rel = r.get("file_path", "unknown")
        content = r.get("content", "")
        score = r.get("score", "N/A")

        lines.append(f"### {i}. {file_path_rel} (score: {score})")
        lines.append("```")
        lines.append(content[:800])
        if len(content) > 800:
            lines.append("... (truncated)")
        lines.append("```")
        lines.append("")

    lines.append("--- End of related code ---")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find semantically related chunks for a file"
    )
    parser.add_argument("file_path", help="Absolute path to the file")
    parser.add_argument("--table", default="work", help="DB table name")
    parser.add_argument("--limit", type=int, default=5, help="Max related chunks")
    args = parser.parse_args()

    file_path = args.file_path

    if not is_text_file(file_path):
        sys.exit(0)

    query = build_search_query(file_path)
    if not query:
        sys.exit(0)

    from agents.pg_agent import PostgresSearchAgent, set_search_table, get_search_table

    table = get_search_table()
    if table == "documents":
        set_search_table(args.table)

    agent = PostgresSearchAgent(
        config={"table_name": args.table, "limit": args.limit * 3}
    )

    results = agent.search_raw(query, limit=args.limit * 3)
    if not results:
        sys.exit(0)

    context = format_related_chunks(file_path, results, max_results=args.limit)
    if context:
        print(context)

    sys.exit(0)


if __name__ == "__main__":
    main()
