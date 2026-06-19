"""
Project auto-detection for DirToRAG MCP server.

Resolves which project (table + watch directory) to use based on the
current environment, using a layered approach:

1. Explicit override (DIRTORAG_TABLE + DIRTORAG_WATCH_DIR env vars or argv)
2. .dirtoRAG.yaml marker file (walk up from CWD)
3. Git repository root
4. CWD fallback
"""

import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Convert a directory name to a valid SQL table name."""
    sanitized = re.sub(r'[^a-z0-9]', '_', name.lower())
    return f"dirto_rag_{sanitized[:60]}"


def resolve_project() -> tuple[str, str, str]:
    """
    Auto-detect project from current environment.

    Returns: (table_name, watch_dir, project_name)
    """
    default_table = os.environ.get("DIRTORAG_TABLE", "")
    watch_dir_env = os.environ.get("DIRTORAG_WATCH_DIR", "")

    # 1. Explicit override from argv
    table = None
    watch_dir = None

    if len(sys.argv) > 1:
        table = sys.argv[1]
    elif default_table and default_table != "documents":
        table = default_table

    if len(sys.argv) > 2:
        watch_dir = sys.argv[2]
    elif watch_dir_env:
        watch_dir = watch_dir_env

    if table and watch_dir:
        project_name = Path(watch_dir).name
        return table, str(Path(watch_dir).resolve()), project_name

    cwd = Path.cwd()

    # 2. .dirtoRAG.yaml marker file — walk up from CWD
    current = cwd
    while current != current.parent:
        marker = current / ".dirtoRAG.yaml"
        if marker.is_file():
            try:
                import yaml
                with marker.open() as f:
                    marker_cfg = yaml.safe_load(f) or {}
                marker_table = marker_cfg.get("table")
                marker_watch = marker_cfg.get("watch_dir", ".")
                marker_name = marker_cfg.get("project_name", current.name)
                resolved_watch = str((current / marker_watch).resolve())
                resolved_table = marker_table or _sanitize_name(current.name)
                logger.info("Found .dirtoRAG.yaml at %s", current)
                return resolved_table, resolved_watch, marker_name
            except Exception as e:
                logger.warning("Failed to read %s: %s", marker, e)
        current = current.parent

    # 3. Git root
    try:
        git_root = os.popen("git rev-parse --show-toplevel 2>/dev/null").read().strip()
        if git_root and os.path.isdir(git_root):
            git_path = Path(git_root)
            project_name = git_path.name
            resolved_table = table or _sanitize_name(project_name)
            resolved_watch = watch_dir or str(git_path)
            logger.info("Detected git repo at %s", git_root)
            return resolved_table, str(Path(resolved_watch).resolve()), project_name
    except Exception:
        pass

    # 4. CWD fallback
    project_name = cwd.name
    resolved_table = table or _sanitize_name(project_name)
    resolved_watch = watch_dir or str(cwd)
    logger.info("Using CWD fallback: %s", cwd)
    return resolved_table, str(Path(resolved_watch).resolve()), project_name


def create_marker(directory: str, table_name: str | None = None, project_name: str | None = None) -> Path:
    """Create a .dirtoRAG.yaml marker file in the given directory."""
    import yaml

    dir_path = Path(directory).resolve()
    marker_path = dir_path / ".dirtoRAG.yaml"

    name = project_name or dir_path.name
    table = table_name or _sanitize_name(name)

    marker_data = {
        "project_name": name,
        "watch_dir": ".",
        "table": table,
    }

    with marker_path.open("w") as f:
        yaml.dump(marker_data, f, default_flow_style=False)

    logger.info("Created marker file: %s", marker_path)
    return marker_path
