# Инструкция: добавление MCP-сервера, PreToolUse-хука и инкрементальной переиндексации в DirToRAG

## Что уже есть

DirToRAG — рабочий RAG-сервер с тремя слоями:

1. **Индексатор** (`cli.py`) — обходит директорию, чанкает файлы, получает эмбеддинги через Ollama API, пишет в PostgreSQL с pgvector. Пропускает уже проиндексированные файлы (сверяясь с `.indexed_files.log`), но **не умеет обнаруживать измененные файлы и удалять старые чанки**.

2. **Поиск** (`agents/pg_agent.py` → `PostgresSearchAgent.build_context()`) — получает эмбеддинг запроса, вызывает SQL-функцию `hybrid_search_{table}()` (RRF: full-text + semantic), форматирует результат в строку для LLM.

3. **Сервер** (`server.py`) — FastAPI с единственным эндпоинтом `/v1/chat/completions`, который берет запрос, получает контекст через `PostgresSearchAgent`, добавляет в system prompt и проксирует в LLM.

**Ключевая функция, которую нужно переиспользовать:** `PostgresSearchAgent.build_context(user_message)` — она делает ровно то, что нужно MCP-серверу: запрос → эмбеддинг → гибридный поиск → чанки. Только возвращает она строку, а для MCP-инструмента лучше возвращать структурированные данные (список чанков с метаданными).

## Архитектура: три компонента

После реализации DirToRAG будет предоставлять три способа взаимодействия:

```
┌─────────────────────────────────────────────────────────────┐
│                       Claude Code                           │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐     │
│  │ MCP-инструмент    │  │ PreToolUse(Read) хук          │     │
│  │ search_codebase   │  │                                │     │
│  │ (явный поиск)     │  │ При чтении файла автоматически  │     │
│  │                   │  │ находит семантически похожие    │     │
│  │ Пользователь      │  │ чанки из других файлов и        │     │
│  │ явно вызывает     │  │ инжектит их в system prompt     │     │
│  │ поиск по кодовой  │  │ (прозрачно, без действий        │     │
│  │ базе              │  │ пользователя)                   │     │
│  └────────┬─────────┘  └──────────────┬───────────────────┘     │
│           │                           │                         │
└───────────┼───────────────────────────┼─────────────────────────┘
            │                           │
            ▼                           ▼
   ┌────────────────┐         ┌─────────────────────┐
   │  mcp_search.py │         │  find_related.py     │
   │  (stdio MCP)   │         │  (вызывается хуком)  │
   │                │         │                      │
   │  search_code-  │         │  Принимает путь к     │
   │  base(query)   │         │  файлу, возвращает    │
   │                │         │  релевантные чанки    │
   │  get_index_    │         │  из других файлов     │
   │  stats()       │         │                      │
   └───────┬────────┘         └──────────┬───────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  PostgresSearchAgent  │
          │  .search_raw()        │
          │  ._get_embedding()    │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  PostgreSQL + pgvector│
          │  hybrid_search_*()    │
          └───────────────────────┘
```

1. **MCP-сервер `search_codebase`** — явный поиск: пользователь просит Claude найти что-то в кодовой базе, Claude вызывает инструмент
2. **PreToolUse(Read) хук** — прозрачный контекст: когда Claude читает файл, хук автоматически находит семантически связанные чанки из других файлов и инжектит их в system prompt. Это аналог claude-mem подхода, но для статической кодовой базы
3. **Инкрементальная переиндексация** — поддержание индекса в актуальном состоянии

### Почему НЕ нужен SessionStart хук

SessionStart в claude-mem инжектит контекст **в начале сессии** — это имеет смысл для динамической памяти (что происходило в прошлых сессиях), потому что контекст нужно загрузить один раз при старте. Для статической кодовой базы такой подход бесполезен:

- Кодовая база не меняется между сессиями (в отличие от истории действий)
- Инжектить все релевантные чанки на старте нельзя — неизвестно, с какими файлами будет работать пользователь
- PreToolUse(Read) предоставляет контекст **в момент чтения файла** — именно тогда, когда он нужен

---

## ДЕТАЛЬНАЯ РЕАЛИЗАЦИЯ

### Шаг 1. Обновить `requirements.txt`

Добавить официальный Python SDK для MCP:

```
# Существующие:
httpx
psycopg
psycopg[binary]
fastapi
uvicorn
pyyaml

# Добавить:
mcp[cli]>=1.0.0
```

`mcp[cli]` устанавливает пакет `mcp` с дополнительным модулем `cli` для запуска dev-сервера.

После изменения файла выполнить:

```bash
cd ~/Tools/DirToRAG
source venv/bin/activate
pip install -r requirements.txt
```

### Шаг 2. Добавить метод `search_raw()` в `PostgresSearchAgent`

В файле `agents/pg_agent.py` добавить метод, который возвращает сырые результаты поиска (не строку для LLM). Это общий код, используемый и MCP-сервером, и PreToolUse-хуком:

```python
def search_raw(self, user_message: str, limit: int | None = None) -> list[dict]:
    """
    Execute hybrid search and return raw results.
    Unlike build_context(), returns structured data.
    """
    try:
        query_embedding = self._get_embedding(user_message)
        if not query_embedding:
            logger.warning("Empty embedding received")
            return []

        if len(query_embedding) != 1024:
            logger.warning(f"Embedding size {len(query_embedding)} != 1024")

        table_name = self._get_table_name()
        func_name = f"hybrid_search_{table_name}"
        conn_str = get_db_connection_string()
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        search_limit = limit if limit is not None else self.limit

        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(conn_str, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM {func_name}(%s, %s::extensions.vector, %s, %s, %s)",
                    (
                        user_message,
                        embedding_str,
                        search_limit,
                        self.full_text_weight,
                        self.semantic_weight,
                    ),
                )
                return cur.fetchall()

    except Exception as e:
        logger.exception("PostgresSearchAgent.search_raw failed: %s", e)
        return []
```

### Шаг 3. Создать `mcp_search.py` — MCP-сервер

Создать файл `~/Tools/DirToRAG/mcp_search.py`. Это stdio MCP-сервер, который предоставляет инструмент `search_codebase` для явного поиска:

```python
#!/usr/bin/env python3
"""
MCP-сервер для retrieval-only поиска по проиндексированной кодовой базе.

Запускается как stdio-процесс. Claude Code вызывает search_codebase,
сервер возвращает релевантные чанки без LLM-генерации.

Для отладки:
    python mcp_search.py          # запуск в stdio-режиме
    mcp dev mcp_search.py         # запуск с MCP Inspector (веб-интерфейс)
"""

import os
import sys
import logging
from typing import Any

# Отключаем системные прокси (как в server.py и cli.py)
for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

# Добавляем директорию проекта в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from agents.pg_agent import PostgresSearchAgent, set_search_table, get_search_table

# Настройка логирования — пишем в stderr (stdout занят под MCP-протокол)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_search")

# Имя таблицы по умолчанию (можно переопределить через set_search_table
# или аргумент командной строки)
DEFAULT_TABLE = os.environ.get("DIRTORAG_TABLE", "documents")

# Инициализируем MCP-сервер
mcp = FastMCP(
    name="DirToRAG Search",
    description="Hybrid codebase search (full-text + semantic) via DirToRAG",
)

# Глобальный search-агент (ленивая инициализация при первом вызове)
_search_agent: PostgresSearchAgent | None = None


def _get_agent() -> PostgresSearchAgent:
    """Лениво создает PostgresSearchAgent с настройками из config.yaml."""
    global _search_agent
    if _search_agent is None:
        table = get_search_table()
        if table == "documents":
            set_search_table(DEFAULT_TABLE)
            table = DEFAULT_TABLE
        logger.info(f"Initializing PostgresSearchAgent for table: {table}")
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
    from agents.pg_agent import get_db_connection_string

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

        from models_loader import load_app_config
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
    """Точка входа для MCP-сервера."""
    if len(sys.argv) > 1:
        table = sys.argv[1]
        set_search_table(table)
        logger.info(f"Table set from command line: {table}")
    else:
        set_search_table(DEFAULT_TABLE)
        logger.info(f"Using default table: {DEFAULT_TABLE}")

    logger.info("Starting DirToRAG MCP server (stdio mode)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**Ключевые моменты:**

1. **Логирование в stderr** — stdout занят под MCP-протокол (JSON-RPC).
2. **Ленивая инициализация агента** — `PostgresSearchAgent` создается при первом вызове.
3. **Переиспользование `search_raw()`** — никакого дублирования поисковой логики.
4. **Структурированный результат** — список словарей, а не строка для LLM.

### Шаг 4. Создать `find_related.py` — скрипт для PreToolUse(Read) хука

Создать файл `~/Tools/DirToRAG/find_related.py`. Этот скрипт вызывается Claude Code хуком **перед каждым вызовом Read**. Он получает путь к файлу, читает его содержимое, находит семантически похожие чанки из **других** файлов и возвращает их. Claude Code инжектит результат в system prompt через механизм `additionalContext`.

```python
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

# Disable system proxies
for var in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,  # Only warnings/errors to stderr
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("find_related")


def build_search_query(file_path: str, max_chars: int = 1500) -> str:
    """
    Read file and build a search query from its content.
    Uses the first chunk of the file content as the search query.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", file_path, e)
        return ""

    if not content.strip():
        return ""

    # Use the first ~1500 chars as the search query
    # This captures the file's purpose better than the whole file
    return content[:max_chars].strip()


def format_related_chunks(
    file_path: str,
    results: list[dict],
    max_results: int = 5,
) -> str:
    """
    Format search results as context for injection.
    Excludes the current file from results.
    """
    current_path = str(Path(file_path))

    # Extract relative path by comparing with known prefixes
    # (Claude Code passes absolute paths; DB stores relative paths)
    # The DB stores paths relative to the indexed root (e.g., ~/Work)
    # We use the filename as a fallback filter

    # Normalize: extract just the filename for filtering
    current_filename = Path(file_path).name

    # Filter out results from the same file
    filtered: list[dict] = []
    for r in results:
        result_path = r.get("file_path", "")
        # Skip if it's the same file (exact match or same absolute path)
        if result_path in current_path or current_path.endswith(result_path):
            continue
        # Also skip if filename matches (in case path formats differ)
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
        lines.append(content[:800])  # Truncate long chunks
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

    # Build search query from file content
    query = build_search_query(file_path)
    if not query:
        sys.exit(0)

    # Search for related chunks
    from agents.pg_agent import PostgresSearchAgent, set_search_table, get_search_table

    table = get_search_table()
    if table == "documents":
        set_search_table(args.table)

    # Create throwaway agent for this query
    agent = PostgresSearchAgent(
        config={"table_name": args.table, "limit": args.limit * 3}
    )

    results = agent.search_raw(query, limit=args.limit * 3)
    if not results:
        sys.exit(0)

    # Format and output to stdout
    context = format_related_chunks(file_path, results, max_results=args.limit)
    if context:
        print(context)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Ключевые моменты реализации `find_related.py`:**

1. **Быстрый поиск** — читает первые ~1500 символов файла, использует их как поисковый запрос для семантического поиска. Это быстрее, чем чанкать весь файл и искать по каждому чанку.

2. **Фильтрация своего же файла** — исключает из результатов чанки, принадлежащие тому же файлу, который читается. Иначе хук будет находить «похожий» код в самом себе.

3. **Форматирование для system prompt** — возвращает контекст в компактном формате с путями к файлам и сниппетами кода. Claude Code автоматически добавит это в system prompt через `additionalContext`.

4. **Тихий выход при ошибках** — если файл не читается, нет результатов или что-то пошло не так, скрипт молча завершается с exit code 0 (чтобы не мешать работе Read).

5. **Логирование в stderr** — stdout используется только для вывода контекста (который попадет в additionalContext).

### Шаг 5. Настроить PreToolUse(Read) хук в Claude Code

#### 5a. Конфигурация хука

Добавить в `~/.claude/settings.json` (или `~/.claude/settings.local.json`) секцию hooks:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "command": "python /Users/an.belyaev/Tools/DirToRAG/find_related.py \"$CLAUDE_TOOL_INPUT\" --table work --limit 5"
      }
    ]
  }
}
```

**Важно:** использовать абсолютные пути. Переменная `$CLAUDE_TOOL_INPUT` содержит аргументы, переданные инструменту Read (то есть путь к файлу).

#### 5b. Как это работает (механизм claude-mem)

Claude Code вызывает хук **перед** выполнением Read. Хук запускает `find_related.py`, передавая путь к файлу. Скрипт выводит в stdout отформатированный контекст. Claude Code берет stdout и инжектит его в system prompt следующего запроса к LLM.

**Формат контекста, который видит LLM:**

```
--- Related code (semantically similar to server.py) ---

### 1. Cloud/some-service/handler.py (score: 0.85)
```
def handle_request(req):
    ...
```

### 2. Other/utils/request_utils.py (score: 0.72)
```
def parse_request_body(body):
    ...
```

--- End of related code ---
```

Это **прозрачно** для пользователя — он просто читает файл, а Claude автоматически получает контекст о семантически связанных файлах и может предлагать более релевантные изменения.

#### 5c. Ограничения и оптимизации

- **Время ответа:** каждый вызов Read добавляет задержку на поиск (~100-300ms). Для больших файлов это приемлемо, для мелких — можно добавить фильтр по расширению в хуке.
- **Кэширование:** можно добавить TTL-кэш в `find_related.py` (ключ — file_path + mtime), чтобы не перезапрашивать для одного и того же файла в рамках одной сессии.
- **Пропуск бинарных файлов:** `find_related.py` должен пропускать бинарные файлы (не читать их содержимое). Можно добавить проверку расширения.

### Шаг 6. Инкрементальная переиндексация в `cli.py`

#### 6a. Изменить формат `.indexed_files.log`

**Было:** каждая строка — относительный путь к файлу:
```
relative/path/to/file.pp
```

**Стало:** каждая строка — путь + mtime через табуляцию:
```
relative/path/to/file.pp	1735689600.123456
```

#### 6b. Изменить функцию `load_indexed_files()`

Заменить текущую реализацию (строки 177-187 в `cli.py`):

```python
def load_indexed_files(log_path: Path) -> dict[str, float]:
    """
    Load indexed files log.
    Returns dict: {rel_path: mtime_timestamp}
    mtime=0 means "indexed before mtime tracking was added".
    """
    if not log_path.exists():
        return {}
    indexed: dict[str, float] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                indexed[parts[0]] = float(parts[1])
            else:
                # Старый формат (только путь, без mtime)
                indexed[parts[0]] = 0.0
    return indexed
```

#### 6c. Изменить функцию `append_indexed_file()`

Заменить текущую реализацию (строки 190-193 в `cli.py`):

```python
def append_indexed_file(log_path: Path, rel_path: str, mtime: float) -> None:
    """Append a file path with mtime to the indexed log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{rel_path}\t{mtime}\n")
```

#### 6d. Добавить функцию `delete_chunks_for_file()`

```python
def delete_chunks_for_file(
    conn_str: str, table_name: str, rel_path: str
) -> int:
    """
    Delete all chunks for a given file path.
    Returns number of deleted rows.
    """
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table_name} WHERE metadata->>'path' = %s",
                (rel_path,),
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted
```

#### 6e. Изменить функцию `index_directory()`

Заменить текущую реализацию (строки 218-343 в `cli.py`). Основные изменения:

```python
def index_directory(
    table_name: str,
    directory: str,
    dry_run: bool = False,
    incremental: bool = False,
) -> None:
    """
    Index a directory into the database table.

    Args:
        table_name: Name of the DB table
        directory: Path to directory to index
        dry_run: If True, print chunks without writing to DB
        incremental: If True, detect changed files (by mtime) and reindex only those.
                     Delete chunks for files that no longer exist on disk.
    """
    repo_path = Path(directory).resolve()
    if not repo_path.is_dir():
        logger.error(f"Directory not found: {repo_path}")
        sys.exit(1)

    log_path = repo_path / INDEXED_LOG_FILENAME
    conn_str = get_postgres_connection_string() if not dry_run else ""

    # Get embedding config
    batch_size, concurrency = get_embedding_config()

    # Load already indexed files (path -> mtime)
    indexed_files = load_indexed_files(log_path)

    # Collect all files in directory
    all_files = list(iter_files(repo_path))
    total_files = len(all_files)
    if total_files == 0:
        logger.info("No files to index")
        return

    # Build set of current file paths for deletion detection
    current_file_paths: set[str] = set()

    # Determine which files need indexing
    files_to_index: list[tuple[Path, str, bool]] = []
    # (absolute_path, relative_path, is_changed)

    for fpath in all_files:
        rel_path = str(fpath.relative_to(repo_path))
        current_file_paths.add(rel_path)

        current_mtime = fpath.stat().st_mtime

        if rel_path not in indexed_files:
            # New file
            files_to_index.append((fpath, rel_path, False))
        elif incremental:
            # Check if file changed (mtime differs)
            indexed_mtime = indexed_files[rel_path]
            if indexed_mtime == 0.0 or abs(current_mtime - indexed_mtime) > 1.0:
                # File changed (1 second tolerance for filesystem resolution)
                files_to_index.append((fpath, rel_path, True))
        # If not incremental, skip already-indexed files

    # Detect deleted files (only in incremental mode)
    deleted_files: list[str] = []
    if incremental and not dry_run:
        deleted_files = [
            path for path in indexed_files if path not in current_file_paths
        ]
        for rel_path in deleted_files:
            n = delete_chunks_for_file(conn_str, table_name, rel_path)
            logger.info(f"Deleted {n} chunks for removed file: {rel_path}")
        # Rewrite log without deleted files
        _rewrite_indexed_log(log_path, {
            k: v for k, v in indexed_files.items()
            if k in current_file_paths
        })

    if not files_to_index and not deleted_files:
        logger.info("No files to index (everything up to date)")
        return

    changed_count = sum(1 for _, _, changed in files_to_index if changed)
    new_count = len(files_to_index) - changed_count
    logger.info(
        f"Files to index: {len(files_to_index)} "
        f"({new_count} new, {changed_count} changed)"
    )

    # Thread pool for embedding requests
    executor = ThreadPoolExecutor(max_workers=concurrency)

    indexed_count = 0
    last_progress = -1

    for fpath, rel_path, is_changed in files_to_index:
        # Check for SIGINT
        if _interrupted:
            logger.info("Indexing interrupted by user")
            break

        # Delete old chunks if file changed
        if is_changed and not dry_run:
            n = delete_chunks_for_file(conn_str, table_name, rel_path)
            if n > 0:
                logger.debug(f"Deleted {n} old chunks for: {rel_path}")

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {rel_path}: {e}")
            continue

        chunks = chunk_text(text)

        if not chunks:
            if not dry_run:
                append_indexed_file(
                    log_path, rel_path, fpath.stat().st_mtime
                )
                indexed_count += 1
            continue

        # Split into batches for parallel embedding
        batches = [
            chunks[i : i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        # Send requests in parallel
        futures = [
            executor.submit(get_embeddings, batch) for batch in batches
        ]

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
            chunks = chunks[: len(all_embs)]

        if not all_embs:
            if not dry_run:
                append_indexed_file(
                    log_path, rel_path, fpath.stat().st_mtime
                )
                indexed_count += 1
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
                "metadata": {"path": rel_path},
            }
            for chunk, emb in zip(chunks, all_embs)
        ]

        # Insert with retries
        max_retries = 3
        inserted = False
        for attempt in range(max_retries):
            try:
                insert_to_postgres(records_to_insert, conn_str, table_name)
                inserted = True
                break
            except Exception as e:
                logger.error(
                    f"Insert error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)

        if not inserted:
            logger.error(
                f"Failed to insert {rel_path} after {max_retries} attempts"
            )
            continue

        # Mark as indexed with current mtime
        append_indexed_file(log_path, rel_path, fpath.stat().st_mtime)
        indexed_count += 1

        # Progress
        total = len(files_to_index)
        progress = int(indexed_count * 100 / total)
        if progress != last_progress:
            logger.info(
                f"Progress: {progress}% ({indexed_count}/{total} files)"
            )
            last_progress = progress

    executor.shutdown(wait=True)
    logger.info(
        f"Indexing finished: {indexed_count}/{len(files_to_index)} files processed"
    )
    if deleted_files:
        logger.info(f"Deleted chunks for {len(deleted_files)} removed files")


def _rewrite_indexed_log(log_path: Path, entries: dict[str, float]) -> None:
    """Rewrite the indexed files log with current entries."""
    tmp_path = log_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for path, mtime in sorted(entries.items()):
            f.write(f"{path}\t{mtime}\n")
    tmp_path.replace(log_path)
```

#### 6f. Добавить аргумент `--incremental` в CLI

В функции `main()`, в секции `index_parser` (строка 389), добавить:

```python
index_parser.add_argument(
    "--incremental",
    action="store_true",
    help="Detect changed files by mtime and reindex only those. "
         "Delete chunks for files that no longer exist.",
)
```

И в вызове `index_directory` (строка 403) передать этот флаг:

```python
elif args.command == "index":
    index_directory(
        args.table,
        args.directory,
        args.dry_run,
        incremental=args.incremental,
    )
```

#### 6g. Добавить обработку SIGINT

Добавить в начало `cli.py` после импортов:

```python
import signal

_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        logger.warning("Second interrupt received, forcing exit...")
        sys.exit(1)
    _interrupted = True
    logger.info(
        "Interrupt received. Finishing current file, then stopping..."
    )

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
```

И в цикле `for fpath, rel_path, is_changed in files_to_index:` добавить проверку (уже включена в код выше):

```python
if _interrupted:
    logger.info("Indexing interrupted by user")
    break
```

---

## Порядок действий (кратко)

1. `requirements.txt` — добавить `mcp[cli]>=1.0.0`, установить
2. `agents/pg_agent.py` — добавить метод `search_raw()`
3. `mcp_search.py` — создать новый файл (MCP-сервер с `search_codebase` и `get_index_stats`)
4. `find_related.py` — создать новый файл (PreToolUse(Read) хук для прозрачного контекста)
5. `~/.claude/settings.json` — добавить PreToolUse(Read) хук и MCP-сервер `dirtoRAG`
6. `cli.py` — изменить формат лога, добавить `--incremental`, `delete_chunks_for_file()`, обработку SIGINT
7. Удалить старый `.indexed_files.log` в ~/Work
8. `python cli.py init work && python cli.py index work ~/Work`
9. Перезапустить Claude Code, проверить через `/mcp`

### Настройка Claude Code (единая конфигурация)

В `~/.claude/settings.json` добавить и MCP-сервер, и хук:

```json
{
  "mcpServers": {
    "dirtoRAG": {
      "type": "stdio",
      "command": "/Users/an.belyaev/Tools/DirToRAG/venv/bin/python",
      "args": [
        "/Users/an.belyaev/Tools/DirToRAG/mcp_search.py",
        "work"
      ],
      "env": {
        "DIRTORAG_TABLE": "work"
      }
    }
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "command": "python /Users/an.belyaev/Tools/DirToRAG/find_related.py \"$CLAUDE_TOOL_INPUT\" --table work --limit 5"
      }
    ]
  }
}
```

**Важно:**
- Использовать полные пути, не `~/`
- `$CLAUDE_TOOL_INPUT` — встроенная переменная Claude Code, содержащая путь к файлу для Read
- Хук и MCP-сервер независимы — можно использовать что-то одно

---

## Тестирование

### Тест MCP-сервера

```bash
cd ~/Tools/DirToRAG
source venv/bin/activate

# Запустить инспектор
npx @modelcontextprotocol/inspector python mcp_search.py work
```

Откроется веб-интерфейс, где можно вызвать `search_codebase` и `get_index_stats`.

### Тест PreToolUse хука

```bash
cd ~/Tools/DirToRAG
source venv/bin/activate

# Протестировать find_related.py вручную
python find_related.py /Users/an.belyaev/Tools/DirToRAG/server.py --table work --limit 5
```

Должен вывести семантически похожие чанки из других файлов.

### Тест инкрементальной индексации

```bash
cd ~/Tools/DirToRAG
source venv/bin/activate

# Создать тестовый файл
echo "print('hello world')" > /tmp/test_incremental.py

# Проиндексировать
python cli.py index test /tmp --incremental

# Изменить файл и переиндексировать
echo "def new_function():\n    return 42" > /tmp/test_incremental.py
python cli.py index test /tmp --incremental

# Удалить файл и переиндексировать
rm /tmp/test_incremental.py
python cli.py index test /tmp --incremental
```

### Тест в Claude Code

После настройки спросить:
- "Сколько файлов проиндексировано в DirToRAG?" — должен вызвать `get_index_stats`
- "Найди в кодовой базе код, связанный с авторизацией" — должен вызвать `search_codebase`
- Прочитать любой файл из ~/Work — PreToolUse хук должен автоматически добавить контекст о похожих файлах

---

## Будущие улучшения (опционально)

1. **Файловый вотчер** — вместо ручного запуска `python cli.py index --incremental`, добавить `watchdog` для автоматической переиндексации при изменении файлов.

2. **Фильтр по типу файла** в `search_codebase` — параметр `file_types: list[str]` для ограничения поиска определенными расширениями.

3. **`/v1/search` в FastAPI** — если нужен HTTP-доступ к поиску (помимо MCP), добавить эндпоинт, использующий `search_raw()`.

4. **Кэширование в `find_related.py`** — TTL-кэш (ключ: file_path + mtime) для быстрых повторных запросов в рамках одной сессии.

5. **PreToolUse для Edit/Write** — помимо Read, можно добавить хук для Edit/Write, который находит код, связанный с редактируемым файлом, перед его изменением. Это поможет Claude предлагать консистентные изменения.

6. **Поддержка нескольких таблиц** — сейчас MCP-сервер работает с одной таблицей. Можно добавить `list_tables` и параметр `table` в `search_codebase`.
