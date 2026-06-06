# DirToRAG

Локальный RAG-сервис с OpenAI-совместимым API и MCP-интеграцией. Индексирует директории с кодом и отвечает на вопросы по содержимому через гибридный поиск (full-text + semantic) в PostgreSQL с pgvector.

## Содержание

- [Быстрый старт](#быстрый-старт)
- [CLI Cheat Sheet](#cli-cheat-sheet)
- [Интеграция с Claude Code](#интеграция-с-claude-code)
- [Конфигурация](#конфигурация)
- [Как работает поиск](#как-работает-поиск)
- [Файлы](#файлы)

---

## Быстрый старт

```bash
# 1. Окружение
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Запустить PostgreSQL (должен быть доступен на localhost:5432)
docker-compose up -d

# 3. Поправить config.yaml — указать URL к embedding-модели и LLM

# 4. Создать таблицу и проиндексировать
python cli.py init work
python cli.py index work ~/Work

# 5. Запустить MCP-сервер (поиск + авто-переиндексация)
python mcp_search.py work ~/Work
```

---

## CLI Cheat Sheet

Все операции через `cli.py`:

```bash
# Инициализация — создать таблицу и SQL-функции
python cli.py init <table>

# Полная индексация директории (пропускает уже проиндексированные файлы)
python cli.py index <table> <directory>

# Инкрементальная индексация — только измененные/новые/удаленные файлы
python cli.py index <table> <directory> --incremental

# Dry-run — посмотреть чанки без записи в БД
python cli.py index <table> <directory> --dry-run

# Запустить HTTP-сервер (OpenAI-совместимый /v1/chat/completions)
python cli.py serve <table>

# MCP-сервер — поиск + файловый вотчер (замена serve для Claude Code)
python mcp_search.py <table> [watch_directory]

# MCP Inspector — отладка MCP-сервера в браузере
mcp dev mcp_search.py <table> [watch_directory]

# PreToolUse(Read) хук — ручной запуск для отладки
python find_related.py <путь_к_файлу> --table <table> --limit 5
```

Можно держать несколько таблиц под разные проекты:

```bash
python cli.py init work && python cli.py index work ~/Work
python cli.py init oss && python cli.py index oss ~/oss
```

### Инкрементальная индексация

Флаг `--incremental` проверяет mtime каждого файла и переиндексирует только изменившиеся. Файлы, которых больше нет на диске — удаляются из БД.

Формат `.indexed_files.log`:
```
relative/path/to/file.py	1735689600.123456
```

Первое поле — путь, второе — timestamp последней индексации. Совместим со старым форматом (только путь, без mtime).

---

## Интеграция с Claude Code

### MCP-сервер (явный поиск)

Добавить в `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "dirtoRAG": {
      "type": "stdio",
      "command": "/Users/an.belyaev/Tools/DirToRAG/venv/bin/python",
      "args": [
        "/Users/an.belyaev/Tools/DirToRAG/mcp_search.py",
        "work",
        "/Users/an.belyaev/Work"
      ],
      "env": {
        "DIRTORAG_TABLE": "work"
      }
    }
  }
}
```

После перезапуска Claude Code появятся инструменты:

- **`search_codebase(query, limit)`** — гибридный поиск по кодовой базе. Возвращает чанки с путями, контентом и скором.
- **`get_index_stats()`** — статистика: сколько чанков, файлов, какая embedding-модель.

Примеры запросов к Claude:
- «Найди в кодовой базе код, связанный с авторизацией»
- «Сколько файлов проиндексировано?»

### PreToolUse(Read) хук (прозрачный контекст)

При чтении файла хук автоматически находит семантически похожие чанки из других файлов и добавляет их в контекст. Пользователь ничего не делает — просто читает файл, а Claude видит связанный код.

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

`$CLAUDE_TOOL_INPUT` — встроенная переменная Claude Code, содержащая путь к файлу.

MCP-сервер и хук независимы — можно использовать что-то одно или оба вместе.

### Автоматическая переиндексация (файловый вотчер)

Если при запуске `mcp_search.py` передан watch-директорий (вторым аргументом или через `DIRTORAG_WATCH_DIR`), сервер запускает фоновый вотчер файловой системы. При любом изменении, создании или удалении файла индекс обновляется автоматически — не нужно вручную запускать `cli.py index --incremental`.

```
python mcp_search.py work ~/Work
```

Вотчер использует debounce 2 секунды, чтобы не переиндексировать файл на каждое сохранение.

### Полная конфигурация Claude Code (MCP + хук + вотчер)

```json
{
  "mcpServers": {
    "dirtoRAG": {
      "type": "stdio",
      "command": "/Users/an.belyaev/Tools/DirToRAG/venv/bin/python",
      "args": [
        "/Users/an.belyaev/Tools/DirToRAG/mcp_search.py",
        "work",
        "/Users/an.belyaev/Work"
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

---

## Конфигурация

Всё в `config.yaml`:

```yaml
llm:
  api_base: http://localhost:1234    # URL к LLM (для server.py)
  api_key: "sk-key"
  model: qwen3.6-27b
  log_context: True                  # печатать контекст в логи
  search_limit: 70                   # макс чанков в контексте

embedding:
  api_base: http://localhost:11434   # URL к embedding API (Ollama, LM Studio и т.д.)
  api_key: key
  model: bge-m3                      # embedding-модель
  batch_size: 10                     # чанков на запрос
  concurrency: 48                    # параллельных запросов

chunker:
  max_chunk_size: 1024
  overlap: 256

database:
  host: localhost
  port: 5432
  name: dirtoRAG
  user: postgres
  password: ""

server:
  host: 0.0.0.0
  port: 8888
```

MCP-сервер использует только секцию `embedding` — языковая модель для генерации ответов ему не нужна.

---

## Как работает поиск

Гибридный поиск (hybrid search) = full-text + semantic, объединенные через Reciprocal Rank Fusion:

1. **Full-text** — PostgreSQL `tsvector` с GIN-индексом, ищет документы по ключевым словам через `websearch_to_tsquery`
2. **Semantic** — pgvector с HNSW-индексом (inner product), находит семантически похожие чанки
3. **RRF** — объединяет результаты с настраиваемыми весами (`full_text_weight`, `semantic_weight`)

SQL-функция `hybrid_search_<table>()` создается автоматически при `python cli.py init <table>`.

---

## Файлы

| Файл | Назначение |
|---|---|
| `cli.py` | CLI: `init`, `index`, `serve` |
| `mcp_search.py` | MCP-сервер: `search_codebase`, `get_index_stats` + файловый вотчер |
| `find_related.py` | PreToolUse(Read) хук: поиск семантически похожих чанков для файла |
| `server.py` | FastAPI сервер с `/v1/chat/completions` |
| `agents/pg_agent.py` | `PostgresSearchAgent`: гибридный поиск (build_context + search_raw) |
| `embedder.py` | OpenAI-совместимый клиент для эмбеддингов |
| `chunker.py` | Разбиение текста на чанки с сохранением логических блоков |
| `models_loader.py` | Загрузчик `config.yaml` |
| `config.yaml` | Вся конфигурация |

### Поддерживаемые расширения

По умолчанию: `.pp`, `.yaml`, `.yml`, `.erb`, `.epp`, `.md`, `.txt`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.rb`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`, `.kt`, `.scala`, `.sh`, `.bash`, `.zsh`, `.fish`, `.sql`, `.json`, `.xml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.env`, `.css`, `.scss`, `.html`, `.vue`, `.svelte`, `.tf`, `.proto`

### Примечания

- Лог проиндексированных файлов — `.indexed_files.log` в корне индексируемой директории
- При повторной индексации уже обработанные файлы пропускаются (если не указан `--incremental`)
- Прокси игнорируются (`trust_env=False`)
- MCP-сервер логирует в stderr (stdout занят протоколом JSON-RPC)
- `find_related.py` пропускает бинарные файлы (ориентируется на расширение)
