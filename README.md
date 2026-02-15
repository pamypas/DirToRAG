# DirToRAG

Локальный RAG-сервис с OpenAI-совместимым API. Индексирует директории с кодом и отвечает на вопросы по содержимому.

Использует PostgreSQL с расширением pgvector для гибридного поиска (full-text + semantic).

---

## Что внутри

- **cli.py** — единая точка входа для всех операций
- **server.py** — FastAPI сервер с эндпоинтом `/v1/chat/completions`
- **embedder.py** — получение эмбеддингов через OpenAI-совместимый API
- **chunker.py** — разбиение текста на чанки с сохранением логических блоков
- **agents/pg_agent.py** — агент для гибридного поиска в PostgreSQL

---

## Быстрый старт

### 1. Подготовка окружения

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Запуск PostgreSQL

```bash
docker-compose up -d
```

Поднимет PostgreSQL с pgvector на `localhost:5432`, без пароля.

### 3. Конфигурация

Всё в одном файле `config.yaml`:

```yaml
llm:
  api_base: https://your-llm-api/v1
  api_key: your-key
  model: gpt-4
  log_context: True  # печатать контекст в логи

embedding:
  api_base: http://localhost:1235
  api_key: key
  model: text-embedding-3-small
  batch_size: 10     # чанков на запрос
  concurrency: 48    # параллельных запросов

database:
  host: localhost
  port: 5432
  name: dirtoRAG
  user: postgres
  password: ""

server:
  host: 0.0.0.0
  port: 8000
```

---

## Использование

Все операции через `cli.py`:

```bash
# Создать таблицу для проекта
python cli.py init my_project

# Проиндексировать директорию
python cli.py index my_project /path/to/repo

# Dry-run — посмотреть чанки без записи в БД
python cli.py index my_project /path/to/repo --dry-run

# Запустить сервер
python cli.py serve my_project
```

Можно создавать сколько угодно таблиц под разные проекты:

```bash
python cli.py init project_a
python cli.py index project_a ./project-a

python cli.py init project_b
python cli.py index project_b ./project-b

# Сервер ищет в указанной таблице
python cli.py serve project_a
```

---

## Как работает поиск

Гибридный поиск = full-text search + векторный поиск:

1. **Full-text** — PostgreSQL `tsvector` сGIN-индексом, находит документы по ключевым словам
2. **Semantic** — pgvector с HNSW-индексом, находит семантически похожие чанки
3. **RRF** (Reciprocal Rank Fusion) — объединяет результаты с весами

Функция `hybrid_search_<table>` создаётся автоматически при `init`.

---

## Запрос к серверу

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Как работает авторизация в этом проекте?"}
    ]
  }'
```

Сервер:
1. Получает эмбеддинг запроса
2. Ищет релевантные чанки в PostgreSQL
3. Добавляет их в контекст
4. Отправляет в LLM и возвращает ответ

---

## Поддерживаемые форматы файлов

По умолчанию: `.pp`, `.yaml`, `.yml`, `.erb`, `.epp`, `.md`, `.txt`

---

## Заметки

- Лог проиндексированных файлов — `.indexed_files.log` в корне директории
- При повторной индексации уже обработанные файлы пропускаются
- Прокси игнорируются (`trust_env=False`)
- `log_context: True` в конфиге — видеть что летит в LLM
