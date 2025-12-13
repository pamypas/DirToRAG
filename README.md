# Local RAG Chat API

Этот проект поднимает локальный OpenAI‑совместимый эндпоинт `/v1/chat/completions`,
который отвечает на вопросы о содержимом локального каталога с текстовыми файлами,
используя Retrieval‑Augmented Generation (RAG) поверх Qdrant.

Текущая архитектура:

- Индексация файлов в Qdrant: `index_repo.py`
- Векторные эмбеддинги: `embedder.py`
- Сервер с системой агентов: `server.py`
  - `RepoSearchAgent` (`agents/agent1.py`) — ищет релевантные фрагменты в Qdrant
  - `ExampleAgent` (`agents/agent2.py`) — пример «пустого» агента, пока ничего не делает

---

## Требования

- Python 3.10+ (рекомендуется)
- Запущенный экземпляр **Qdrant** (по умолчанию в коде: `http://127.0.0.1:6333`)
- Доступ к **OpenAI‑совместимому API**
- Ваш API‑ключ для этого провайдера

---

## Установка

1. Клонируйте репозиторий и перейдите в него:

   ```bash
   git clone <your-repo-url> rag
   cd rag
   ```

2. Создайте и активируйте виртуальное окружение:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   # venv\Scripts\activate   # Windows PowerShell
   ```

3. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

4. Сконфигурируйте модели и API в `models.yaml` (можно взять за основу `models.yaml.example`):

   ```yaml
   llm:
     api_base: http://localhost:1234/
     api_key: YOUR_LLM_API_KEY
     model: deepseek-chat

   embedding:
     api_base: http://192.168.1.242:1234
     api_key: YOUR_EMBEDDING_API_KEY
     model: text-embedding-qwen3-embedding-0.6b
   ```

   - `llm` — конфиг для чат‑модели (используется сервером `server.py`).
   - `embedding` — конфиг для модели эмбеддингов (используется `embedder.py` и `index_repo.py`).
   - `api_base` — базовый URL OpenAI‑совместимого API (без `/v1`).
   - `api_key` — ваш секретный ключ (держите локально, не коммитьте).
   - `model` — имя модели у вашего провайдера.

---

## Индексация директории в Qdrant

1. Убедитесь, что Qdrant запущен и доступен по URL, указанному в:

   - `index_repo.py` → `QDRANT_URL`
   - `agents/agent1.py` → `QDRANT_URL`

   По умолчанию в коде:

   ```python
   QDRANT_URL = "http://127.0.0.1:6333"
   COLLECTION_NAME = "repo_chunks"
   ```

   При необходимости вы можете изменить `COLLECTION_NAME`, если хотите использовать другое имя коллекции.

2. Запустите индексатор, указав путь к директории с текстовыми файлами:

   ```bash
   python index_repo.py /path/to/your/text/directory
   ```

   Индексатор:

   - Рекурсивно обходит директорию.
   - Пропускает скрытые директории (начинающиеся с `.`) и скрытые файлы.
   - Индексирует файлы с расширениями: `.pp`, `.yaml`, `.yml`, `.erb`, `.epp`, `.md`, `.txt`.
   - Делит содержимое на пересекающиеся чанки по символам.
   - Вызывает внешний сервис эмбеддингов через `embedder.py`.
   - Записывает точки в Qdrant с payload:
     - `text` – текст чанка,
     - `path` – относительный путь к файлу.
   - Ведёт лог уже проиндексированных файлов в `.indexed_files.log` в корне указанного репо
     и при повторном запуске пропускает их.

   После завершения в Qdrant будет коллекция с чанками репозитория.

---

## Сервер с агентами (`server.py`)

Сервер реализует OpenAI‑совместимый эндпоинт `/v1/chat/completions` и использует
систему агентов для «умного» наполнения контекста перед вызовом LLM.

### Агенты

Список агентов инициализируется в `server.py`:

```python
from agents.agent1 import RepoSearchAgent
from agents.agent2 import ExampleAgent

agents = [
    RepoSearchAgent(),
    ExampleAgent(),  # пример "пустого" агента
]
```

Каждый агент должен реализовывать метод:

```python
def build_context(self, user_message: str) -> str:
    ...
```

Возвращаемая строка добавляется к системному контексту перед вызовом LLM.

#### RepoSearchAgent (`agents/agent1.py`)

- Проверяет наличие коллекции `repo_chunks` в Qdrant.
- Строит эмбеддинг пользовательского запроса через `get_embeddings`.
- Делает поиск по Qdrant (REST API) и возвращает текстовый контекст вида:

  ```text
  [DOC 1] file: path/to/file1
  <chunk text>

  [DOC 2] file: path/to/file2
  <chunk text>
  ```

Если коллекции нет или поиск/эмбеддинги падают с ошибкой, агент возвращает пустую строку.

#### ExampleAgent (`agents/agent2.py`)

Простейший пример агента:

```python
class ExampleAgent:
    def build_context(self, user_message: str) -> str:
        return ""
```

Сейчас он ничего не добавляет в контекст и нужен как шаблон для будущих агентов.

---

## Запуск сервера

Из корня репозитория:

```bash
python server.py
```

или эквивалентно через uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Сервер:

- Отключает системные HTTP(S)‑прокси для предсказуемого поведения.
- Загружает конфиг LLM из `models.yaml`.
- При каждом запросе:
  1. Находит последнее сообщение пользователя.
  2. Вызывает всех агентов (`RepoSearchAgent`, `ExampleAgent` и т.д.) и собирает их контекст.
  3. Формирует сообщения для LLM:
     - `system` с базовым `SYSTEM_PROMPT`,
     - опционально `system` с «Repository context: ...», если агенты вернули контекст,
     - далее оригинальные сообщения клиента.
  4. Проксирует запрос к внешнему LLM (`/v1/chat/completions`) и возвращает ответ в формате OpenAI.

Если пользовательского сообщения нет, запрос просто проксируется в LLM без добавления контекста.

---

## Пример запроса к серверу

После запуска `server.py`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Что делает этот репозиторий?"}
    ]
  }'
```

Сервис:

1. Возьмёт последнее `user`‑сообщение.
2. `RepoSearchAgent` найдёт релевантные чанки в Qdrant и сформирует контекст.
3. Сформирует системные сообщения с этим контекстом.
4. Вызовет внешний LLM и вернёт его ответ.

---

## Заметки и устранение неполадок

- Если Qdrant пуст или коллекция `repo_chunks` не существует:
  - `RepoSearchAgent` вернёт пустой контекст,
  - сервер всё равно ответит, но без RAG‑контекста.
- Убедитесь, что `QDRANT_URL` в `index_repo.py` и `agents/agent1.py`
  указывает на ваш запущенный Qdrant.
- Если вы меняете модель эмбеддингов в `models.yaml`, рекомендуется:
  - удалить/переименовать существующую коллекцию в Qdrant,
  - заново запустить `index_repo.py`, чтобы коллекция была создана с правильным размером вектора.
- Переменные окружения HTTP(S)_PROXY и NO_PROXY игнорируются в `embedder.py` и `server.py`
  (используется `trust_env=False`), чтобы избежать неожиданных прокси‑настроек.

--- 

## Расширение системы агентов

Чтобы добавить нового агента:

1. Создайте файл, например `agents/agentN.py`.
2. Реализуйте класс с методом `build_context(self, user_message: str) -> str`.
3. Подключите его в `server.py`:

   ```python
   from agents.agentN import MyNewAgent

   agents = [
       RepoSearchAgent(),
       ExampleAgent(),
       MyNewAgent(),
   ]
   ```

Агент может:

- читать файлы,
- ходить во внешние сервисы,
- использовать свои собственные кэши и т.п.

Главное — возвращать строку, которая будет добавлена к общему контексту перед вызовом LLM.
