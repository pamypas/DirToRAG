# Local RAG Chat API

This project exposes an OpenAI-compatible `/v1/chat/completions` endpoint that answers questions about the contents of a local directory with text files, using Retrieval-Augmented Generation (RAG) over Qdrant.

It is generic: you can point it at any directory with text-like files.

---

## Requirements

- Python 3.10+ (recommended)
- Running **Qdrant** instance (default URL in code: `http://192.168.1.242:6333`)
- Network access to an **OpenAI-compatible API** (e.g. proxyapi.ru)
- Your own API key for that provider (do not commit it)

---

## Installation

1. Clone the repository and enter it:

   ```bash
   git clone <your-repo-url> rag
   cd rag
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure models and API in `models.yaml`:

   ```yaml
   api_base: https://api.proxyapi.ru/openai
   api_key: YOUR_API_KEY_HERE        # put your real key here, do NOT commit it
   models:
     - name: gpt-4.1-nano-2025-04-14 # LLM model
     - name: text-embedding-3-small  # embedding model
   ```

   - `api_base` – base URL of the OpenAI-compatible API (without `/v1`).
   - `api_key` – your secret key (keep it local).
   - The first non-embedding model is used as the chat LLM, the first model with `"embedding"` in its name is used for embeddings.

---

## Indexing a directory into Qdrant

1. Make sure Qdrant is running and reachable at the URL configured in:

   - `index_repo.py` → `QDRANT_URL`
   - `rag_proxy.py` → `QDRANT_URL`

   Default in code:

   ```python
   QDRANT_URL = "http://192.168.1.242:6333"
   COLLECTION_NAME = "repo_chunks"
   ```

   You can change `COLLECTION_NAME` if you want a different collection name.

2. Run the indexer, pointing it at any directory with text files:

   ```bash
   python index_repo.py /path/to/your/text/directory
   ```

   The indexer:

   - Recursively walks the directory.
   - Reads files with extensions: `.pp`, `.yaml`, `.yml`, `.erb`, `.epp`, `.md`, `.txt`.
   - Splits content into overlapping chunks.
   - Calls the external embeddings API via `embedder.py`.
   - Upserts points into Qdrant with payload:
     - `text` – chunk text,
     - `path` – relative file path.

   After it finishes, Qdrant will contain the indexed chunks.

---

## Running the chat API

Start the FastAPI app:

```bash
uvicorn rag_proxy:app --host 0.0.0.0 --port 8000
```

By default it will listen on `http://0.0.0.0:8000` and expose an OpenAI‑compatible
`/v1/chat/completions` endpoint.

---

## Querying the API

You can call it with any OpenAI‑compatible client by pointing it to your local URL.
For example, using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1-nano-2025-04-14",
    "messages": [
      {"role": "user", "content": "What does this repository do?"}
    ]
  }'
```

The service will:

1. Embed the latest user message via the external embeddings API.
2. Search the `repo_chunks` collection in Qdrant for relevant chunks.
3. Build a system prompt with the retrieved context.
4. Call the upstream LLM (`api_base` / `/v1/chat/completions`) and return its response
   in the standard OpenAI format.

---

## Notes and troubleshooting

- If Qdrant is empty or the collection does not exist, the service will still answer,
  but without repository context.
- Make sure `QDRANT_URL` in both `index_repo.py` and `rag_proxy.py` matches your
  running Qdrant instance.
- If you change the embedding model in `models.yaml`, re‑run `index_repo.py` to
  recreate the collection with the correct vector size.

