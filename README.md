# Puppet Repo RAG Chat API

This service exposes an OpenAI-compatible `/v1/chat/completions` endpoint that answers questions about a local Puppet repository using RAG (Retrieval-Augmented Generation).

## Architecture

- **Qdrant** stores vector embeddings of repo chunks in collection `puppet_repo_chunks`.
- **index_repo.py**:
  - Walks a local repo, reads files with extensions: `.pp`, `.yaml`, `.yml`, `.erb`, `.epp`, `.md`, `.txt`.
  - Splits text into overlapping chunks (`CHUNK_SIZE = 1500`, `CHUNK_OVERLAP = 200`).
  - Uses `embedder.get_embeddings` to get embeddings from an OpenAI-compatible API.
  - Upserts points into Qdrant with payload: `{"text": chunk, "path": "<relative/path>"}`.
- **rag_proxy.py** (FastAPI):
  - Implements `POST /v1/chat/completions` (OpenAI-style).
  - For each request:
    1. Extracts the latest user message.
    2. Embeds it via `get_embeddings`.
    3. Searches Qdrant via REST: `POST /collections/{COLLECTION_NAME}/points/search`.
    4. Builds a “Repository context” system message from retrieved chunks.
    5. Calls the external LLM `/v1/chat/completions` and returns its response.

The system prompt forces the LLM to use only repository context and to avoid saying “no information” when relevant context exists.

## Configuration

### `models.yaml`

