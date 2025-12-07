#!../venv/bin/python3
import os
import logging
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from qdrant_client import QdrantClient

from models_loader import load_models_config
from embedder import get_embeddings

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "repo_chunks"

# use Uvicorn / FastAPI logger
logger = logging.getLogger("uvicorn.error")

# disable system proxies (as in index_repo.py)
for var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
):
    os.environ.pop(var, None)

# Загружаем раздельные конфиги для LLM и embedding
cfg = load_models_config()
llm_cfg = cfg["llm"]

llm_client = httpx.AsyncClient(
    base_url=llm_cfg["api_base"],
    headers={"Authorization": f"Bearer {llm_cfg['api_key']}"} if llm_cfg["api_key"] else {},
    timeout=120.0,
    trust_env=False,
)

LLM_MODEL = llm_cfg["model"]

app = FastAPI()

# prefer_grpc=False to use HTTP client
qdrant = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=False,
)

# plain HTTP client for Qdrant REST API search
qdrant_http_client = httpx.Client(
    base_url=QDRANT_URL,
    timeout=30.0,
    trust_env=False,
)

SYSTEM_PROMPT = (
    "You are a code assistant. Use ONLY the repository context below to answer. "
    "If the context clearly contains relevant information, DO NOT say that you have no information. "
    "If there is truly nothing relevant in the context, then say that."
)


async def call_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Вызов LLM. При ошибке подключения возвращаем контролируемый JSON,
    чтобы FastAPI не падал 500 с трейсбеком.
    """
    try:
        resp = await llm_client.post(
            "/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.exception("LLM request failed: %s", e)
        # Возвращаем "ответ модели" в формате, который клиент сможет обработать.
        return {
            "error": {
                "type": "llm_connection_error",
                "message": f"Failed to connect to LLM backend: {e}",
            }
        }


def collection_exists(client: QdrantClient, name: str) -> bool:
    """
    Check if a Qdrant collection exists.
    """
    try:
        collections = client.get_collections().collections
    except Exception:
        return False
    return any(c.name == name for c in collections)


def qdrant_search_http(
    collection_name: str,
    vector: List[float],
    limit: int = 8,
    with_payload: bool = True,
) -> List[Dict]:
    """
    Perform a search in Qdrant using the REST API directly.
    Returns a list of hit dicts (each has 'payload', 'score', etc.).
    """
    url = f"/collections/{collection_name}/points/search"
    body = {
        "vector": vector,
        "limit": limit,
        "with_payload": with_payload,
    }
    resp = qdrant_http_client.post(url, json=body)
    resp.raise_for_status()
    data = resp.json()
    # Qdrant REST returns {"result": [ ... ], "time": ...}
    return data.get("result", [])


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # If there is no user message, just proxy to LLM
    if not user_msg:
        resp = await call_llm(messages)
        return JSONResponse(resp)

    # If collection does not exist, answer without context
    if not collection_exists(qdrant, COLLECTION_NAME):
        new_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + messages
        resp = await call_llm(new_messages)
        return JSONResponse(resp)

    # 1. search in Qdrant
    emb = get_embeddings([user_msg])[0]

    try:
        # Use direct REST API search to avoid client-version differences
        search_res = qdrant_search_http(
            collection_name=COLLECTION_NAME,
            vector=emb,
            limit=8,
            with_payload=True,
        )
    except Exception as e:
        # If Qdrant search fails, better answer without context than return 500
        logger.exception("Qdrant search failed, answering without RAG: %s", e)
        new_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + messages
        resp = await call_llm(new_messages)
        return JSONResponse(resp)

    context_parts = []
    for i, hit in enumerate(search_res, 1):
        payload = hit.get("payload") or {}
        text = payload.get("text", "")
        path = payload.get("path", "unknown")
        context_parts.append(f"[DOC {i}] file: {path}\n{text}\n")

    context_text = "\n\n".join(context_parts)

    new_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Repository context:\n" + context_text,
        },
    ] + messages

    resp = await call_llm(new_messages)
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag_proxy:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
