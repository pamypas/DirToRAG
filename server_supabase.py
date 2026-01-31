import os
import logging
from typing import List, Dict, Any
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from supabase import create_client, Client

from models_loader import load_app_config

# Use Uvicorn / FastAPI logger
logger = logging.getLogger("uvicorn.error")

# Отключаем системные прокси
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

# Конфигурация Supabase
SUPABASE_URL = "http://192.168.1.169:8000"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNzY4Njc1OTU5LCJleHAiOjE5MjYzNTU5NTl9.6SWlDUqRqlMYooSNeJG9fI_UuT8LyFPYqfxbr5tZahE"
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

cfg = load_app_config()
llm_cfg = cfg.get("llm", {})
emb_cfg = cfg.get("embedding", {})

LLM_API_BASE = llm_cfg.get("api_base", "http://192.168.1.169:11434/api")
LLM_MODEL = llm_cfg.get("model", "mistral")
LLM_DEBUG_CONTEXT = bool(llm_cfg.get("log_context", False))

EMBEDDING_API_BASE = emb_cfg.get("api_base", LLM_API_BASE)
EMBEDDING_MODEL = emb_cfg.get("model", LLM_MODEL)

app = FastAPI()

SYSTEM_PROMPT = (
    "You are a code assistant. Use ONLY the repository context below to answer. "
    "If the context clearly contains relevant information, DO NOT say that you have no information. "
    "If there is truly nothing relevant in the context, then say that."
)


async def call_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Вызов LLM через Ollama. При ошибке подключения возвращаем контролируемый JSON.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": False
                },
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Возвращаем ответ в формате OpenAI API
            return {
                "id": data.get("id", "chatcmpl-123"),
                "object": "chat.completion",
                "created": int(data.get("created", 1234567890)),
                "model": data.get("model", LLM_MODEL),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0)
                }
            }
    except httpx.HTTPError as e:
        logger.exception("LLM request failed: %s", e)
        return {
            "error": {
                "type": "llm_connection_error",
                "message": f"Failed to connect to LLM backend: {e}",
            }
        }


def format_search_results(results: List[Dict]) -> str:
    """Форматирует результаты поиска в строку контекста для LLM."""
    context_parts = []
    for result in results:
        metadata = result.get("metadata", {})
        content = result.get("content", "")
        if content and metadata:
            context_parts.append(
                f"From file: {metadata.get('path', 'unknown')}\n"
                f"Content: {content}\n"
                "-------------------"
            )
    return "\n".join(context_parts)


async def hybrid_search(query: str, match_count: int = 10) -> List[Dict]:
    """
    Выполняет гибридный поиск в Supabase с использованием FTS и векторного поиска
    """
    try:
        # Сначала получаем эмбеддинг для пользовательского запроса
        async with httpx.AsyncClient(timeout=120.0) as client:
            embedding_response = await client.post(
                f"{EMBEDDING_API_BASE}/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": [query]
                }
            )
            embedding_response.raise_for_status()
            embedding_data = embedding_response.json()
        
        if not embedding_data:
            logger.error("No embedding data received")
            return []
            
        # Извлекаем массив чисел из вложенной структуры ответа
        # Ожидаемый формат: [{"index": 0, "embedding": [[...]]}]
        if isinstance(embedding_data, list) and len(embedding_data) > 0 and "embedding" in embedding_data[0]:
            # Некоторые API возвращают вложенный список [[...]]
            nested_embedding = embedding_data[0]["embedding"]
            if isinstance(nested_embedding, list) and len(nested_embedding) > 0:
                # Берём первый элемент, если это вложенный список
                if isinstance(nested_embedding[0], list):
                    query_embedding = nested_embedding[0]
                else:
                    query_embedding = nested_embedding
            else:
                query_embedding = nested_embedding
        else:
            # Если формат ответа неожиданный, используем старую логику как запасной вариант
            query_embedding = embedding_data[0]
        
        # Убедимся, что embedding — это плоский список чисел
        if isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        
        # Проверяем размерность эмбеддинга
        if len(query_embedding) != 1024:
            logger.warning(f"Embedding size {len(query_embedding)} doesn't match expected 1024")
        
        # Выполняем гибридный поиск через Supabase REST API
        # Передаём embedding как список чисел, который Supabase преобразует в vector(512)
        try:
            response = supabase_client.rpc(
                "hybrid_search",
                {
                    "query_text": query,
                    "query_embedding": query_embedding,  # передаём как список чисел
                    "match_count": match_count
                }
            ).execute()
            
            return response.data
        except Exception as rpc_error:
            logger.error(f"RPC call failed: {rpc_error}")
            # Возвращаем пустой результат
            return []
        
    except Exception as e:
        logger.exception("Hybrid search failed: %s", e)
        return []


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # Если нет пользовательского сообщения — просто проксируем в LLM
    if not user_msg:
        resp = await call_llm(messages)
        return JSONResponse(resp)

    # Выполняем гибридный поиск
    search_results = await hybrid_search(user_msg)
    
    # Форматируем результаты поиска в контекст для LLM
    context_text = format_search_results(search_results) if search_results else ""

    # --- отладочный вывод контекста, добавляемого к запросу в модель ---
    if LLM_DEBUG_CONTEXT:
        if context_text:
            logger.info(
                "LLM контекст, собранный server.py для запроса пользователя:\n%s",
                context_text,
            )
        else:
            logger.info("LLM контекст, собранный server.py: пустой (данных из Supabase нет)")

    # Формируем сообщение для LLM с контекстом
    new_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if context_text:
        new_messages.append(
            {
                "role": "system",
                "content": "Repository context:\n" + context_text,
            }
        )

    new_messages.extend(messages)

    resp = await call_llm(new_messages)
    logger.info(f"LLM response: {resp}")
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_supabase:app", 
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug"
    )
