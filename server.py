import os
import logging
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from models_loader import load_app_config
from agents.pg_agent import PostgresSearchAgent, set_search_table as _set_table

# use Uvicorn / FastAPI logger
logger = logging.getLogger("uvicorn.error")

# disable system proxies
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


def set_search_table(table_name: str) -> None:
    """
    Set the search table name for PostgresSearchAgent.
    This is called by CLI before starting the server.
    """
    _set_table(table_name)

# Load config
cfg = load_app_config()
llm_cfg = cfg["llm"]

llm_client = httpx.AsyncClient(
    base_url=llm_cfg["api_base"],
    headers={"Authorization": f"Bearer {llm_cfg['api_key']}"} if llm_cfg["api_key"] else {},
    timeout=120.0,
    trust_env=False,
)

LLM_MODEL = llm_cfg["model"]
LLM_DEBUG_CONTEXT = bool(llm_cfg.get("log_context", False))
SEARCH_LIMIT = int(llm_cfg.get("search_limit", 10))

app = FastAPI()

# Initialize search agent
search_agent = PostgresSearchAgent(config={"limit": SEARCH_LIMIT})

SYSTEM_PROMPT = (
    "You are a code assistant. Use ONLY the repository context below to answer. "
    "If the context clearly contains relevant information, DO NOT say that you have no information. "
    "If there is truly nothing relevant in the context, then say that."
)


async def call_llm(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call LLM API."""
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
        return {
            "error": {
                "type": "llm_connection_error",
                "message": f"Failed to connect to LLM backend: {e}",
            }
        }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # If no user message - proxy to LLM directly
    if not user_msg:
        resp = await call_llm(messages)
        return JSONResponse(resp)

    # Build context via search agent
    context_text = search_agent.build_context(user_msg)

    # Debug output
    if LLM_DEBUG_CONTEXT:
        if context_text:
            logger.info("LLM context for user query:\n%s", context_text)
        else:
            logger.info("LLM context is empty (no data found)")

    new_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if context_text:
        new_messages.append({
            "role": "system",
            "content": "Repository context:\n" + context_text,
        })

    new_messages.extend(messages)

    resp = await call_llm(new_messages)
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
    )
