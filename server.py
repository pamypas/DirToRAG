import json
import os
import logging
from typing import List, Dict, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


async def call_llm(messages: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
    """Call LLM API."""
    try:
        resp = await llm_client.post(
            "/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": stream,
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


async def stream_llm(messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """Stream LLM API response."""
    async with llm_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
        },
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {data}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract system prompt from request (first message with role="system")
    system_prompt = "You are a code assistant."
    for m in messages:
        if m.get("role") == "system":
            system_prompt = m.get("content", system_prompt)
            break

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # If no user message - proxy to LLM directly
    if not user_msg:
        if stream:
            return StreamingResponse(
                stream_llm(messages),
                media_type="text/event-stream",
            )
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

    system_content = system_prompt
    if context_text:
        system_content += "\n\nRepository context:\n" + context_text

    new_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]

    # Add non-system messages from original request
    new_messages.extend([m for m in messages if m.get("role") != "system"])

    if stream:
        return StreamingResponse(
            stream_llm(new_messages),
            media_type="text/event-stream",
        )

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
