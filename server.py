import os
import logging
from typing import List, Dict, Any
from importlib import import_module

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from models_loader import load_app_config
from agents.agent1 import RepoSearchAgent
from agents.agent2 import ExampleAgent

# use Uvicorn / FastAPI logger
logger = logging.getLogger("uvicorn.error")

# отключаем системные прокси (как в rag_proxy.py)
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

# Загружаем единый конфиг приложения (LLM, embedding, агенты)
cfg = load_app_config()
llm_cfg = cfg["llm"]

llm_client = httpx.AsyncClient(
    base_url=llm_cfg["api_base"],
    headers={"Authorization": f"Bearer {llm_cfg['api_key']}"} if llm_cfg["api_key"] else {},
    timeout=120.0,
    trust_env=False,
)

LLM_MODEL = llm_cfg["model"]

app = FastAPI()

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
        return {
            "error": {
                "type": "llm_connection_error",
                "message": f"Failed to connect to LLM backend: {e}",
            }
        }


def init_agents(app_cfg: Dict[str, Any]) -> List[Any]:
    """
    Инициализирует агентов на основе секции `agents` в config.yaml.

    Формат секции agents:
      - name: RepoSearchAgent
        module: agents.agent1
        enabled: true
        config: {...}
    """
    agents_cfg = app_cfg.get("agents", [])
    result: List[Any] = []

    for a in agents_cfg:
        if not a.get("enabled", True):
            continue

        module_name = a.get("module")
        class_name = a.get("name")
        if not module_name or not class_name:
            logger.error("Некорректная запись агента в конфиге: %s", a)
            continue

        agent_conf = a.get("config", {}) or {}

        try:
            module = import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as e:
            logger.exception(
                "Не удалось импортировать агента %s из модуля %s: %s",
                class_name,
                module_name,
                e,
            )
            continue

        try:
            # по соглашению — конструктор принимает config: dict
            agent = cls(config=agent_conf)
        except TypeError:
            # на случай старых агентов без параметра config
            logger.warning(
                "Агент %s не принимает параметр config, инициализируем без него",
                class_name,
            )
            agent = cls()
        except Exception as e:
            logger.exception("Не удалось инициализировать агента %s: %s", class_name, e)
            continue

        result.append(agent)

    return result


# Инициализируем агентов, которые будут наполнять контекст
agents = init_agents(cfg)


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

    # Собираем контекст от всех агентов
    context_parts: List[str] = []
    for agent in agents:
        try:
            ctx = agent.build_context(user_msg)
            if ctx:
                context_parts.append(ctx)
        except Exception as e:
            logger.exception("Agent %s failed to build context: %s", agent.__class__.__name__, e)

    context_text = "\n\n".join(context_parts) if context_parts else ""

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
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
