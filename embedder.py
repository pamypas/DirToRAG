import httpx
from typing import List, Dict, Any

from models_loader import load_app_config

_cfg = load_app_config()
_emb_cfg = _cfg["embedding"]

EMBEDDING_MODEL = _emb_cfg["model"]

# Создаем новый клиент для каждого вызова, чтобы избежать проблем с потокобезопасностью
# при использовании ThreadPoolExecutor в index_repo.py
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Получает эмбеддинги для списка текстов.
    Создает новый экземпляр клиента для каждого вызова для безопасности в многопоточной среде.
    """
    client = httpx.Client(
        base_url=_emb_cfg["api_base"],
        headers={"Authorization": f"Bearer {_emb_cfg['api_key']}"} if _emb_cfg["api_key"] else {},
        timeout=300.0,  # 5 минут
        trust_env=False,   # <─ не читать HTTP(S)_PROXY, NO_PROXY и т.п.
    )
    
    try:
        resp = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]
    finally:
        client.close()
