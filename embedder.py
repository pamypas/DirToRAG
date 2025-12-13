import httpx
from models_loader import load_app_config

_cfg = load_app_config()
_emb_cfg = _cfg["embedding"]

_client = httpx.Client(
    base_url=_emb_cfg["api_base"],
    headers={"Authorization": f"Bearer {_emb_cfg['api_key']}"} if _emb_cfg["api_key"] else {},
    timeout=300.0,  # 5 минут
    trust_env=False,   # <─ не читать HTTP(S)_PROXY, NO_PROXY и т.п.
)

EMBEDDING_MODEL = _emb_cfg["model"]


def get_embeddings(texts):
    """
    texts: list[str]
    return: list[list[float]]
    """
    resp = _client.post(
        "/v1/embeddings",
        json={"model": EMBEDDING_MODEL, "input": texts},
    )
    resp.raise_for_status()
    data = resp.json()
    # ожидается формат openai embeddings
    embs = [item["embedding"] for item in data["data"]]
    return embs
