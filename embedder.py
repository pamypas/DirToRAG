import httpx
from models_loader import load_models_config

_cfg = load_models_config()

_client = httpx.Client(
    base_url=_cfg["api_base"],
    headers={"Authorization": f"Bearer {_cfg['api_key']}"},
    timeout=60.0,
    trust_env=False,   # <─ не читать HTTP(S)_PROXY, NO_PROXY и т.п.
)

EMBEDDING_MODEL = _cfg["embedding_model"]

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
