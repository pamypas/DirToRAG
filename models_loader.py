import yaml
from pathlib import Path


def load_models_config(path: str = "models.yaml"):
    """
    Загружает конфиг моделей из models.yaml.

    Ожидаемый формат:

    llm:
      api_base: http://host:port
      api_key: key-for-llm
      model: some-llm-model-name

    embedding:
      api_base: http://host:port
      api_key: key-for-embedding
      model: some-embedding-model-name
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Не найден файл конфигурации моделей: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # --- LLM ---
    llm_cfg = data.get("llm") or {}
    llm_api_base = (llm_cfg.get("api_base") or "").rstrip("/")
    llm_api_key = llm_cfg.get("api_key") or ""
    llm_model = llm_cfg.get("model") or None

    # --- Embedding ---
    emb_cfg = data.get("embedding") or {}
    emb_api_base = (emb_cfg.get("api_base") or "").rstrip("/")
    emb_api_key = emb_cfg.get("api_key") or ""
    emb_model = emb_cfg.get("model") or None

    if not llm_api_base or not llm_model:
        raise ValueError(
            "В секции 'llm' в models.yaml должны быть заданы 'api_base' и 'model'."
        )

    if not emb_api_base or not emb_model:
        raise ValueError(
            "В секции 'embedding' в models.yaml должны быть заданы 'api_base' и 'model'."
        )

    return {
        "llm": {
            "api_base": llm_api_base,
            "api_key": llm_api_key,
            "model": llm_model,
        },
        "embedding": {
            "api_base": emb_api_base,
            "api_key": emb_api_key,
            "model": emb_model,
        },
    }
