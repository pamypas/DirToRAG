import yaml
from pathlib import Path
from typing import Any, Dict


def load_app_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Загружает общий конфиг приложения (LLM, embedding, агенты и т.п.).
    Ожидаемый формат см. в config.yaml.

    Структура (пример):

    llm:
      api_base: http://host:port
      api_key: key-for-llm
      model: some-llm-model-name

    embedding:
      api_base: http://host:port
      api_key: key-for-embedding
      model: some-embedding-model-name

    agents:
      - name: RepoSearchAgent
        module: agents.agent1
        enabled: true
        config:
          limit: 8
          qdrant_url: http://127.0.0.1:6333
          collection_name: repo_chunks
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Не найден файл конфигурации: {config_path}")

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
            "В секции 'llm' в config.yaml должны быть заданы 'api_base' и 'model'."
        )

    if not emb_api_base or not emb_model:
        raise ValueError(
            "В секции 'embedding' в config.yaml должны быть заданы 'api_base' и 'model'."
        )

    data["llm"] = {
        "api_base": llm_api_base,
        "api_key": llm_api_key,
        "model": llm_model,
    }
    data["embedding"] = {
        "api_base": emb_api_base,
        "api_key": emb_api_key,
        "model": emb_model,
    }

    return data


def load_models_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Обёртка над load_app_config для обратной совместимости.
    Старый код, который ожидает models.yaml, теперь может
    использовать тот же единый config.yaml.
    """
    return load_app_config(path)
