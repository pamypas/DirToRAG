import yaml
from pathlib import Path

def load_models_config(path: str = "models.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    api_base = data["api_base"].rstrip("/")
    api_key = data["api_key"]
    # по именам разделим для удобства
    llm_model = None
    embedding_model = None
    for m in data["models"]:
        name = m["name"]
        if "embedding" in name:
            embedding_model = name
        else:
            llm_model = name
    return {
        "api_base": api_base,
        "api_key": api_key,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
    }
