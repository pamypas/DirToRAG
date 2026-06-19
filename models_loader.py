import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Path to config.yaml — resolved relative to this file, not CWD
_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


def _env_or(value: Any, env_var: str, default: Any = None) -> Any:
    """Return env var if set, otherwise the value, otherwise default."""
    env_val = os.environ.get(env_var)
    if env_val is not None:
        return env_val
    if value is not None and value != "":
        return value
    return default


def load_app_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load application config from YAML file, with env var overrides.

    Config path priority:
    1. DIRTORAG_CONFIG env var
    2. `path` argument
    3. config.yaml next to this file (fallback)
    """
    env_config = os.environ.get("DIRTORAG_CONFIG")
    if env_config:
        config_path = Path(env_config)
    elif path is not None:
        config_path = Path(path)
    else:
        config_path = _DEFAULT_CONFIG_PATH

    data: Dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # --- LLM ---
    llm_cfg = data.get("llm") or {}
    llm_api_base = _env_or(llm_cfg.get("api_base"), "DIRTORAG_LLM_API_BASE")
    llm_api_key = _env_or(llm_cfg.get("api_key"), "DIRTORAG_LLM_API_KEY", "key")
    llm_model = _env_or(llm_cfg.get("model"), "DIRTORAG_LLM_MODEL")

    # --- Embedding ---
    emb_cfg = data.get("embedding") or {}
    emb_api_base = _env_or(emb_cfg.get("api_base"), "DIRTORAG_EMBEDDING_API_BASE")
    emb_api_key = _env_or(emb_cfg.get("api_key"), "DIRTORAG_EMBEDDING_API_KEY", "key")
    emb_model = _env_or(emb_cfg.get("model"), "DIRTORAG_EMBEDDING_MODEL")

    # --- Database ---
    db_cfg = data.get("database") or {}
    db_host = _env_or(db_cfg.get("host"), "DIRTORAG_DB_HOST", "localhost")
    db_port = int(_env_or(db_cfg.get("port"), "DIRTORAG_DB_PORT", "5432"))
    db_name = _env_or(db_cfg.get("name"), "DIRTORAG_DB_NAME", "dirtoRAG")
    db_user = _env_or(db_cfg.get("user"), "DIRTORAG_DB_USER", "postgres")
    db_password = _env_or(db_cfg.get("password"), "DIRTORAG_DB_PASSWORD", "")

    # --- LLM validation (skip if no LLM needed, e.g. MCP-only usage) ---
    if not llm_api_base or not llm_model:
        llm_cfg = data.get("llm") or {}
        llm_api_base = (llm_cfg.get("api_base") or "").rstrip("/")
        llm_model = llm_cfg.get("model") or None

    if not emb_api_base:
        raise ValueError(
            "Embedding 'api_base' must be set via config.yaml or DIRTORAG_EMBEDDING_API_BASE env var."
        )
    if not emb_model:
        raise ValueError(
            "Embedding 'model' must be set via config.yaml or DIRTORAG_EMBEDDING_MODEL env var."
        )

    # --- Chunker ---
    chunker_cfg = data.get("chunker") or {}
    max_chunk_size = int(_env_or(chunker_cfg.get("max_chunk_size"), "DIRTORAG_CHUNK_SIZE", "1024"))
    overlap = int(_env_or(chunker_cfg.get("overlap"), "DIRTORAG_CHUNK_OVERLAP", "256"))

    # --- Build result ---
    data["llm"] = {
        "api_base": (llm_api_base or "").rstrip("/"),
        "api_key": llm_api_key,
        "model": llm_model,
        "log_context": llm_cfg.get("log_context", True) if data.get("llm") else True,
        "search_limit": llm_cfg.get("search_limit", 70) if data.get("llm") else 70,
    }

    data["embedding"] = {
        "api_base": (emb_api_base or "").rstrip("/"),
        "api_key": emb_api_key,
        "model": emb_model,
        "batch_size": emb_cfg.get("batch_size", 10),
        "concurrency": emb_cfg.get("concurrency", 48),
    }

    data["database"] = {
        "host": db_host,
        "port": db_port,
        "name": db_name,
        "user": db_user,
        "password": db_password,
    }

    data["chunker"] = {
        "max_chunk_size": max_chunk_size,
        "overlap": overlap,
    }

    # Preserve server section
    if "server" not in data:
        data["server"] = {}

    # --- Dashboard ---
    dash_cfg = data.get("dashboard") or {}
    auto_open_raw = _env_or(dash_cfg.get("auto_open"), "DIRTORAG_DASHBOARD_AUTO_OPEN", "true")
    if isinstance(auto_open_raw, bool):
        auto_open_val = auto_open_raw
    else:
        auto_open_val = str(auto_open_raw).lower() in ("true", "1", "yes")
    data["dashboard"] = {
        "host": _env_or(dash_cfg.get("host"), "DIRTORAG_DASHBOARD_HOST", "127.0.0.1"),
        "port": int(_env_or(dash_cfg.get("port"), "DIRTORAG_DASHBOARD_PORT", "8889")),
        "auto_open": auto_open_val,
    }

    return data


def load_models_config(path: str | None = None) -> Dict[str, Any]:
    """Backward-compatible wrapper around load_app_config."""
    return load_app_config(path)
