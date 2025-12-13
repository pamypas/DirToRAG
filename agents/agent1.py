import logging
from typing import List, Dict

import httpx
from qdrant_client import QdrantClient

from embedder import get_embeddings

logger = logging.getLogger("uvicorn.error")

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "repo_chunks"


def collection_exists(client: QdrantClient, name: str) -> bool:
    """
    Проверка существования коллекции в Qdrant.
    """
    try:
        collections = client.get_collections().collections
    except Exception:
        return False
    return any(c.name == name for c in collections)


class RepoSearchAgent:
    """
    Агент, который наполняет контекст фрагментами из репозитория,
    найденными по векторному поиску в Qdrant.
    """

    def __init__(self, config: dict | None = None, limit: int | None = None):
        """
        config:
          limit: int
          qdrant_url: str
          collection_name: str
          timeout: float
        """
        config = config or {}

        # приоритет: явный аргумент > конфиг > дефолт
        self.limit = limit if limit is not None else config.get("limit", 8)

        qdrant_url = config.get("qdrant_url", QDRANT_URL)
        self.collection_name = config.get("collection_name", COLLECTION_NAME)
        timeout = config.get("timeout", 30.0)

        # prefer_grpc=False — используем HTTP-клиент
        self.qdrant = QdrantClient(
            url=qdrant_url,
            prefer_grpc=False,
        )
        # отдельный HTTP-клиент для REST API поиска
        self.http_client = httpx.Client(
            base_url=qdrant_url,
            timeout=timeout,
            trust_env=False,
        )

    def _qdrant_search_http(
        self,
        collection_name: str,
        vector: List[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[Dict]:
        """
        Поиск в Qdrant через REST API.
        Возвращает список хитов (каждый — dict с 'payload', 'score' и т.п.).
        """
        url = f"/collections/{collection_name}/points/search"
        body = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        resp = self.http_client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", [])

    def build_context(self, user_message: str) -> str:
        """
        Строит текстовый контекст для LLM на основе запроса пользователя.
        Возвращает строку (может быть пустой, если контекст не найден).
        """
        # Если коллекции нет — не добавляем контекст
        if not collection_exists(self.qdrant, self.collection_name):
            return ""

        # 1. Получаем эмбеддинг запроса
        try:
            emb = get_embeddings([user_message])[0]
        except Exception as e:
            logger.exception("Failed to get embeddings in RepoSearchAgent: %s", e)
            return ""

        # 2. Ищем в Qdrant
        try:
            search_res = self._qdrant_search_http(
                collection_name=self.collection_name,
                vector=emb,
                limit=self.limit,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("Qdrant search failed in RepoSearchAgent: %s", e)
            return ""

        # 3. Формируем текст контекста
        context_parts: List[str] = []
        for i, hit in enumerate(search_res, 1):
            payload = hit.get("payload") or {}
            text = payload.get("text", "")
            path = payload.get("path", "unknown")
            context_parts.append(f"[DOC {i}] file: {path}\n{text}\n")

        return "\n\n".join(context_parts)
