import os
from typing import List
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embedder import get_embeddings

QDRANT_URL = "http://192.168.1.242:6333"
COLLECTION_NAME = "repo_chunks"

# простое разбиение на chunk'и по символам
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

ALLOWED_EXT = {".pp", ".yaml", ".yml", ".erb", ".epp", ".md", ".txt"}


def iter_files(repo_path: Path) -> List[Path]:
    for root, dirs, files in os.walk(repo_path):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() in ALLOWED_EXT:
                yield p


def chunk_text(text: str):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to local git repo")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()

    # Используем тот же режим, что и в rag_proxy.py: HTTP, без gRPC
    client = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
    )

    # создаём коллекцию, если нет
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        # размер вектора возьмём после первого вызова get_embeddings
        # поэтому сначала получим фиктивный embedding
        dim = len(get_embeddings(["test"])[0])
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    points = []
    point_id = 1

    for fpath in iter_files(repo_path):
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue

        embs = get_embeddings(chunks)
        for chunk, emb in zip(chunks, embs):
            meta = {
                "path": str(fpath.relative_to(repo_path)),
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={"text": chunk, **meta},
                )
            )
            point_id += 1

        # по батчам, чтобы не жечь память
        if len(points) >= 500:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    print("Indexing finished")


if __name__ == "__main__":
    main()
