import os
import time
from typing import List, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embedder import get_embeddings
from models_loader import load_app_config

_cfg = load_app_config()
_agents_cfg = {a["name"]: a for a in _cfg.get("agents", [])}
_repo_agent_cfg = _agents_cfg.get("RepoSearchAgent", {}).get("config", {})

QDRANT_URL = _repo_agent_cfg.get("qdrant_url", "http://127.0.0.1:6333")
COLLECTION_NAME = _repo_agent_cfg.get("collection_name", "repo_chunks")

# простое разбиение на chunk'и по символам
CHUNK_SIZE = 250
CHUNK_OVERLAP = 200

ALLOWED_EXT = {".pp", ".yaml", ".yml", ".erb", ".epp", ".md", ".txt"}

# размер батча для запросов к сервису эмбеддингов
EMBEDDING_BATCH_SIZE = 16

# количество параллельных запросов к embedding API
EMBEDDING_CONCURRENCY = 12

# файл, в который пишем успешно проиндексированные файлы
INDEXED_LOG_FILENAME = ".indexed_files.log"


def iter_files(repo_path: Path) -> List[Path]:
    for root, dirs, files in os.walk(repo_path):
        # не заходить в директории, имя которых начинается с точку
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fname in files:
            # при этом сами файлы, начинающиеся с точки, мы тоже не индексируем
            if fname.startswith("."):
                continue
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


def load_indexed_files(log_path: Path) -> Set[str]:
    """
    Читает лог уже проиндексированных файлов.
    Возвращает множество относительных путей (строки).
    """
    if not log_path.exists():
        return set()
    indexed: Set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indexed.add(line)
    return indexed


def append_indexed_file(log_path: Path, rel_path: str) -> None:
    """
    Добавляет один относительный путь файла в лог.
    """
    with log_path.open("a", encoding="utf-8") as f:
        f.write(rel_path + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to local git repo")
    parser.add_argument("collection_name", help="Qdrant collection name to create or use")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    collection_name = args.collection_name or COLLECTION_NAME
    log_path = repo_path / INDEXED_LOG_FILENAME

    # загружаем список уже проиндексированных файлов
    indexed_files_set = load_indexed_files(log_path)

    # Используем тот же режим, что и в rag_proxy.py: HTTP, без gRPC
    # Увеличиваем таймаут до 60 секунд, чтобы избежать WriteTimeout при больших батчах
    client = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
        timeout=60.0,
    )

    # создаём коллекцию, если нет
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        # размер вектора возьмём после первого вызова get_embeddings
        # поэтому сначала получим фиктивный embedding
        dim = len(get_embeddings(["test"])[0])
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    # сначала посчитаем общее количество файлов для индексации
    all_files = list(iter_files(repo_path))
    total_files = len(all_files)
    if total_files == 0:
        print("Нет файлов для индексации")
        return

    points = []
    point_id = 1
    indexed_files = 0  # счётчик успешно проиндексированных файлов (в этом запуске)
    last_progress = -1  # чтобы не спамить одинаковыми значениями

    # Инициализируем пул потоков для параллельных запросов
    executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY)

    for fpath in all_files:
        rel_path = str(fpath.relative_to(repo_path))

        # пропускаем файлы, которые уже были успешно проиндексированы ранее
        if rel_path in indexed_files_set:
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # не удалось прочитать файл — просто пропускаем
            continue

        chunks = chunk_text(text)
        if not chunks:
            # нечего индексировать
            append_indexed_file(log_path, rel_path)
            indexed_files += 1
            continue

        # разбиваем чанки на батчи для параллельной обработки
        batches = [chunks[i : i + EMBEDDING_BATCH_SIZE] for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE)]
        
        # отправляем запросы параллельно
        futures = [executor.submit(get_embeddings, batch) for batch in batches]
        
        all_embs = []
        file_failed = False
        for future in futures:
            try:
                batch_embs = future.result()
                all_embs.extend(batch_embs)
            except Exception as e:
                # Если один из батчей упал с ошибкой (например, таймаут),
                # пропускаем весь файл, чтобы не записывать частичные данные.
                print(f"Ошибка при обработке файла {rel_path}: {e}")
                file_failed = True
                break
        
        if file_failed:
            continue

        # если по итогу эмбеддингов меньше, чем чанков, обрежем список чанков
        if len(all_embs) < len(chunks):
            chunks = chunks[: len(all_embs)]

        # если вообще не получили эмбеддингов — считаем, что файл не проиндексирован
        if not all_embs:
            continue

        for chunk, emb in zip(chunks, all_embs):
            meta = {
                "path": rel_path,
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={"text": chunk, **meta},
                )
            )
            point_id += 1

        # файл успешно проиндексирован — записываем в лог
        append_indexed_file(log_path, rel_path)
        indexed_files_set.add(rel_path)
        indexed_files += 1

        # считаем прогресс в процентах и выводим только при изменении
        progress = int(indexed_files * 100 / total_files)
        if progress != last_progress:
            print(
                f"Прогресс индексации: {progress}% "
                f"({indexed_files}/{total_files} файлов)"
            )
            last_progress = progress

        # по батчам, чтобы не жечь память
        if len(points) >= 500:
            # Пробуем отправить батч с повторными попытками при ошибках сети
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                    points = []
                    break
                except Exception as e:
                    print(f"Ошибка при отправке батча в Qdrant (попытка {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # пауза перед повторной попыткой
                    else:
                        print("Не удалось отправить батч после нескольких попыток. Пропуск...")
                        points = []  # очищаем, чтобы не зациклиться, но теряем эти точки

    # Завершаем работу executor
    executor.shutdown(wait=True)

    if points:
        # Финальная отправка оставшихся точек с ретраями
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                break
            except Exception as e:
                print(f"Ошибка при отправке финального батча (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    print(
        f"Indexing finished, всего проиндексировано файлов в этом запуске: "
        f"{indexed_files} из {total_files}"
    )


if __name__ == "__main__":
    main()
