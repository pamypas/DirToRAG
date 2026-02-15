import os
import time
import json
from typing import List, Set, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import httpx

from embedder import get_embeddings
from models_loader import load_app_config
from chunker import chunk_text

# Supabase конфигурация
SUPABASE_URL = "http://192.168.1.169:8000/rest/v1/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNzY4Njc1OTU5LCJleHAiOjE5MjYzNTU5NTl9.6SWlDUqRqlMYooSNeJG9fI_UuT8LyFPYqfxbr5tZahE"
TABLE_NAME = "documents"

ALLOWED_EXT = {".pp", ".yaml", ".yml", ".erb", ".epp", ".md", ".txt"}

# размер батча для запросов к сервису эмбеддингов
EMBEDDING_BATCH_SIZE = 10

# количество параллельных запросов к embedding API
EMBEDDING_CONCURRENCY = 48

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


def insert_to_supabase(data: List[Dict[str, Any]]) -> None:
    """
    Вставляет данные в Supabase через REST API.
    """
    client = httpx.Client()
    try:
        # Отправляем данные по одному, так как Supabase может не поддерживать batch insert
        for record in data:
            # Преобразуем эмбеддинг в формат, который ожидает Supabase для типа vector(1024)
            if "embedding" in record and isinstance(record["embedding"], list):
                # Преобразуем список чисел в строку вида "[1.2, 3.4, 5.6]"
                embedding_str = "[" + ",".join(str(x) for x in record["embedding"]) + "]"
                record["embedding"] = embedding_str
                
            response = client.post(
                f"{SUPABASE_URL}{TABLE_NAME}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                },
                json=record
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"Ошибка при вставке данных: {e}")
        print(f"Ответ сервера: {e.response.text}")
        raise
    finally:
        client.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to local git repo")
    parser.add_argument("--dry-run", action="store_true", help="Выводить чанки и эмбеддинги в консоль без загрузки в БД")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    log_path = repo_path / INDEXED_LOG_FILENAME
    dry_run = args.dry_run

    # загружаем список уже проиндексированных файлов
    indexed_files_set = load_indexed_files(log_path)

    # сначала посчитаем общее количество файлов для индексации
    all_files = list(iter_files(repo_path))
    total_files = len(all_files)
    if total_files == 0:
        print("Нет файлов для индексации")
        return

    indexed_files = 0  # счётчик успешно проиндексированных файлов (в этом запуске)
    last_progress = -1  # чтобы не спамить одинаковыми значениями

    # Инициализируем пул потоков для параллельных запросов
    executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY)

    for fpath in all_files:
        rel_path = str(fpath.relative_to(repo_path))

        # пропускаем файлы, которые уже были успешно проиндексированы ранее
        # В режиме dry-run мы можем игнорировать этот лог, чтобы видеть вывод для всех файлов
        if not dry_run and rel_path in indexed_files_set:
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # не удалось прочитать файл - просто пропускаем
            continue

        # Используем импортированную функцию chunk_text из chunker.py
        chunks = chunk_text(text)
        
        if not chunks:
            # нечего индексировать
            if not dry_run:
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

        # если вообще не получили эмбеддингов - считаем, что файл не проиндексирован
        if not all_embs:
            continue

        # Режим dry-run: вывод в консоль
        if dry_run:
            print(f"\n{'='*20} FILE: {rel_path} {'='*20}")
            for i, (chunk, emb) in enumerate(zip(chunks, all_embs)):
                print(f"\n--- Chunk {i+1} ---")
                print("Content:")
                print(chunk)
                print("\nEmbedding (first 10 values):")
                print(json.dumps(emb[:10]))
                print(f"Vector size: {len(emb)}")
            continue

        # Подготавливаем данные для вставки в Supabase
        records_to_insert = []
        for chunk, emb in zip(chunks, all_embs):
            records_to_insert.append({
                "content": chunk,
                "embedding": emb,
                "metadata": {"path": rel_path}
            })

        # Отправляем данные в Supabase с ретраями
        max_retries = 3
        for attempt in range(max_retries):
            try:
                insert_to_supabase(records_to_insert)
                break
            except Exception as e:
                print(f"Ошибка при отправке данных в Supabase (попытка {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # пауза перед повторной попыткой
                else:
                    print("Не удалось отправить данные после нескольких попыток. Пропуск...")
                    continue

        # файл успешно проиндексирован - записываем в лог
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

    # Завершаем работу executor
    executor.shutdown(wait=True)

    print(
        f"Indexing finished, всего проиндексировано файлов в этом запуске: "
        f"{indexed_files} из {total_files}"
    )


if __name__ == "__main__":
    main()
