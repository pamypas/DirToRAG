import re
from typing import List
from models_loader import load_app_config

# Загружаем конфигурацию чанкера
_cfg = load_app_config()
_chunker_cfg = _cfg.get("chunker", {})
MAX_CHUNK_SIZE = _chunker_cfg.get("max_chunk_size", 1024)
OVERLAP = _chunker_cfg.get("overlap", 256)


def chunk_text(text: str) -> List[str]:
    """
    Разбивает текст на чанки, стараясь сохранять логические блоки целостными.
    Алгоритм:
    1. Разбивает текст на логические блоки (class, define, node, if, case).
    2. Объединяет маленькие блоки в чанки размером до MAX_CHUNK_SIZE.
    3. Если блок слишком большой, разбивает его на части.
    4. Добавляет перекрытие (overlap) между чанками.
    """
    # Шаг 1: Получаем логические блоки
    raw_blocks = _split_into_logical_blocks(text)
    
    # Шаг 2: Объединяем блоки в чанки, соблюдая лимит размера
    merged_chunks = _merge_blocks_to_chunks(raw_blocks, MAX_CHUNK_SIZE)
    
    # Шаг 3: Разбиваем слишком большие чанки и добавляем перекрытие
    final_chunks = []
    for chunk in merged_chunks:
        if len(chunk) <= MAX_CHUNK_SIZE:
            final_chunks.append(chunk)
        else:
            # Если чанк все еще слишком большой (например, один огромный блок),
            # разбиваем его принудительно
            split_parts = _split_large_text(chunk, MAX_CHUNK_SIZE)
            final_chunks.extend(split_parts)
            
    # Шаг 4: Добавляем перекрытие (overlap) между соседними чанками
    if OVERLAP > 0 and len(final_chunks) > 1:
        final_chunks = _add_overlap(final_chunks, OVERLAP)
        
    return final_chunks


def _split_into_logical_blocks(text: str) -> List[str]:
    """
    Разбивает текст на логические блоки Puppet (class, define, node, if, case).
    """
    pattern = re.compile(r'^\s*(class|define|node|if|case)\b(.*)', re.MULTILINE)
    blocks = []
    last_end = 0
    n = len(text)
    
    for match in pattern.finditer(text):
        start = match.start()
        
        # Сохраняем "хвост" предыдущего блока (код между блоками)
        if start > last_end:
            snippet = text[last_end:start].strip()
            if snippet:
                blocks.append(snippet)
        
        # Находим конец текущего блока
        block_end = _find_block_end(text, start)
        
        if block_end != -1:
            block_content = text[start:block_end]
            blocks.append(block_content)
            last_end = block_end
        else:
            # Если скобка не найдена, берем остаток файла
            remainder = text[start:].strip()
            if remainder:
                blocks.append(remainder)
            last_end = n
            break
            
    # Добавляем финальный хвост
    if last_end < n:
        tail = text[last_end:].strip()
        if tail:
            blocks.append(tail)
            
    return blocks


def _merge_blocks_to_chunks(blocks: List[str], limit: int) -> List[str]:
    """
    Объединяет маленькие блоки в чанки, не превышающие limit.
    """
    chunks = []
    current_chunk = ""
    
    for block in blocks:
        # Если текущий блок сам по себе больше лимита, добавляем его как есть (или разбиваем позже)
        if len(block) > limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(block)
        else:
            # Проверяем, превысит ли добавление блока лимит
            if len(current_chunk) + len(block) + 1 > limit: # +1 за перенос строки
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = block
            else:
                if current_chunk:
                    current_chunk += "\n" + block
                else:
                    current_chunk = block
                    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks


def _split_large_text(text: str, limit: int) -> List[str]:
    """
    Разбивает большой текст на части, стараясь не разрывать строки.
    """
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + limit, n)
        if end < n:
            # Ищем ближайший перенос строки назад, чтобы не разрывать строку
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start:
                end = newline_pos + 1
        
        chunks.append(text[start:end].strip())
        start = end
        
    return chunks


def _add_overlap(chunks: List[str], overlap_size: int) -> List[str]:
    """
    Добавляет перекрытие к концу каждого чанка (кроме последнего)
    из начала следующего чанка.
    """
    if not chunks or len(chunks) == 1:
        return chunks
        
    overlapped_chunks = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        if i < len(chunks) - 1:
            next_chunk = chunks[i+1]
            # Берем хвост из следующего чанка
            overlap_part = next_chunk[:overlap_size]
            # Пытаемся найти конец предложения или строки для красивого обреза
            cut_pos = overlap_part.rfind('\n')
            if cut_pos == -1:
                cut_pos = overlap_part.rfind('.')
            
            if cut_pos != -1:
                overlap_part = overlap_part[:cut_pos+1]
                
            chunk = chunk + "\n... \n" + overlap_part
        overlapped_chunks.append(chunk)
        
    return overlapped_chunks


def _find_block_end(text: str, start_pos: int) -> int:
    """
    Находит позицию закрывающей скобки для блока, начинающегося в start_pos.
    """
    stack = []
    i = text.find('{', start_pos)
    
    if i == -1:
        return -1
    
    stack.append('{')
    i += 1
    
    while i < len(text):
        char = text[i]
        
        if char == '{':
            stack.append('{')
        elif char == '}':
            stack.pop()
            if not stack:
                return i + 1
        
        # Игнорируем скобки внутри строк
        if char == '"' or char == "'":
            quote_char = char
            i += 1
            while i < len(text) and text[i] != quote_char:
                if text[i] == '\\':
                    i += 1
                i += 1
        
        # Игнорируем комментарии
        if char == '#':
            while i < len(text) and text[i] != '\n':
                i += 1
                
        if char == '/' and i + 1 < len(text) and text[i+1] == '*':
            i += 2
            while i < len(text) and not (text[i] == '*' and i + 1 < len(text) and text[i+1] == '/'):
                i += 1
            i += 1
            
        i += 1
        
    return -1
