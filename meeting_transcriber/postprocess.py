# -*- coding: utf-8 -*-
"""
Постобработка транскрипции: удаление галлюцинаций Whisper.

Whisper часто галлюцинирует на тишине или шумах, добавляя:
- "Субтитры от Amara.org"
- "Продолжение следует"
- "Спасибо за просмотр"
- Повторяющиеся фразы
- И другие артефакты
"""

import re
from typing import List, Dict, Any, Optional

from .logging_setup import get_logger

logger = get_logger()


# === ПАТТЕРНЫ ГАЛЛЮЦИНАЦИЙ ===
# Регулярные выражения для обнаружения типичных галлюцинаций Whisper

HALLUCINATION_PATTERNS = [
    # Субтитры и титры
    r"субтитр\w*\s*(от|by|создан\w*|подготовлен\w*)?\s*(amara|youtube|google)?",
    r"subtitl\w*\s*(by|from)?\s*(amara|youtube)?",
    r"титр\w*\s*(от|by)?",
    
    # Окончания видео
    r"продолжение\s+следует",
    r"to\s+be\s+continued",
    r"конец\s*(фильма|серии|эпизода)?",
    r"the\s+end",
    
    # Призывы к действию (YouTube)
    r"подписыва\w+\s*(на)?\s*(канал)?",
    r"(ставь\w*|поставь\w*)\s*(лайк\w*|палец)",
    r"subscribe\s*(to)?\s*(the)?\s*(channel)?",
    r"like\s*(and)?\s*subscribe",
    r"не\s+забуд\w+\s+(подписаться|лайк)",
    
    # Благодарности
    r"спасибо\s+(за\s+)?(просмотр|внимание|подписку)",
    r"thanks?\s+(for\s+)?(watching|viewing)",
    r"thank\s+you\s+(for\s+)?(watching|viewing)",
    
    # Музыкальные вставки
    r"^\s*♪+\s*$",
    r"^\s*\[музыка\]\s*$",
    r"^\s*\[music\]\s*$",
    r"^\s*\(музыка\)\s*$",
    r"^\s*музыка\s*$",
    
    # Технические артефакты
    r"^\s*\.\.\.\s*$",
    r"^\s*…\s*$",
    r"www\.\w+\.\w+",  # URL-подобные строки
    
    # Пустые или бессмысленные фразы
    r"^\s*[.。,،、]+\s*$",
    r"^\s*\?\s*$",
    r"^\s*!\s*$",
]

# Компилируем паттерны для производительности
COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE) 
    for p in HALLUCINATION_PATTERNS
]


def is_hallucination(text: str) -> bool:
    """
    Проверяет, является ли текст галлюцинацией.
    
    Args:
        text: Текст для проверки
        
    Returns:
        True если текст похож на галлюцинацию
    """
    if not text or not text.strip():
        return True
    
    text_clean = text.strip()
    
    # Проверяем по паттернам
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text_clean):
            return True
    
    # Проверяем на повторяющиеся символы (например: "а а а а а")
    words = text_clean.split()
    if len(words) >= 3:
        unique_words = set(w.lower() for w in words)
        # Если все слова одинаковые
        if len(unique_words) == 1 and len(words) >= 3:
            return True
    
    return False


def is_repeated_segment(
    current_text: str, 
    previous_texts: List[str], 
    threshold: float = 0.9
) -> bool:
    """
    Проверяет, является ли сегмент повторением предыдущих.
    
    Args:
        current_text: Текущий текст
        previous_texts: Список предыдущих текстов (последние N)
        threshold: Порог схожести (0.0 - 1.0)
        
    Returns:
        True если текст является повторением
    """
    if not current_text or not previous_texts:
        return False
    
    current_clean = current_text.strip().lower()
    if len(current_clean) < 5:  # Слишком короткий для сравнения
        return False
    
    for prev in previous_texts:
        prev_clean = prev.strip().lower()
        if not prev_clean:
            continue
        
        # Точное совпадение
        if current_clean == prev_clean:
            return True
        
        # Частичное совпадение (один текст содержится в другом)
        if current_clean in prev_clean or prev_clean in current_clean:
            shorter = min(len(current_clean), len(prev_clean))
            longer = max(len(current_clean), len(prev_clean))
            if shorter / longer >= threshold:
                return True
    
    return False


def filter_hallucinations(
    segments: List[Dict[str, Any]],
    check_repeats: bool = True,
    repeat_window: int = 3
) -> List[Dict[str, Any]]:
    """
    Фильтрует галлюцинации из списка сегментов.
    
    Args:
        segments: Список сегментов транскрипции
        check_repeats: Проверять повторяющиеся сегменты
        repeat_window: Сколько предыдущих сегментов проверять на повторы
        
    Returns:
        Отфильтрованный список сегментов
    """
    if not segments:
        return segments
    
    filtered = []
    removed_count = 0
    previous_texts: List[str] = []
    
    for seg in segments:
        text = seg.get("text", "").strip()
        
        # Проверяем на галлюцинацию
        if is_hallucination(text):
            removed_count += 1
            logger.debug(f"Удалена галлюцинация: '{text[:50]}...'")
            continue
        
        # Проверяем на повторение
        if check_repeats and is_repeated_segment(text, previous_texts):
            removed_count += 1
            logger.debug(f"Удалён повтор: '{text[:50]}...'")
            continue
        
        # Сегмент прошёл фильтры
        filtered.append(seg)
        
        # Обновляем окно предыдущих текстов
        previous_texts.append(text)
        if len(previous_texts) > repeat_window:
            previous_texts.pop(0)
    
    if removed_count > 0:
        logger.info(f"🧹 Удалено галлюцинаций/повторов: {removed_count}")
    
    return filtered


def clean_text(text: str) -> str:
    """
    Очищает текст от артефактов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    if not text:
        return text
    
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем пробелы перед знаками препинания
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Удаляем повторяющиеся знаки препинания
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    # Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def postprocess_transcription(
    result: Dict[str, Any],
    filter_enabled: bool = True
) -> Dict[str, Any]:
    """
    Полная постобработка результата транскрипции.
    
    Args:
        result: Результат транскрипции (text, segments)
        filter_enabled: Включить фильтрацию галлюцинаций
        
    Returns:
        Обработанный результат
    """
    if not result:
        return result
    
    segments = result.get("segments", [])
    
    if filter_enabled and segments:
        # Фильтруем галлюцинации
        segments = filter_hallucinations(segments)
        
        # Очищаем текст в каждом сегменте
        for seg in segments:
            if "text" in seg:
                seg["text"] = clean_text(seg["text"])
        
        # Пересобираем полный текст
        full_text = " ".join(
            seg.get("text", "").strip() 
            for seg in segments 
            if seg.get("text", "").strip()
        )
        full_text = clean_text(full_text)
        
        result["segments"] = segments
        result["text"] = full_text
    
    return result

