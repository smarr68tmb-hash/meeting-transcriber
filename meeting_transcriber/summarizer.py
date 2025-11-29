# -*- coding: utf-8 -*-
"""
LLM суммаризация транскриптов через Groq API.

Возможности:
- Краткое саммари встречи
- Извлечение action items
- Анализ по спикерам (обязательства, вопросы)
- Ключевые решения и темы

Использование:
    export GROQ_API_KEY="gsk_xxx"
    meeting-transcriber transcribe file.wav --summarize
    meeting-transcriber transcribe file.wav --summarize --summary-lang en
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import Config
from .logging_setup import get_logger

logger = get_logger()

# Groq Chat API endpoint
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class SummarizerError(Exception):
    """Ошибка суммаризации."""
    pass


class MeetingSummarizer:
    """
    Суммаризатор встреч через Groq LLM API.
    
    Использует Llama 3 (или другие модели Groq) для:
    - Генерации краткого саммари
    - Извлечения action items
    - Анализа по спикерам
    """
    
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.model = Config.SUMMARIZER_MODEL
        self.timeout = Config.SUMMARIZER_TIMEOUT
        self.max_tokens = Config.SUMMARIZER_MAX_TOKENS
        
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY не установлен. Суммаризация недоступна. "
                "Получите ключ: https://console.groq.com"
            )
    
    def is_available(self) -> bool:
        """Проверить доступность LLM API."""
        return bool(self.api_key)
    
    def _call_llm(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.3
    ) -> str:
        """
        Вызов Groq LLM API.
        
        Args:
            system_prompt: Системный промпт
            user_prompt: Пользовательский промпт
            temperature: Температура генерации (0.0 - 1.0)
            
        Returns:
            Ответ модели
        """
        if not self.is_available():
            raise SummarizerError("GROQ_API_KEY не установлен")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        
        body = json.dumps(payload).encode('utf-8')
        
        request = Request(
            GROQ_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        
        try:
            response = urlopen(request, timeout=self.timeout)
            result = json.loads(response.read().decode('utf-8'))
            
            return result["choices"][0]["message"]["content"]
            
        except HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            
            if e.code == 429:
                raise SummarizerError(f"Превышен лимит Groq API. Попробуйте позже.")
            elif e.code == 401:
                raise SummarizerError("Неверный GROQ_API_KEY")
            else:
                logger.error(f"Groq LLM error {e.code}: {error_body}")
                raise SummarizerError(f"Ошибка API: {e.code}")
                
        except URLError as e:
            raise SummarizerError(f"Сетевая ошибка: {e}")
    
    def summarize(
        self,
        transcript: str,
        speakers: Optional[List[str]] = None,
        language: str = "ru",
        include_action_items: bool = True,
        include_speaker_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Суммаризировать транскрипт встречи.
        
        Args:
            transcript: Полный текст транскрипции
            speakers: Список спикеров (если есть диаризация)
            language: Язык вывода (ru/en)
            include_action_items: Извлекать action items
            include_speaker_analysis: Анализ по спикерам
            
        Returns:
            Словарь с summary, action_items, decisions, speaker_summary
        """
        if not transcript or not transcript.strip():
            raise SummarizerError("Пустой транскрипт")
        
        logger.info(f"🧠 Суммаризация транскрипта ({len(transcript)} символов)...")
        print(f"🧠 Суммаризация...")
        
        t0 = time.time()
        
        # Определяем язык промптов
        if language == "en":
            prompts = self._get_english_prompts()
        else:
            prompts = self._get_russian_prompts()
        
        result = {}
        
        # === 1. Основное саммари ===
        try:
            summary = self._generate_summary(transcript, prompts, speakers)
            result["summary"] = summary
        except Exception as e:
            logger.error(f"Ошибка генерации саммари: {e}")
            result["summary"] = None
            result["error"] = str(e)
        
        # === 2. Action Items ===
        if include_action_items and result.get("summary"):
            try:
                action_items = self._extract_action_items(transcript, prompts, speakers)
                result["action_items"] = action_items
            except Exception as e:
                logger.warning(f"Не удалось извлечь action items: {e}")
                result["action_items"] = []
        
        # === 3. Анализ по спикерам ===
        if include_speaker_analysis and speakers and len(speakers) > 1:
            try:
                speaker_analysis = self._analyze_speakers(transcript, prompts, speakers)
                result["speaker_analysis"] = speaker_analysis
            except Exception as e:
                logger.warning(f"Не удалось проанализировать спикеров: {e}")
                result["speaker_analysis"] = {}
        
        elapsed = time.time() - t0
        result["processing_time"] = elapsed
        result["model"] = self.model
        
        logger.info(f"✅ Суммаризация завершена за {elapsed:.1f} сек")
        print(f"✅ Суммаризация: {elapsed:.1f} сек")
        
        return result
    
    def _get_russian_prompts(self) -> Dict[str, str]:
        """Русские промпты для суммаризации."""
        return {
            "system_summary": """Ты — профессиональный аналитик деловых встреч.
Твоя задача — создавать чёткие, структурированные саммари.
Пиши кратко, по делу, выделяя главное.
Используй маркированные списки для ключевых пунктов.""",

            "user_summary": """Создай краткое саммари этой встречи.

Структура:
1. **Тема встречи** — одно предложение
2. **Ключевые обсуждения** — 3-5 пунктов
3. **Принятые решения** — что решили
4. **Следующие шаги** — что планируется

ТРАНСКРИПТ:
{transcript}""",

            "user_summary_with_speakers": """Создай краткое саммари этой встречи.
Участники: {speakers}

Структура:
1. **Тема встречи** — одно предложение
2. **Ключевые обсуждения** — 3-5 пунктов (кто что предлагал)
3. **Принятые решения** — что решили
4. **Следующие шаги** — что планируется

ТРАНСКРИПТ:
{transcript}""",

            "system_actions": """Ты — ассистент по извлечению задач из деловых встреч.
Извлекай конкретные, исполнимые action items.
Каждый item должен иметь: действие, ответственного (если указан), срок (если упомянут).
Отвечай ТОЛЬКО в JSON формате.""",

            "user_actions": """Извлеки action items (задачи к исполнению) из транскрипта.

Верни JSON массив в формате:
[
  {{"action": "описание задачи", "assignee": "кто делает или null", "deadline": "срок или null"}},
  ...
]

Если задач нет, верни пустой массив: []

ТРАНСКРИПТ:
{transcript}""",

            "system_speakers": """Ты — аналитик деловых коммуникаций.
Анализируй вклад каждого участника встречи.
Выделяй: позицию, обязательства, вопросы, предложения.""",

            "user_speakers": """Проанализируй вклад каждого спикера в встречу.
Спикеры: {speakers}

Для каждого спикера укажи:
- Основная позиция/роль на встрече
- Взятые обязательства
- Заданные вопросы
- Ключевые предложения

ТРАНСКРИПТ:
{transcript}"""
        }
    
    def _get_english_prompts(self) -> Dict[str, str]:
        """English prompts for summarization."""
        return {
            "system_summary": """You are a professional meeting analyst.
Your task is to create clear, structured meeting summaries.
Be concise, focus on key points.
Use bullet points for clarity.""",

            "user_summary": """Create a brief summary of this meeting.

Structure:
1. **Meeting Topic** — one sentence
2. **Key Discussions** — 3-5 points
3. **Decisions Made** — what was decided
4. **Next Steps** — what's planned

TRANSCRIPT:
{transcript}""",

            "user_summary_with_speakers": """Create a brief summary of this meeting.
Participants: {speakers}

Structure:
1. **Meeting Topic** — one sentence
2. **Key Discussions** — 3-5 points (who proposed what)
3. **Decisions Made** — what was decided
4. **Next Steps** — what's planned

TRANSCRIPT:
{transcript}""",

            "system_actions": """You are an assistant for extracting tasks from business meetings.
Extract specific, actionable items.
Each item should have: action, assignee (if mentioned), deadline (if mentioned).
Respond ONLY in JSON format.""",

            "user_actions": """Extract action items from the transcript.

Return a JSON array:
[
  {{"action": "task description", "assignee": "who or null", "deadline": "when or null"}},
  ...
]

If no tasks found, return empty array: []

TRANSCRIPT:
{transcript}""",

            "system_speakers": """You are a business communication analyst.
Analyze each participant's contribution to the meeting.
Highlight: position, commitments, questions, proposals.""",

            "user_speakers": """Analyze each speaker's contribution to the meeting.
Speakers: {speakers}

For each speaker indicate:
- Main position/role in the meeting
- Commitments made
- Questions asked
- Key proposals

TRANSCRIPT:
{transcript}"""
        }
    
    def _truncate_transcript(self, transcript: str, max_chars: int = 30000) -> str:
        """Обрезать транскрипт если слишком длинный."""
        if len(transcript) <= max_chars:
            return transcript
        
        logger.warning(f"Транскрипт слишком длинный ({len(transcript)} символов), обрезаю до {max_chars}")
        
        # Обрезаем, сохраняя начало и конец
        separator = "\n\n[...пропущено...]\n\n"
        available = max_chars - len(separator)
        half = available // 2
        return transcript[:half] + separator + transcript[-half:]
    
    def _generate_summary(
        self, 
        transcript: str, 
        prompts: Dict[str, str],
        speakers: Optional[List[str]] = None
    ) -> str:
        """Генерация основного саммари."""
        transcript_truncated = self._truncate_transcript(transcript)
        
        if speakers and len(speakers) > 1:
            user_prompt = prompts["user_summary_with_speakers"].format(
                transcript=transcript_truncated,
                speakers=", ".join(speakers)
            )
        else:
            user_prompt = prompts["user_summary"].format(
                transcript=transcript_truncated
            )
        
        return self._call_llm(
            prompts["system_summary"],
            user_prompt,
            temperature=0.3
        )
    
    def _extract_action_items(
        self, 
        transcript: str, 
        prompts: Dict[str, str],
        speakers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Извлечение action items."""
        transcript_truncated = self._truncate_transcript(transcript, max_chars=20000)
        
        user_prompt = prompts["user_actions"].format(
            transcript=transcript_truncated
        )
        
        response = self._call_llm(
            prompts["system_actions"],
            user_prompt,
            temperature=0.1  # Низкая температура для структурированного вывода
        )
        
        # Парсим JSON из ответа
        try:
            # Ищем JSON в ответе
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                items = json.loads(json_str)
                
                if isinstance(items, list):
                    return items
        except json.JSONDecodeError as e:
            logger.warning(f"Не удалось распарсить action items: {e}")
        
        return []
    
    def _analyze_speakers(
        self, 
        transcript: str, 
        prompts: Dict[str, str],
        speakers: List[str]
    ) -> Dict[str, str]:
        """Анализ вклада каждого спикера."""
        transcript_truncated = self._truncate_transcript(transcript, max_chars=25000)
        
        user_prompt = prompts["user_speakers"].format(
            transcript=transcript_truncated,
            speakers=", ".join(speakers)
        )
        
        response = self._call_llm(
            prompts["system_speakers"],
            user_prompt,
            temperature=0.3
        )
        
        # Возвращаем как текст (структурированный ответ)
        return {"analysis": response}


def format_summary_text(summary_result: Dict[str, Any]) -> str:
    """
    Форматировать результат суммаризации в читаемый текст.
    
    Args:
        summary_result: Результат от MeetingSummarizer.summarize()
        
    Returns:
        Отформатированный текст
    """
    lines = []
    
    # Заголовок
    lines.append("=" * 60)
    lines.append("📋 САММАРИ ВСТРЕЧИ")
    lines.append("=" * 60)
    lines.append("")
    
    # Основное саммари
    if summary_result.get("summary"):
        lines.append(summary_result["summary"])
        lines.append("")
    
    # Action Items
    if summary_result.get("action_items"):
        lines.append("-" * 40)
        lines.append("📌 ACTION ITEMS")
        lines.append("-" * 40)
        
        for i, item in enumerate(summary_result["action_items"], 1):
            action = item.get("action", "")
            assignee = item.get("assignee")
            deadline = item.get("deadline")
            
            line = f"{i}. {action}"
            if assignee:
                line += f" → {assignee}"
            if deadline:
                line += f" (срок: {deadline})"
            
            lines.append(line)
        
        lines.append("")
    
    # Анализ спикеров
    if summary_result.get("speaker_analysis"):
        lines.append("-" * 40)
        lines.append("👥 АНАЛИЗ СПИКЕРОВ")
        lines.append("-" * 40)
        
        analysis = summary_result["speaker_analysis"].get("analysis", "")
        if analysis:
            lines.append(analysis)
        
        lines.append("")
    
    # Метаданные
    lines.append("-" * 40)
    if summary_result.get("model"):
        lines.append(f"Модель: {summary_result['model']}")
    if summary_result.get("processing_time"):
        lines.append(f"Время обработки: {summary_result['processing_time']:.1f} сек")
    
    return "\n".join(lines)


def check_summarizer_available() -> bool:
    """Проверить доступность суммаризации."""
    return bool(Config.GROQ_API_KEY)

