"""LLM-агент-ассистент тренера (MVP: GigaChat, бесплатный тариф).

Обёртка совместимости (Шаг 5 рефакторинга): реализация перенесена в
src/app/agent/. Пакет оставлен в корне, чтобы существующие импорты
(`from agent.llm_client import ...` в src/app/main.py и тестах)
продолжали работать без изменений.
"""

from src.app.agent.llm_client import get_llm_client, is_llm_configured
from src.app.agent.prompts import build_report_context, format_context_as_json

__all__ = [
    "get_llm_client",
    "is_llm_configured",
    "build_report_context",
    "format_context_as_json",
]
