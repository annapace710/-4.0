"""LLM-агент-ассистент тренера (MVP: GigaChat, бесплатный тариф)."""

from src.app.agent.llm_client import get_llm_client, is_llm_configured
from src.app.agent.prompts import build_report_context, format_context_as_json

__all__ = [
    "get_llm_client",
    "is_llm_configured",
    "build_report_context",
    "format_context_as_json",
]
