"""
Системный промпт и сборка контекста отчёта — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/agent/prompts.py. Этот модуль оставлен
в корне, чтобы существующие импорты (`from agent.prompts import ...`
в src/app/main.py и тестах) продолжали работать без изменений.
"""

from src.app.agent.prompts import *  # noqa: F401,F403
from src.app.agent.prompts import (  # noqa: F401
    SYSTEM_PROMPT,
    build_report_context,
    format_context_as_json,
)
