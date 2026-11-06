"""
Инструменты (function calling) LLM-агента — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/agent/tools.py. Этот модуль оставлен
в корне, чтобы существующие импорты (`from agent.tools import TOOL_SPECS`
в тестах) продолжали работать без изменений.
"""

from src.app.agent.tools import *  # noqa: F401,F403
from src.app.agent.tools import (  # noqa: F401
    TOOL_SPECS,
    execute_tool,
)
