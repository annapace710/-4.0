"""
Клиент LLM — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/agent/llm_client.py. Этот модуль оставлен
в корне, чтобы существующие импорты (`from agent.llm_client import ...`,
патчи `agent.llm_client.get_llm_client` в тестах) продолжали работать.
"""

from src.app.agent.llm_client import *  # noqa: F401,F403
from src.app.agent.llm_client import (  # noqa: F401
    BaseLLMClient,
    GigaChatClient,
    LLMNotConfiguredError,
    LLMRequestError,
    get_llm_client,
    is_llm_configured,
)
