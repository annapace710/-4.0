"""
Pydantic-схемы запросов и ответов API (Шаг 3 рефакторинга).

Схемы дают проверку типов и автодокументацию (/docs) для простых эндпоинтов.

Важно: тело /agent/chat валидируется вручную через model_validate — эндпоинт
должен отвечать 400 (а не стандартную 422 FastAPI) на некорректное тело,
как и раньше, поэтому тело не объявляется параметром эндпоинта.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Сообщение истории диалога с агентом."""

    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    """Тело запроса POST /agent/chat/{report_id}."""

    message: str
    history: Optional[List[ChatMessage]] = None


class AgentChatResponse(BaseModel):
    """Ответ POST /agent/chat/{report_id}."""

    reply: str
    report_id: str


class HealthResponse(BaseModel):
    """Ответ GET /health."""

    status: str
    pose_model_ok: bool


class AgentHealthResponse(BaseModel):
    """Ответ GET /agent/health."""

    llm_configured: bool
    provider: str
