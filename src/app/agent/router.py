"""Роуты LLM-агента-ассистента тренера (Шаг 6в рефакторинга).

Эндпоинты /agent/chat/{report_id} и /agent/health перенесены из
src/app/main.py без изменения логики.

Особенность (важно для тестов): get_llm_client резолвится ЛЕНИВО, в момент
запроса, через пространство имён src.app.main — тесты патчат
main.get_llm_client, и патч должен попадать ровно в тот объект, который
вызывает эндпоинт. Аналогично is_llm_configured в /agent/health
импортируется внутри функции из корневой обёртки agent.llm_client,
которую патчат тесты.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

import storage

from ..auth import require_agent_access
from ..config import AGENT_MAX_HISTORY_MESSAGES
from ..schemas import AgentChatRequest, AgentChatResponse, AgentHealthResponse
from .llm_client import LLMNotConfiguredError, LLMRequestError
from .prompts import build_report_context

router = APIRouter()


@router.post("/agent/chat/{report_id}", response_model=AgentChatResponse)
async def agent_chat(
    report_id: str,
    request: Request,
    _api_key: str = Depends(require_agent_access),
):
    """
    Чат с LLM-агентом по отчёту о тренировке.

    Тело запроса (JSON):
      {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}

    Ответ: {"reply": "..."}.
    Ошибки: 401/429 — auth/лимит; 404 — отчёт не найден; 400 — пустое сообщение
    или некорректное тело; 503 — LLM не настроен; 502 — ошибка провайдера LLM.
    """
    # 1. Тело запроса. Валидация вручную через model_validate (а не параметром
    # эндпоинта), чтобы на некорректное тело отвечать 400, как и раньше —
    # стандартная FastAPI-валидация вернула бы 422.
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный JSON в теле запроса")

    try:
        payload = AgentChatRequest.model_validate(body)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Некорректное тело запроса")

    message = payload.message
    history = payload.history or []

    # Пустое (из одних пробелов) сообщение — прежняя ошибка 400;
    # Pydantic-схема сама это не отсеивает.
    if not message.strip():
        raise HTTPException(
            status_code=400,
            detail="Поле message обязательно и не может быть пустым",
        )

    # 2. LLM должен быть настроен до обращения к хранилищу (fail fast).
    # Ленивый резолв через src.app.main: тесты патчат main.get_llm_client,
    # поэтому фабрика берётся из пространства имён модуля main в момент
    # запроса, а не на уровне импорта роутера.
    try:
        from src.app import main as main_module

        llm = main_module.get_llm_client()
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 3. Контекст отчёта (детерминированные данные; отчёт не найден -> 404)
    try:
        context = build_report_context(report_id, reports_dir=storage.REPORTS_DIR)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 4. Диалог: системный промпт + контекст + ограниченная история + сообщение
    from agent.prompts import SYSTEM_PROMPT, format_context_as_json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Данные отчёта (JSON):\n"
                + format_context_as_json(context)
                + "\n\nВопрос тренера: "
                + message.strip()
            ),
        },
    ]
    # История: только последние AGENT_MAX_HISTORY_MESSAGES сообщений
    for h in history[-AGENT_MAX_HISTORY_MESSAGES:]:
        if h.content.strip():
            messages.append({"role": h.role, "content": h.content.strip()})

    # 5. Запрос к LLM (цикл function calling внутри ask(): модель может
    #    запросить инструменты динамики — error_trend / previous_report)
    try:
        reply = llm.ask(messages)
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"reply": reply, "report_id": report_id}


@router.get("/agent/health", response_model=AgentHealthResponse)
async def agent_health():
    """Доступность LLM-агента: настроен ли провайдер и установлен ли SDK."""
    # Ленивый импорт из корневой обёртки: тесты патчат is_llm_configured
    # в agent.llm_client, патч должен быть виден в момент запроса.
    from agent.llm_client import is_llm_configured

    return {
        "llm_configured": is_llm_configured(),
        "provider": os.environ.get("LLM_PROVIDER", "gigachat").lower(),
    }
