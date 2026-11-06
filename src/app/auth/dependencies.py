"""FastAPI-зависимости авторизации (Шаг 7 рефакторинга).

Зависимости require_api_key и require_agent_access перенесены из
src/app/auth.py (Шаг 3) без изменения логики. Логика проверки —
в src/app/auth/service.py.
"""

from fastapi import HTTPException, Request

from .service import _check_agent_rate_limit, _client_id, _require_api_key_impl


async def require_api_key(request: Request) -> str:
    """FastAPI-зависимость: auth + rate limit для /upload."""
    return _require_api_key_impl(request)


async def require_agent_access(request: Request) -> str:
    """
    FastAPI-зависимость: auth + rate limit для /agent/chat.
    Возвращает ключ клиента (нужен для проверки лимита агента в эндпоинте).
    """
    api_key = _require_api_key_impl(request)
    if not _check_agent_rate_limit(_client_id(request, api_key)):
        # Ленивый резолв лимита: тесты патчат src.app.auth.AGENT_RATE_LIMIT_PER_MINUTE
        from src.app import auth as auth_pkg

        limit = auth_pkg.AGENT_RATE_LIMIT_PER_MINUTE
        raise HTTPException(
            status_code=429,
            detail=(
                f"Превышен лимит сообщений: {limit} в минуту. "
                "Повторите попытку позже."
            ),
        )
    return api_key
