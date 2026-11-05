"""
Авторизация и rate limiting (Шаг 3 рефакторинга).

Проверки вынесены из main.py в отдельный модуль и оформлены как
FastAPI-зависимости (Depends), чтобы эндпоинты декларировали требования,
а не реализовали проверку вручную.

Важно для тестов: счетчик AGENT_RATE_LIMIT_PER_MINUTE читается из имен
ЭТОГО модуля — патчить в тестах нужно src.app.auth.AGENT_RATE_LIMIT_PER_MINUTE
(раньше патчился атрибут src.app.main).
"""

import os
import threading
import time

from fastapi import HTTPException, Request

from .config import API_KEYS, API_KEY_HEADER, UPLOAD_RATE_LIMIT_PER_MINUTE

# Rate limit агента: сообщений в минуту на клиента (жёстче, чем /upload,
# потому что бесплатный тариф GigaChat имеет лимиты запросов).
AGENT_RATE_LIMIT_PER_MINUTE = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "10"))

_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict = {}  # client_id -> list[timestamp] последних запросов


def _client_id(request: Request, api_key: str) -> str:
    """Идентификатор клиента для rate limit: API-ключ или IP-адрес."""
    return api_key or (request.client.host if request.client else "unknown")


def _check_rate_limit(client_id: str) -> bool:
    """
    Скользящее окно 60 секунд: возвращает True, если запрос разрешен
    (не превышен UPLOAD_RATE_LIMIT_PER_MINUTE).
    """
    now = time.monotonic()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(client_id, [])
        # Убираем записи старше 60 секунд
        bucket[:] = [ts for ts in bucket if now - ts < 60.0]
        if len(bucket) >= UPLOAD_RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True


def _check_agent_rate_limit(client_id: str) -> bool:
    """Скользящее окно 60 с для эндпоинта агента (AGENT_RATE_LIMIT_PER_MINUTE)."""
    now = time.monotonic()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(f"agent:{client_id}", [])
        bucket[:] = [ts for ts in bucket if now - ts < 60.0]
        if len(bucket) >= AGENT_RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True


def _require_api_key_impl(request: Request) -> str:
    """
    Проверяет API-ключ (если авторизация настроена через API_KEYS) и лимит /upload.
    Возвращает ключ клиента (или ""), если авторизация выключена.
    Бросает HTTPException 401/429 при ошибке.
    """
    api_key = request.headers.get(API_KEY_HEADER, "")
    if API_KEYS:
        if api_key not in API_KEYS:
            raise HTTPException(
                status_code=401,
                detail="Недействительный API-ключ. Передайте ключ в заголовке X-API-Key.",
            )
    if not _check_rate_limit(_client_id(request, api_key)):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Превышен лимит загрузок: {UPLOAD_RATE_LIMIT_PER_MINUTE} в минуту. "
                "Повторите попытку позже."
            ),
        )
    return api_key


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
        raise HTTPException(
            status_code=429,
            detail=(
                f"Превышен лимит сообщений: {AGENT_RATE_LIMIT_PER_MINUTE} в минуту. "
                "Повторите попытку позже."
            ),
        )
    return api_key
