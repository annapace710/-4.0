"""Пакет авторизации и rate limiting (Шаг 7 рефакторинга).

Модуль src/app/auth.py (Шаг 3) превращён в пакет src/app/auth/:
service.py — логика (API-ключи, скользящие окна лимитов),
dependencies.py — FastAPI-зависимости, этот __init__.py — реэкспорт
прежних имён, чтобы существующие импорты продолжали работать:

- src/app/main.py и src/app/agent/router.py: `from .auth import require_api_key, require_agent_access`;
- корневая обёртка main.py: `from src.app.auth import AGENT_RATE_LIMIT_PER_MINUTE`;
- тесты: `import src.app.auth as auth` + патч auth.AGENT_RATE_LIMIT_PER_MINUTE.

Важно для тестов: патч src.app.auth.AGENT_RATE_LIMIT_PER_MINUTE меняет
атрибут ПАКЕТА (этот __init__.py), поэтому реэкспорт сделан через имя,
а логика чтения лимита резолвит атрибут лениво (см. service.py и
dependencies.py).
"""

from .dependencies import require_agent_access, require_api_key
from .service import AGENT_RATE_LIMIT_PER_MINUTE

__all__ = [
    "AGENT_RATE_LIMIT_PER_MINUTE",
    "require_agent_access",
    "require_api_key",
]
