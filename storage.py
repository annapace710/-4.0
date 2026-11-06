"""
Хранение отчётов тренировок — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/storage.py. Этот модуль оставлен в корне,
чтобы существующие импорты (`import storage` в src/app/main.py и тестах)
продолжали работать без изменений.
"""

from src.app.storage import *  # noqa: F401,F403
from src.app.storage import REPORTS_DIR, RETENTION_DAYS  # noqa: F401
