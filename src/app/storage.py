"""
Хранение отчётов — обёртка совместимости (Шаг 6б рефакторинга).

Реализация перенесена в src/app/reports/storage.py. Этот модуль оставлен,
чтобы существующие импорты в src/app/main.py (`import storage`) и тестах
(патчи `storage.REPORTS_DIR`) продолжали работать без изменений: обёртка
отдаёт тот же объект модуля, что использует приложение.
"""

from .reports import storage as _storage  # noqa: F401
from .reports.storage import *  # noqa: F401,F403
from .reports.storage import REPORTS_DIR, RETENTION_DAYS  # noqa: F401
