"""
Генерация PDF-карточки тренировки — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/report_generator.py. Этот модуль оставлен в корне,
чтобы существующие импорты (`from report_generator import ...` в src/app/main.py)
продолжали работать без изменений.
"""

from src.app.report_generator import *  # noqa: F401,F403
