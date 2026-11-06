"""
Трекинг булав — обёртка совместимости (Шаг 5 рефакторинга).

Реализация перенесена в src/app/club_tracker.py. Этот модуль оставлен в корне,
чтобы существующие импорты (`from club_tracker import ...` в src/app/main.py
и тестах) продолжали работать без изменений.
"""

from src.app.club_tracker import *  # noqa: F401,F403
