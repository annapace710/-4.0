"""
Трекинг булав — обёртка совместимости (Шаг 6б рефакторинга).

Реализация перенесена в пакет src/app/clubs/:
- src/app/clubs/detector.py — ClubTracker, HSV-константы, NMS;
- src/app/clubs/flight.py — analyze_club_flight, assign_clubs_to_athletes;
- src/app/clubs/schemas.py — ClubDetection, ClubTrajectory.

Этот модуль оставлен, чтобы существующие импорты в src/app/main.py
(`from club_tracker import ...`) и тестах продолжали работать без изменений.
"""

from .clubs.detector import ClubTracker  # noqa: F401
from .clubs.flight import analyze_club_flight, assign_clubs_to_athletes  # noqa: F401
from .clubs.schemas import ClubDetection, ClubTrajectory  # noqa: F401
