"""Домен clubs: трекинг булав (детектор, полёт, схемы).

Шаг 6б рефакторинга: выделен из club_tracker.py.
"""

from .schemas import ClubDetection, ClubTrajectory
from .detector import ClubTracker, DEFAULT_HSV_RANGES
from .flight import analyze_club_flight, assign_clubs_to_athletes

__all__ = [
    "ClubDetection",
    "ClubTrajectory",
    "ClubTracker",
    "DEFAULT_HSV_RANGES",
    "analyze_club_flight",
    "assign_clubs_to_athletes",
]
