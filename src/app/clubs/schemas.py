"""Схемы данных трекинга булав (датаклассы).

Шаг 6б рефакторинга: вынесено из club_tracker.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ClubDetection:
    """Обнаруженная булава в одном кадре."""
    x: float          # нормализованный центр x (0..1)
    y: float          # нормализованный центр y (0..1)
    area: float       # площадь в пикселях
    confidence: float = 1.0


@dataclass
class ClubTrajectory:
    """Траектория одной булавы за всё видео."""
    positions: List[Tuple[float, float]] = field(default_factory=list)  # (x, y)
    frames: List[int] = field(default_factory=list)  # индексы кадров детекций (для командного анализа)
    peak_y: Optional[float] = None     # минимальный y (самая высокая точка)
    peak_frame: Optional[int] = None   # номер кадра, когда достигнут пик
    catch_x: Optional[float] = None    # x в момент ловли
    catch_y: Optional[float] = None    # y в момент ловли
