"""Командная обратная связь (групповой формат): рассинхрон старта, потеря ритма.

Шаг 6а рефакторинга: вынесено из src/app/main.py. Тела функций не менялись.
"""

import numpy as np

# Порог скорости центра масс (нормализованных ед./кадр), ниже которого гимнастка
# считается неподвижной («пауза»)
PAUSE_SPEED_THRESHOLD = 0.01
# Минимальная длительность паузы в кадрах анализа
PAUSE_MIN_FRAMES = 2
# Допустимый рассинхрон старта движения внутри команды (сек)
TEAM_START_SYNC_TOLERANCE_S = 0.3
# Порог «заметного движения» для определения старта (нормализованных ед./кадр)
MOVEMENT_START_SPEED = 0.02


def _pause_frames(com_positions: list) -> int:
    """Число подряд идущих кадров, где гимнастка неподвижна (пауза)."""
    if len(com_positions) < 2:
        return 0
    com = np.array(com_positions)
    speeds = np.linalg.norm(np.diff(com, axis=0), axis=1)
    current = max_run = 0
    for s in speeds:
        if s < PAUSE_SPEED_THRESHOLD:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _movement_start_frame(com_positions: list) -> int:
    """Кадр (индекс в ряду анализа), где скорость впервые превысила порог движения."""
    if len(com_positions) < 2:
        return 0
    com = np.array(com_positions)
    speeds = np.linalg.norm(np.diff(com, axis=0), axis=1)
    for i, s in enumerate(speeds):
        if s >= MOVEMENT_START_SPEED:
            return i + 1
    return 0


def detect_team_feedback(target_tracks: list, fps: float, clubs_per_athlete: list) -> dict:
    """
    Командная обратная связь для группового формата: сопоставляет события
    разных гимнасток по кадрам и формирует общие для команды замечания.

    clubs_per_athlete: список результатов analyze_club_flight по гимнасткам
    (каждый может содержать поле "dropped": {"left": bool, "right": bool}).

    Возвращает:
    {
        "messages": [{"severity": ..., "message": ...}],
        "start_spread_sec": ...,
    }
    """
    messages = []

    # --- Правило 1: рассинхрон старта движения ---
    start_frames = [
        _movement_start_frame(t.com_positions) if t.com_positions else None
        for t in target_tracks
    ]
    valid = [f for f in start_frames if f is not None]
    start_spread_sec = None
    if len(valid) >= 2:
        spread_frames = max(valid) - min(valid)
        start_spread_sec = round(spread_frames / max(fps, 1e-6), 2)
        if start_spread_sec > TEAM_START_SYNC_TOLERANCE_S:
            # Кто начинает раньше всех / позже всех (нумерация с 1 для тренера)
            early_idx = start_frames.index(min(valid))
            late_idx = start_frames.index(max(valid))
            messages.append({
                "severity": "warning",
                "message": (
                    f"Рассинхрон старта движения {start_spread_sec} с: гимнастка №{early_idx + 1} "
                    f"начинает раньше остальных, гимнастка №{late_idx + 1} — позже. "
                    "Синхронизируйте ритм разбега под общий счёт."
                ),
            })

    # --- Правило 2: потеря ритма после падения булавы ---
    # Гимнастка, у которой упала булава, и кадр падения (последний кадр траектории)
    drop_events = []  # (athlete_idx, drop_frame)
    for a_idx, clubs_info in enumerate(clubs_per_athlete or []):
        dropped = (clubs_info or {}).get("dropped") or {}
        # Траектории не сохраняются в результате analyze_club_flight,
        # поэтому кадр падения определяем приближённо: момент ловли неизвестен,
        # используем кадр пика как верхнюю границу.
        if any(v for v in dropped.values() if v is not None):
            drop_events.append(a_idx)

    for a_idx in drop_events:
        # Пауза у каждой гимнастки в целом по видео — прокси для «замерли»
        paused = [
            i for i, t in enumerate(target_tracks)
            if i != a_idx and _pause_frames(t.com_positions) >= PAUSE_MIN_FRAMES
        ]
        if len(paused) >= len(target_tracks) - 1:
            messages.append({
                "severity": "warning",
                "message": (
                    f"Команда теряет ритм после падения булавы у гимнастки №{a_idx + 1}: "
                    f"{len(paused)} из {len(target_tracks) - 1} остальных остановились. "
                    "Продолжайте комбинацию и подхватывайте элемент без общей паузы."
                ),
            })

    return {
        "messages": messages,
        "start_spread_sec": start_spread_sec,
    }
