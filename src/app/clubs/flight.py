"""Анализ полёта булав: метрики броска и привязка предметов к гимнасткам.

Шаг 6б рефакторинга: вынесено из club_tracker.py. Тела функций не менялись.
"""

from typing import List, Tuple

from .schemas import ClubTrajectory


def analyze_club_flight(
    trajectories: List[ClubTrajectory],
    fps: float,
) -> dict:
    """
    Анализирует траектории булав и возвращает метрики полёта:

    {
        "clubs_detected": 0..N,
        "throw_height": { "left": ..., "right": ..., "max": ..., "min": ... },  # 0..1 (0 — верх кадра)
        "symmetry": { "height_diff": ..., "peak_frame_diff": ... },
        "simultaneity": { "peak_frame_diff": ..., "simultaneous": bool },
        "catch_position": { "left": {"x":..., "y":...}, "right": {...} },
    }
    """
    result = {
        "clubs_detected": len(trajectories),
        "throw_height": None,
        "symmetry": None,
        "simultaneity": None,
        "catch_position": None,
        "dropped": {},
    }

    if not trajectories:
        return result

    # Высота броска: y растёт вниз, поэтому МЕНЬШЕ y = ВЫШЕ бросок
    heights = {}
    for side, traj in zip(["left", "right"], trajectories):
        heights[side] = round(traj.peak_y, 3) if traj.peak_y is not None else None
    heights["max"] = round(min(h for h in heights.values() if h is not None), 3)
    heights["min"] = round(max(h for h in heights.values() if h is not None), 3)
    result["throw_height"] = heights

    # Критерий падения булавы (эвристика): траектория заканчивается у пола
    # (последний y близок к нижней границе кадра) и после пика больше не поднимается
    # на заметную высоту. Ловля, напротив, происходит в зоне рук (y заметно меньше 1).
    CLUB_DROP_Y_THRESHOLD = 0.85
    CLUB_DROP_REBOUND = 0.10
    result["dropped"] = {}
    for side, traj in zip(["left", "right"], trajectories):
        traj_dropped = None
        if traj.peak_y is not None and traj.positions:
            last_y = traj.positions[-1][1]
            ys_after_peak = [p[1] for p in traj.positions[traj.peak_frame:]]
            min_after_peak = min(ys_after_peak) if ys_after_peak else last_y
            rebound = traj.peak_y - min_after_peak  # насколько после пика снова поднималась
            traj_dropped = last_y >= CLUB_DROP_Y_THRESHOLD and rebound <= CLUB_DROP_REBOUND
        result["dropped"][side] = traj_dropped

    if len(trajectories) >= 2:
        left, right = trajectories[0], trajectories[1]
        # Симметрия высоты: разница пиков (0 — идеально)
        height_diff = None
        if left.peak_y is not None and right.peak_y is not None:
            height_diff = round(abs(left.peak_y - right.peak_y), 3)
        result["symmetry"] = {
            "height_diff": height_diff,
            "peak_frame_diff": None,
        }
        # Одновременность: разница номеров кадров достижения пика
        peak_frame_diff = None
        if left.peak_frame is not None and right.peak_frame is not None:
            peak_frame_diff = abs(left.peak_frame - right.peak_frame)
        result["symmetry"]["peak_frame_diff"] = peak_frame_diff
        result["simultaneity"] = {
            "peak_frame_diff": peak_frame_diff,
            "simultaneous": peak_frame_diff is not None and peak_frame_diff <= max(1, int(fps / 10)),
        }
        # Позиция ловли
        result["catch_position"] = {
            "left": {"x": left.catch_x, "y": left.catch_y},
            "right": {"x": right.catch_x, "y": right.catch_y},
        }

    return result


def assign_clubs_to_athletes(
    trajectories: List[ClubTrajectory],
    athlete_wrist_positions: List[List[Tuple[float, float]]],
) -> List[List[ClubTrajectory]]:
    """
    Связывает траектории булав с гимнастками по близости к запястьям.

    athlete_wrist_positions: список; для каждой гимнастки — список позиций запястий
    (нормализованные (x, y)) по кадрам. Размеры списков должны совпадать по кадрам:
    для каждого кадра число позиций запястий = число кадров, где эта гимнастка видна.

    Возвращает список списков траекторий: для каждой гимнастки — её булавы (до 2).
    """
    if not athlete_wrist_positions:
        return []

    # Для каждой гимнастки собираем все позиции запястий (по всем кадрам)
    wrist_points_per_athlete = []
    for wrists in athlete_wrist_positions:
        pts = [(x, y) for (x, y) in wrists]
        wrist_points_per_athlete.append(pts)

    # Для каждой траектории вычисляем среднюю позицию
    traj_means = []
    for traj in trajectories:
        if traj.positions:
            xs = [p[0] for p in traj.positions]
            ys = [p[1] for p in traj.positions]
            traj_means.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        else:
            traj_means.append((0.5, 0.5))

    # Жадное назначение: каждой траектории — ближайшая гимнастка по среднему расстоянию
    # до её запястий
    assignments = []  # для каждой траектории — индекс гимнастки
    used_athletes = set()
    # Сортируем траектории по «силе» (количеству точек) — сначала самые надёжные
    order = sorted(range(len(trajectories)), key=lambda i: len(trajectories[i].positions), reverse=True)
    for i in order:
        best_athlete = None
        best_dist = float("inf")
        for a_idx, wrist_pts in enumerate(wrist_points_per_athlete):
            if a_idx in used_athletes and len(trajectories) <= len(wrist_points_per_athlete):
                # Если гимнасток столько же, сколько траекторий — не переиспользуем
                continue
            if not wrist_pts:
                continue
            dist = min(
                (traj_means[i][0] - wx) ** 2 + (traj_means[i][1] - wy) ** 2
                for (wx, wy) in wrist_pts
            )
            if dist < best_dist:
                best_dist = dist
                best_athlete = a_idx
        if best_athlete is None:
            best_athlete = 0
        assignments.append((i, best_athlete))
        used_athletes.add(best_athlete)

    # Собираем результат: для каждой гимнастки — список её траекторий
    result = [[] for _ in range(len(athlete_wrist_positions))]
    for traj_idx, athlete_idx in assignments:
        result[athlete_idx].append(trajectories[traj_idx])

    # В каждой группе сортируем по x (левая/правая)
    for group in result:
        group.sort(key=lambda t: sum(p[0] for p in t.positions) / len(t.positions))

    return result
