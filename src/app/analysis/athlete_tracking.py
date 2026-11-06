"""Трекинг гимнасток: AthleteTrack, трекинг по кадрам, выбор целевых треков.

Шаг 6а рефакторинга: вынесено из src/app/main.py. Тела функций не менялись.
"""

import numpy as np

from .metrics import (
    LANDMARK_LEFT_WRIST,
    LANDMARK_RIGHT_WRIST,
    compute_ankle_height,
    compute_ankle_y,
    compute_arm_symmetry,
    compute_center_of_mass,
    compute_hip_height,
    compute_knee_angle,
    compute_leg_opening_angle,
    compute_rotation_angle_world,
    compute_torso_tilt,
    compute_torso_tilt_signed,
    compute_wrist_height,
    _point,
    _skeleton_center,
    _skeleton_size,
)

# Размер группы для группового формата (число гимнасток)
GROUP_SIZE = 5

# Максимальное нормализованное расстояние для сопоставления скелета с треком гимнастки
MAX_ATHLETE_MATCH_DIST = 0.30


def select_target_athletes(poses, format_type: str, group_size: int = GROUP_SIZE):
    """
    Выбирает целевых гимнасток из списка всех обнаруженных поз.

    poses: список списков landmark-точек (каждый элемент — скелет одной гимнастки).
    format_type: "individual" — один самый крупный скелет; "group" — group_size самых крупных.

    Возвращает список выбранных скелетов (для individual — не более 1).
    """
    if not poses:
        return []

    # Оценка «размера» скелета по bounding box ключевых точек
    def _size(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    ranked = sorted(poses, key=_size, reverse=True)

    if format_type == "group":
        return ranked[:group_size]
    # individual — один самый крупный
    return ranked[:1]


class AthleteTrack:
    """
    Трек одной гимнастки: накапливает временные ряды признаков по кадрам.
    Используется для последующего вычисления агрегированных метрик.
    """

    def __init__(self, index: int):
        self.index = index
        self.torso_angles = []
        self.wrist_heights = []
        self.wrist_y_left = []   # высота левого запястья по кадрам (для асимметрии подброса)
        self.wrist_y_right = []  # высота правого запястья по кадрам
        self.com_positions = []
        self.knee_angles_left = []
        self.knee_angles_right = []
        self.leg_opening_angles = []
        self.hip_heights = []
        self.signed_torso_angles = []
        self.ankle_heights = []
        self.ankle_y_left = []
        self.ankle_y_right = []
        self.arm_symmetries = []
        self.rotation_angles_world = []
        self.wrist_positions = []   # средние позиции запястий (для связывания булав)
        self.last_center = None
        self.size_sum = 0.0
        self.frames_seen = 0

    def add_frame(self, landmarks, world_landmarks=None) -> None:
        """Добавляет кадр: вычисляет все признаки и сохраняет в ряды."""
        self.torso_angles.append(compute_torso_tilt(landmarks))
        self.wrist_heights.append(compute_wrist_height(landmarks))
        self.wrist_y_left.append(float(landmarks[LANDMARK_LEFT_WRIST].y))
        self.wrist_y_right.append(float(landmarks[LANDMARK_RIGHT_WRIST].y))
        self.com_positions.append(compute_center_of_mass(landmarks))
        self.knee_angles_left.append(compute_knee_angle(landmarks, "left"))
        self.knee_angles_right.append(compute_knee_angle(landmarks, "right"))
        self.leg_opening_angles.append(compute_leg_opening_angle(landmarks))
        self.hip_heights.append(compute_hip_height(landmarks))
        self.signed_torso_angles.append(compute_torso_tilt_signed(landmarks))
        self.ankle_heights.append(compute_ankle_height(landmarks))
        self.ankle_y_left.append(compute_ankle_y(landmarks, "left"))
        self.ankle_y_right.append(compute_ankle_y(landmarks, "right"))
        self.arm_symmetries.append(compute_arm_symmetry(landmarks))

        left_w = _point(landmarks, LANDMARK_LEFT_WRIST)
        right_w = _point(landmarks, LANDMARK_RIGHT_WRIST)
        self.wrist_positions.append(((left_w[0] + right_w[0]) / 2, (left_w[1] + right_w[1]) / 2))

        if world_landmarks is not None:
            try:
                self.rotation_angles_world.append(compute_rotation_angle_world(world_landmarks))
            except Exception:
                pass

        self.last_center = _skeleton_center(landmarks)
        self.size_sum += _skeleton_size(landmarks)
        self.frames_seen += 1

    @property
    def avg_size(self) -> float:
        """Средний размер скелета за всё время наблюдения."""
        return self.size_sum / max(self.frames_seen, 1)


def track_athletes(
    frames_with_poses,
    fps: float,
) -> list:
    """
    Строит треки гимнасток по последовательности кадров.

    frames_with_poses: список кортежей (frame_bgr, pose_landmarks, world_landmarks).
    Возвращает список AthleteTrack.
    """
    tracks: list = []

    for frame, pose_landmarks_list, world_landmarks_list in frames_with_poses:
        if not pose_landmarks_list:
            continue

        used_tracks = set()
        for i, landmarks in enumerate(pose_landmarks_list):
            center = _skeleton_center(landmarks)
            world = None
            if world_landmarks_list and i < len(world_landmarks_list):
                world = world_landmarks_list[i]

            # Ищем ближайший трек, у которого есть last_center
            best_track = None
            best_dist = MAX_ATHLETE_MATCH_DIST  # порог
            for t in tracks:
                if id(t) in used_tracks or t.last_center is None:
                    continue
                d = (center[0] - t.last_center[0]) ** 2 + (center[1] - t.last_center[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_track = t

            if best_track is None:
                best_track = AthleteTrack(index=len(tracks))
                tracks.append(best_track)

            best_track.add_frame(landmarks, world)
            used_tracks.add(id(best_track))

    return tracks


def select_target_tracks(tracks: list, format_type: str, group_size: int = GROUP_SIZE) -> list:
    """
    Выбирает целевые треки: individual — самый крупный, group — group_size самых крупных.
    """
    ranked = sorted(tracks, key=lambda t: t.avg_size, reverse=True)
    if format_type == "group":
        return ranked[:group_size]
    return ranked[:1]
