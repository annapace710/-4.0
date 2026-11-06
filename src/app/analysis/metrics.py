"""Чистые геометрические метрики движения (без HTTP-понятий).

Шаг 6а рефакторинга: вынесено из src/app/main.py. Тела функций не менялись.
"""

import numpy as np

# Индексы ключевых точек MediaPipe Pose (33 точки)
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26
LANDMARK_LEFT_ANKLE = 27
LANDMARK_RIGHT_ANKLE = 28


def _point(landmarks, index: int) -> np.ndarray:
    """Возвращает координаты (x, y) ключевой точки."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Угол между двумя векторами в градусах."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _skeleton_center(landmarks) -> tuple:
    """Центр скелета: среднее всех ключевых точек."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _skeleton_size(landmarks) -> float:
    """Площадь bounding box скелета (для ранжирования «кто на переднем плане»)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def compute_torso_tilt(landmarks) -> float:
    """
    Угол наклона корпуса от вертикали (в градусах).
    Используется середина плеч и середина таза.
    """
    shoulder_mid = (_point(landmarks, LANDMARK_LEFT_SHOULDER) + _point(landmarks, LANDMARK_RIGHT_SHOULDER)) / 2
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2

    torso_vec = shoulder_mid - hip_mid  # от таза к плечам
    vertical = np.array([0.0, -1.0])    # направление «вверх»

    return _angle_between(torso_vec, vertical)


def compute_wrist_height(landmarks) -> float:
    """
    Нормализованная высота запястий (среднее y левого и правого).
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ руки.
    """
    left_y = landmarks[LANDMARK_LEFT_WRIST].y
    right_y = landmarks[LANDMARK_RIGHT_WRIST].y
    return float((left_y + right_y) / 2)


def compute_center_of_mass(landmarks) -> np.ndarray:
    """
    Приблизительный центр масс тела: среднее всех видимых точек.
    Используется для отслеживания траектории движения.
    """
    visible = [
        np.array([lm.x, lm.y])
        for lm in landmarks
        if getattr(lm, "visibility", 1.0) > 0.5
    ]
    if not visible:
        return np.array([0.5, 0.5])
    return np.mean(visible, axis=0)


def compute_knee_angle(landmarks, side: str) -> float:
    """
    Угол в коленном суставе (0–180°).
    side: "left" или "right". 180° — полностью выпрямленная нога.
    """
    if side == "left":
        hip = _point(landmarks, LANDMARK_LEFT_HIP)
        knee = _point(landmarks, LANDMARK_LEFT_KNEE)
        ankle = _point(landmarks, LANDMARK_LEFT_ANKLE)
    else:
        hip = _point(landmarks, LANDMARK_RIGHT_HIP)
        knee = _point(landmarks, LANDMARK_RIGHT_KNEE)
        ankle = _point(landmarks, LANDMARK_RIGHT_ANKLE)

    v1 = hip - knee
    v2 = ankle - knee
    return _angle_between(v1, v2)


def compute_leg_opening_angle(landmarks) -> float:
    """
    Угол раскрытия ног (шпагат) при прыжке: угол между векторами
    «таз → левая стопа» и «таз → правая стопа» (0–180°).
    Чем ближе к 180°, тем шире раскрыты ноги.
    """
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2
    left_vec = _point(landmarks, LANDMARK_LEFT_ANKLE) - hip_mid
    right_vec = _point(landmarks, LANDMARK_RIGHT_ANKLE) - hip_mid
    return _angle_between(left_vec, right_vec)


def compute_hip_height(landmarks) -> float:
    """
    Нормализованная высота таза (среднее y левого и правого бедра).
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ таз.
    """
    left_y = landmarks[LANDMARK_LEFT_HIP].y
    right_y = landmarks[LANDMARK_RIGHT_HIP].y
    return float((left_y + right_y) / 2)


def compute_ankle_y(landmarks, side: str) -> float:
    """
    Нормализованная высота лодыжки (y) для заданной стороны.
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ стопа.
    Используется для определения опорной ноги в равновесии.
    """
    idx = LANDMARK_LEFT_ANKLE if side == "left" else LANDMARK_RIGHT_ANKLE
    return float(landmarks[idx].y)


def compute_ankle_height(landmarks) -> float:
    """
    Средняя нормализованная высота стоп (среднее y левой и правой лодыжки).
    Чем МЕНЬШЕ значение, тем ВЫШЕ подняты стопы (прыжок).
    """
    left_y = landmarks[LANDMARK_LEFT_ANKLE].y
    right_y = landmarks[LANDMARK_RIGHT_ANKLE].y
    return float((left_y + right_y) / 2)


def compute_arm_symmetry(landmarks) -> float:
    """
    Асимметрия рук: модуль разницы высот левого и правого запястья (0..1).
    Чем ближе к 0, тем симметричнее руки (важно для пируэта).
    """
    left_y = landmarks[LANDMARK_LEFT_WRIST].y
    right_y = landmarks[LANDMARK_RIGHT_WRIST].y
    return float(abs(left_y - right_y))


def compute_rotation_angle_world(world_landmarks) -> float:
    """
    Угол ориентации плечевой линии в горизонтальной плоскости (градусы).
    Использует 3D world landmarks (x — вправо, y — вверх, z — от камеры).
    Применяется для подсчёта числа витков в пируэте.
    """
    left = world_landmarks[LANDMARK_LEFT_SHOULDER]
    right = world_landmarks[LANDMARK_RIGHT_SHOULDER]
    dx = right.x - left.x
    dz = right.z - left.z
    return float(np.degrees(np.arctan2(dz, dx)))


def compute_torso_tilt_signed(landmarks) -> float:
    """
    Угол наклона корпуса от вертикали в градусах со знаком.
    Положительный — наклон влево, отрицательный — вправо (в плоскости кадра).
    Используется для анализа симметрии оси вращения в пируэте.
    """
    shoulder_mid = (_point(landmarks, LANDMARK_LEFT_SHOULDER) + _point(landmarks, LANDMARK_RIGHT_SHOULDER)) / 2
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2

    torso_vec = shoulder_mid - hip_mid  # от таза к плечам
    vertical = np.array([0.0, -1.0])    # направление «вверх»

    # Угол со знаком: положительный при наклоне влево от вертикали
    # (если плечи смещены влево относительно таза, torso_vec[0] < 0 -> угол положительный)
    signed_deg = float(np.degrees(np.arctan2(-torso_vec[0], -torso_vec[1])))
    # Ограничиваем диапазон до [-90, 90]
    return float(np.clip(signed_deg, -90.0, 90.0))


def compute_deep_metrics(
    torso_angles: list,
    wrist_heights: list,
    com_positions: list,
    fps: float,
    knee_angles_left: list = None,
    knee_angles_right: list = None,
    leg_opening_angles: list = None,
    hip_heights: list = None,
    signed_torso_angles: list = None,
    ankle_heights: list = None,
    ankle_y_left: list = None,
    ankle_y_right: list = None,
    arm_symmetries: list = None,
    rotation_angles_world: list = None,
    wrist_y_left: list = None,
    wrist_y_right: list = None,
) -> dict:
    """Агрегирует метрики углублённого анализа движений."""
    result = {}

    # 1. Угол наклона корпуса при кувырке
    if torso_angles:
        arr = np.array(torso_angles)
        result["torso_tilt"] = {
            "avg_deg": round(float(np.mean(arr)), 1),
            "max_deg": round(float(np.max(arr)), 1),
            "min_deg": round(float(np.min(arr)), 1),
        }
    else:
        result["torso_tilt"] = None

    # 2. Высота рук при подбросе булав
    if wrist_heights:
        arr = np.array(wrist_heights)
        highest_y = float(np.min(arr))          # самая верхняя точка запястий
        wrist_metrics = {
            "avg_y": round(float(np.mean(arr)), 3),
            "highest_y": round(highest_y, 3),
            # Доля высоты кадра, на которую поднимались руки (0..1)
            "reach_ratio": round(max(0.0, 1.0 - highest_y), 3),
        }
        # Высоты каждой руки отдельно: усреднение левого/правого запястья
        # «размывает» асимметрию, поэтому досягаемость проверяется и по каждой руке
        for side, side_arr in (("left", wrist_y_left), ("right", wrist_y_right)):
            if side_arr:
                s = np.array(side_arr)
                side_highest_y = float(np.min(s))
                wrist_metrics[f"{side}_highest_y"] = round(side_highest_y, 3)
                wrist_metrics[f"{side}_reach_ratio"] = round(max(0.0, 1.0 - side_highest_y), 3)
        # Самая низкая точка запястий за всё видео: признак «руки опущены на пол»
        # (переворот вперёд — постановка рук на опору перед проходом)
        wrist_metrics["lowest_y"] = round(float(np.max(arr)), 3)
        result["wrist_height"] = wrist_metrics
    else:
        result["wrist_height"] = None

    # 3. Траектория движения (центр масс)
    if len(com_positions) > 1:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        result["trajectory"] = {
            "total_distance": round(float(np.sum(distances)), 4),
            "x_range": round(float(np.ptp(com[:, 0])), 4),
            "y_range": round(float(np.ptp(com[:, 1])), 4),
        }
    else:
        result["trajectory"] = None

    # 4. Скорость и плавность движения
    if len(com_positions) > 2:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        speeds = distances * fps  # нормализованные единицы в секунду

        accelerations = np.diff(speeds)
        # Количество «рывков» — смен знака ускорения
        jerk_count = int(np.sum(np.abs(np.diff(np.sign(accelerations))) > 0))
        # Плавность: 1 при нулевом разбросе ускорений, стремится к 0 при рывках
        smoothness = round(1.0 / (1.0 + float(np.std(accelerations))), 4)

        result["motion"] = {
            "avg_speed": round(float(np.mean(speeds)), 4),
            "max_speed": round(float(np.max(speeds)), 4),
            "smoothness": smoothness,
            "jerk_count": jerk_count,
        }
    else:
        result["motion"] = None

    # 5. Признаки, специфичные для упражнений (собираются только при наличии данных)
    # Равновесие: угол в колене опорной ноги (чем ближе к 180°, тем выпрямленнее нога)
    if knee_angles_left or knee_angles_right:
        knee = {}
        if knee_angles_left:
            arr_l = np.array(knee_angles_left)
            knee["left_avg_deg"] = round(float(np.mean(arr_l)), 1)
            knee["left_min_deg"] = round(float(np.min(arr_l)), 1)
        if knee_angles_right:
            arr_r = np.array(knee_angles_right)
            knee["right_avg_deg"] = round(float(np.mean(arr_r)), 1)
            knee["right_min_deg"] = round(float(np.min(arr_r)), 1)
        result["knee_angle"] = knee

    # Прыжок: угол раскрытия ног (шпагат) — чем ближе к 180°, тем шире
    if leg_opening_angles:
        arr = np.array(leg_opening_angles)
        result["leg_opening"] = {
            "max_deg": round(float(np.max(arr)), 1),
            "avg_deg": round(float(np.mean(arr)), 1),
        }

    # Прыжок/кувырок: высота таза (для оценки вертикального отрыва и группировки)
    if hip_heights:
        arr = np.array(hip_heights)
        result["hip_height"] = {
            "min_y": round(float(np.min(arr)), 3),  # самая высокая точка таза (прыжок)
            "max_y": round(float(np.max(arr)), 3),  # самая низкая точка таза (кувырок)
            "avg_y": round(float(np.mean(arr)), 3),
        }

    # Пируэт: наклон корпуса со знаком (симметрия оси вращения)
    if signed_torso_angles:
        arr = np.array(signed_torso_angles)
        result["torso_tilt_signed"] = {
            "avg_deg": round(float(np.mean(arr)), 1),
            "abs_max_deg": round(float(np.max(np.abs(arr))), 1),
        }

    # Прыжок: высота стоп при отрыве
    if ankle_heights:
        arr = np.array(ankle_heights)
        result["ankle_height"] = {
            "min_y": round(float(np.min(arr)), 3),  # самая высокая точка стоп
            "avg_y": round(float(np.mean(arr)), 3),
        }

    # Равновесие: высоты лодыжек (для определения опорной ноги)
    if ankle_y_left and ankle_y_right:
        result["ankle_y"] = {
            "left_avg_y": round(float(np.mean(ankle_y_left)), 3),
            "right_avg_y": round(float(np.mean(ankle_y_right)), 3),
            # Средняя разница высот: > 0 => левая выше (правая опорная), < 0 => наоборот
            "avg_diff": round(float(np.mean(np.array(ankle_y_left) - np.array(ankle_y_right))), 3),
        }
    elif ankle_y_left:
        result["ankle_y"] = {
            "left_avg_y": round(float(np.mean(ankle_y_left)), 3),
            "right_avg_y": None,
            "avg_diff": None,
        }
    elif ankle_y_right:
        result["ankle_y"] = {
            "left_avg_y": None,
            "right_avg_y": round(float(np.mean(ankle_y_right)), 3),
            "avg_diff": None,
        }

    # Пируэт: асимметрия рук
    if arm_symmetries:
        arr = np.array(arm_symmetries)
        result["arm_symmetry"] = {
            "avg_diff": round(float(np.mean(arr)), 3),
            "max_diff": round(float(np.max(arr)), 3),
        }

    # Пируэт: число витков по углу поворота плечевой линии
    if rotation_angles_world:
        angles = np.unwrap(np.radians(np.array(rotation_angles_world)))
        total_rotation_deg = float(np.degrees(angles[-1] - angles[0]))
        result["rotation_turns"] = {
            "total_deg": round(total_rotation_deg, 1),
            "turns": round(abs(total_rotation_deg) / 360.0, 2),
        }

    return result


def _latency_stats(frame_latencies_ms: list) -> dict:
    """
    Статистика латентности обработки кадра (мс): средняя, p95, максимум.
    Используется для контроля требований realtime («задержка кадра < 100 мс»).
    """
    if not frame_latencies_ms:
        return None
    arr = np.array(frame_latencies_ms)
    return {
        "avg_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "samples": len(frame_latencies_ms),
    }
