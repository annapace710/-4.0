"""Детектор булав: HSV-сегментация + опциональная ONNX-модель (YOLOv8).

Шаг 6б рефакторинга: вынесено из club_tracker.py. Тела функций не менялись.
Булавы художественной гимнастики обычно имеют яркую окраску (жёлтые, красные,
оранжевые, синие), что позволяет выделять их цветовой сегментацией в HSV.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schemas import ClubDetection, ClubTrajectory

# --- Конфигурация HSV-сегментации по умолчанию ---
# Диапазоны подобраны для типичных ярких булав (жёлтый/оранжевый/красный).
DEFAULT_HSV_RANGES = [
    # Красный (два диапазона из-за разрыва оттенка на границе 0/180)
    ((0, 100, 80), (10, 255, 255)),
    ((160, 100, 80), (180, 255, 255)),
    # Оранжевый
    ((10, 100, 80), (25, 255, 255)),
    # Жёлтый
    ((25, 100, 80), (38, 255, 255)),
]

# Параметры морфологической очистки маски
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 250          # минимальная площадь контура булавы (в пикселях)
MAX_CONTOUR_AREA_RATIO = 0.08   # максимальная доля площади кадра для контура
MIN_DETECTION_CONFIDENCE = 0.35 # порог уверенности для ONNX-модели
NMS_IOU_THRESHOLD = 0.45        # порог IoU для подавления дублирующихся боксов ONNX-модели


def _hsv_mask(hsv: np.ndarray) -> np.ndarray:
    """Строит бинарную маску по заданным HSV-диапазонам."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in DEFAULT_HSV_RANGES:
        part = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, part)
    return mask


def _contours_to_detections(mask: np.ndarray) -> List[ClubDetection]:
    """Превращает маску в список детекций булав (по контурам)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    frame_h, frame_w = mask.shape[:2]
    max_area = MAX_CONTOUR_AREA_RATIO * frame_h * frame_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        detections.append(
            ClubDetection(
                x=cx / frame_w,
                y=cy / frame_h,
                area=area,
            )
        )

    # Сортируем по x, чтобы «левая» и «правая» булава были стабильными
    detections.sort(key=lambda d: d.x)
    return detections


def _apply_nms(detections: List[ClubDetection], boxes: List[List[float]]) -> List[ClubDetection]:
    """Подавление немаксимумов (NMS): убирает пересекающиеся боксы-дубли.

    detections и boxes параллельны (boxes[i] — бокс detections[i] в формате
    [x0, y0, x1, y1], нормализованные координаты). Возвращает подмножество
    detections, отсортированное как исходный список.
    """
    if len(detections) <= 1:
        return detections
    scores = [det.confidence for det in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, MIN_DETECTION_CONFIDENCE, NMS_IOU_THRESHOLD)
    if indices is None or len(indices) == 0:
        # NMS не сработал (например, пустой ввод) — возвращаем как есть
        return detections
    # OpenCV может вернуть индексы как [[i], ...] или [i, ...] — нормализуем
    flat = [int(item) for item in np.array(indices).flatten()]
    # Сохраняем исходный порядок сортировки по x (список уже отсортирован по x)
    keep = sorted(flat)
    return [detections[i] for i in keep]


class ClubTracker:
    """
    Трекер булав: детекция на каждом кадре + сопоставление детекций
    с траекториями по расстоянию (простой двухкадровый трекинг).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hsv_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    ):
        self.model_path = model_path
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES
        self._net = None
        self._input_size = (640, 640)

        if model_path and os.path.exists(model_path):
            try:
                self._net = cv2.dnn.readNetFromONNX(model_path)
            except Exception as exc:  # pragma: no cover - зависит от окружения
                print(f"Не удалось загрузить ONNX-модель {model_path}: {exc}")
                self._net = None

    @property
    def uses_custom_model(self) -> bool:
        """Используется ли кастомная ONNX-модель."""
        return self._net is not None

    def _detect_with_model(self, frame_bgr: np.ndarray) -> List[ClubDetection]:
        """Детекция булав кастомной ONNX-моделью (формат выхода YOLOv8).

        Выход сети имеет форму (4+num_classes, num_anchors) после снятия батч-оси
        (или (1, 4+num_classes, num_anchors) — обрабатывается через squeeze).
        Координаты боксов приходят в масштабе входа сети (640x640), поэтому
        нормализуются на размер входа, а не исходного кадра.
        """
        blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, self._input_size, swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward()

        detections = []
        # Снимаем только батч-ось: (1, 4+nc, A) -> (4+nc, A).
        # Внимание: np.squeeze здесь нельзя — он снял бы и ось якорей длины 1
        # (5, 1) -> (5,), ломая парсинг одиночной детекции.
        out = outputs[0] if outputs.ndim == 3 and outputs.shape[0] == 1 else outputs
        if out.ndim != 2 or out.shape[0] < 5:
            # Неожиданная форма выхода — детекций нет (fallback на HSV в detect())
            return detections

        num_anchors = out.shape[1]
        input_w, input_h = self._input_size
        kept_boxes = []  # [x0, y0, x1, y1] в нормализованных координатах, параллельно detections
        for i in range(num_anchors):
            row = out[:, i]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < MIN_DETECTION_CONFIDENCE:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # Координаты сети — в масштабе входа: нормализуем к 0..1
            cx_n = cx / input_w
            cy_n = cy / input_h
            w_n = w / input_w
            h_n = h / input_h
            # Бокс целиком вне кадра (с учётом клампа) — пропускаем
            if w_n <= 0 or h_n <= 0 or cx_n + w_n / 2 <= 0 or cy_n + h_n / 2 <= 0 \
                    or cx_n - w_n / 2 >= 1 or cy_n - h_n / 2 >= 1:
                continue
            x0 = max(0.0, cx_n - w_n / 2)
            y0 = max(0.0, cy_n - h_n / 2)
            x1 = min(1.0, cx_n + w_n / 2)
            y1 = min(1.0, cy_n + h_n / 2)
            kept_boxes.append([x0, y0, x1, y1])
            detections.append(
                ClubDetection(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    area=(x1 - x0) * (y1 - y0) * input_w * input_h,
                    confidence=confidence,
                )
            )

        # NMS: убираем дублирующиеся боксы на одну булаву (YOLOv8 без NMS
        # отдаёт много пересекающихся детекций, что сбивает трекинг)
        detections = _apply_nms(detections, kept_boxes)

        detections.sort(key=lambda d: d.x)
        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[ClubDetection]:
        """Детектирует булавы на кадре (модель или HSV-сегментация)."""
        if self._net is not None:
            dets = self._detect_with_model(frame_bgr)
            if dets:
                return dets
            # fallback на HSV, если модель ничего не нашла
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = _hsv_mask(hsv)
        return _contours_to_detections(mask)

    def track(self, frames: List[np.ndarray]) -> List[ClubTrajectory]:
        """
        Обрабатывает последовательность кадров и возвращает траектории булав.

        frames: список кадров BGR.
        Возвращает список траекторий (до 2), отсортированных по средней позиции x.
        """
        trajectories = []

        for frame_idx, frame in enumerate(frames):
            detections = self.detect(frame)

            if not detections:
                # Если булавы не видны — траектории просто не получают новую точку
                continue

            # Если траекторий меньше, чем детекций — добавляем новые
            while len(trajectories) < len(detections):
                trajectories.append(ClubTrajectory())

            # Жадное сопоставление: для каждой детекции (по x) ищем ближайшую траекторию
            used = set()
            for det in detections:
                best_idx = None
                best_dist = float("inf")
                for idx, traj in enumerate(trajectories):
                    if idx in used or not traj.positions:
                        continue
                    last_x, last_y = traj.positions[-1]
                    dist = (det.x - last_x) ** 2 + (det.y - last_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is None:
                    # Нет подходящей траектории — создаём новую для этой детекции
                    # (или берём первую пустую)
                    empty_idx = next((i for i, t in enumerate(trajectories) if not t.positions), None)
                    if empty_idx is not None:
                        best_idx = empty_idx
                    else:
                        trajectories.append(ClubTrajectory())
                        best_idx = len(trajectories) - 1

                trajectories[best_idx].positions.append((det.x, det.y))
                trajectories[best_idx].frames.append(frame_idx)
                used.add(best_idx)

        # Постобработка: вычисляем пики и точку ловли для каждой траектории
        for traj in trajectories:
            if not traj.positions:
                continue
            ys = [p[1] for p in traj.positions]
            peak_idx = int(np.argmin(ys))
            traj.peak_y = float(np.min(ys))
            traj.peak_frame = peak_idx
            # Точка ловли — последняя точка траектории (возврат в зону рук)
            traj.catch_x, traj.catch_y = traj.positions[-1]

        # Оставляем только непустые траектории
        trajectories = [t for t in trajectories if t.positions]

        # Сортируем по среднему x, чтобы левая/правая были стабильны
        trajectories.sort(key=lambda t: sum(p[0] for p in t.positions) / len(t.positions))
        return trajectories
