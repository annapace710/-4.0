"""Проверка целостности модели PoseLandmarker (используется lifespan приложения).

Шаг 6а рефакторинга: вынесено из src/app/main.py. Тела функций не менялись.
Скрипт scripts/download_pose_model.py получает эти имена через корневую
обёртку main.py — его код не меняется.
"""

import os

from .config import MODEL_PATH, MODEL_SHA256_PATH


def _file_sha256(path: str) -> str:
    """Вычисляет SHA-256 файла (потоково, для больших файлов)."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_pose_model() -> bool:
    """
    Проверяет наличие модели и её целостность по pin-хешу (если pin-файл есть).

    Возвращает True, только если модель существует И (pin-файл отсутствует —
    тогда в /health будет degraded до первой проверки скриптом; или хеш совпал).
    """
    if not os.path.exists(MODEL_PATH):
        print(
            "Модель PoseLandmarker не найдена. Запустите: "
            "python scripts/download_pose_model.py"
        )
        return False
    if not os.path.exists(MODEL_SHA256_PATH):
        print(
            "ПРЕДУПРЕЖДЕНИЕ: pin-файл SHA-256 модели отсутствует "
            f"({MODEL_SHA256_PATH}). Запустите python scripts/download_pose_model.py, "
            "чтобы зафиксировать доверенный хеш."
        )
        return False
    with open(MODEL_SHA256_PATH, "r", encoding="utf-8") as f:
        pinned = f.read().strip()
    if _file_sha256(MODEL_PATH) != pinned:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: SHA-256 модели PoseLandmarker не совпадает с pin-хешем. "
            "Запустите python scripts/download_pose_model.py, чтобы скачать доверенную версию."
        )
        return False
    return True
