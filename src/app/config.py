"""
Единая конфигурация приложения (Шаг 2 рефакторинга).

Все настройки собраны в одном модуле: пути, лимиты, авторизация, CORS.
Значения берутся из переменных окружения с разумными значениями по умолчанию.

Важно для тестов: эндпоинты читают конфигурацию через имена, импортированные
в src.app.main (из этого модуля), поэтому патчи атрибутов модуля src.app.main
в тестах продолжают работать.
"""

import os

# --- Корень проекта ---
# src/app/config.py лежит на два уровня глубже корня. Пути к модели, ONNX-модели
# булав и директориям данных якорятся к корню, чтобы физическое расположение
# этих файлов не менялось при переносе кода.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Пути к моделям и данным ---
# Модель PoseLandmarker: ставится скриптом scripts/download_pose_model.py
# (скачивание при импорте убрано — side effect на import + race condition воркеров).
MODEL_PATH = os.path.join(_PROJECT_ROOT, "pose_landmarker_lite.task")
MODEL_SHA256_PATH = MODEL_PATH + ".sha256"

# Кастомная ONNX-модель детекции булав (обучается через train_club_detector.py).
# Если файл отсутствует — используется HSV-сегментация (club_tracker.py).
CLUB_MODEL_PATH = os.path.join(_PROJECT_ROOT, "club_detector.onnx")

# Директория для временного хранения загруженных видео
UPLOAD_DIR = os.path.join(_PROJECT_ROOT, "uploads", "temp")


def ensure_upload_dir() -> None:
    """Гарантирует существование директории для загрузок."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# Директория создаётся при импорте конфига — как и раньше (операция идемпотентная,
# race condition воркеров не страшен: exist_ok=True).
ensure_upload_dir()


# --- Загрузка видео ---
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v", ".hevc", ".h264", ".h265"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Авторизация и rate limiting на /upload (пункт 6 ревью) ---
# Список валидных API-ключей задаётся переменной окружения API_KEYS (через запятую).
# Если переменная не задана или пуста — открытый доступ (режим разработки/MVP).
# Для продакшена: задайте API_KEYS в окружении, клиенты передают ключ в заголовке X-API-Key.
API_KEYS = [
    k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()
]
API_KEY_HEADER = "X-API-Key"

# Rate limit на /upload: сколько загрузок в минуту разрешено на один ключ/IP.
# Счётчики хранятся в памяти процесса (сбрасываются при рестарте) — для MVP достаточно.
UPLOAD_RATE_LIMIT_PER_MINUTE = int(os.environ.get("UPLOAD_RATE_LIMIT_PER_MINUTE", "5"))

# --- LLM-агент-ассистент тренера ---
# Rate limit на агент: сообщений в минуту на клиента (жёстче, чем /upload,
# потому что бесплатный тариф GigaChat имеет лимиты запросов).
AGENT_RATE_LIMIT_PER_MINUTE = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "10"))
# Максимум сообщений истории диалога, отправляемых в LLM (ограничение стоимости)
AGENT_MAX_HISTORY_MESSAGES = 12

# --- CORS ---
# Разрешаем обращения с фронтенда (SourceCraft Sites), локальной разработки
# и туннелей Google Colab (Cloudflare/ngrok).
ALLOWED_ORIGINS = [
    "https://annapace.sourcecraft.site",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
ALLOWED_ORIGIN_REGEX = r"https?://(.*\.trycloudflare\.com|.*\.ngrok-free\.app|.*\.ngrok\.io)$"
