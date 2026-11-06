import os
import time
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2  # OpenCV
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from club_tracker import ClubTracker, analyze_club_flight, assign_clubs_to_athletes
import storage
# get_llm_client остаётся в пространстве имён main (Шаг 6в): тесты патчат
# main.get_llm_client, а агент-роутер (src/app/agent/router.py) резолвит
# фабрику лениво через этот модуль в момент запроса.
from agent.llm_client import get_llm_client  # noqa: F401

from .analysis.athlete_tracking import (
    GROUP_SIZE,
    MAX_ATHLETE_MATCH_DIST,
    AthleteTrack,
    select_target_athletes,
    select_target_tracks,
    track_athletes,
    _skeleton_center,
    _skeleton_size,
)
from .analysis.feedback_rules import (
    SEVERITY_PENALTY,
    SUGGESTED_DRILLS,
    THRESHOLDS,
    detect_errors_and_recommendations,
    _fmt,
    _suggested_drill,
)
from .analysis.metrics import (
    LANDMARK_LEFT_ANKLE,
    LANDMARK_LEFT_ELBOW,
    LANDMARK_LEFT_HIP,
    LANDMARK_LEFT_KNEE,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_LEFT_WRIST,
    LANDMARK_RIGHT_ANKLE,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_RIGHT_HIP,
    LANDMARK_RIGHT_KNEE,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_RIGHT_WRIST,
    compute_ankle_height,
    compute_ankle_y,
    compute_arm_symmetry,
    compute_center_of_mass,
    compute_deep_metrics,
    compute_hip_height,
    compute_knee_angle,
    compute_leg_opening_angle,
    compute_rotation_angle_world,
    compute_torso_tilt,
    compute_torso_tilt_signed,
    compute_wrist_height,
    _angle_between,
    _latency_stats,
    _point,
)
from .analysis.team_feedback import (
    MOVEMENT_START_SPEED,
    PAUSE_MIN_FRAMES,
    PAUSE_SPEED_THRESHOLD,
    TEAM_START_SYNC_TOLERANCE_S,
    detect_team_feedback,
    _movement_start_frame,
    _pause_frames,
)
from .lifespan import _file_sha256, verify_pose_model

from .auth import require_agent_access, require_api_key
from .config import (
    AGENT_MAX_HISTORY_MESSAGES,
    ALLOWED_EXTENSIONS,
    ALLOWED_ORIGINS,
    ALLOWED_ORIGIN_REGEX,
    CLUB_MODEL_PATH,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MODEL_PATH,
    MODEL_SHA256_PATH,
    UPLOAD_DIR,
)
from .schemas import HealthResponse
from .agent.router import router as agent_router
from .reports.router import router as reports_router

# Шаг 6а рефакторинга: чистые функции анализа вынесены в домен src/app/analysis/
# (metrics, athlete_tracking, feedback_rules, team_feedback), проверка модели —
# в src/app/lifespan.py. Шаг 6в: роуты отчётов и LLM-агента вынесены в домены
# src/app/reports/router.py и src/app/agent/router.py и подключаются через
# include_router ниже. Этот модуль реэкспортирует прежние имена (см. импорты
# выше), поэтому `from main import ...` в тестах и скриптах работает как раньше.

# Приложение создаётся с lifespan-контекстом: тяжёлые проверки (целостность
# модели PoseLandmarker) выполняются при старте сервиса, а не на уровне
# модуля (side effect на import мешал тестам и воркерам).
POSE_MODEL_OK = True  # консервативное значение до первого запуска lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Проверяет целостность модели PoseLandmarker при старте приложения."""
    global POSE_MODEL_OK
    POSE_MODEL_OK = verify_pose_model()
    yield


app = FastAPI(lifespan=lifespan)

# --- Авторизация и rate limiting на /upload и /agent/chat ---
# Проверки вынесены в src/app/auth.py и подключаются как зависимости
# (Depends). Константы (API_KEYS, лимиты) по-прежнему берутся из config.py.


# --- Ленивый singleton PoseLandmarker (пункт 5 ревью) ---
# Инстанс MediaPipe дорогой: создается один раз и переиспользуется между запросами.
# threading.Lock защищает от гонки при одновременных первых запросах (threadpool).
_pose_lock = threading.Lock()
_pose_landmarker = None

# Каждое новое видео анализируется со своим базовым таймстампом:
# detect_for_video требует монотонно растущие таймстампы между вызовами
# одного инстанса, поэтому к таймстампам кадра добавляем смещение per-video.
_VIDEO_TS_OFFSET_MS = 100_000  # запас > максимальной длительности одного видео
_pose_ts_offset = [0]  # общий счетчик смещений (под защищенным доступом via _pose_lock)


def get_pose_landmarker():
    """Возвращает переиспользуемый инстанс PoseLandmarker (создается лениво)."""
    global _pose_landmarker
    if _pose_landmarker is None:
        with _pose_lock:
            if _pose_landmarker is None:  # double-checked locking
                base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# CORS: разрешаем обращения с фронтенда (SourceCraft Sites), локальной разработки
# и туннелей Google Colab (Cloudflare/ngrok). Значения импортируются из config.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Домены отчётов и LLM-агента (Шаг 6в): роуты перенесены в
# src/app/reports/router.py и src/app/agent/router.py без изменения логики.
app.include_router(reports_router)
app.include_router(agent_router)

# Пути (MODEL_PATH, MODEL_SHA256_PATH, CLUB_MODEL_PATH, UPLOAD_DIR), лимиты
# загрузок (MAX_FILE_SIZE_MB/BYTES, ALLOWED_EXTENSIONS) и корень проекта
# (_PROJECT_ROOT) импортируются из config.py.


def validate_video_file(filename: str, content_length: int) -> None:
    """Проверяет формат и размер загружаемого видеофайла."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Недопустимый формат файла: {ext}. "
            f"Поддерживаются: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if content_length > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Файл слишком большой. Максимальный размер — {MAX_FILE_SIZE_MB} МБ."
        )


def _capture_error_frames(
    video_path: str,
    fps: float,
    frames_with_poses: list,
    feedback: dict,
    frames_dir: str,
) -> dict:
    """
    Снимает скриншоты моментов ошибок из видео.

    Для каждой ошибки feedback выбирается кадр из числа анализированных:
    для ошибок про наклон/кувырок — кадр с максимальным наклоном корпуса,
    для остальных — равномерно распределённые кадры. На PNG добавляется
    обводка скелета (координаты landmark'ов сохраняются в результат).

    Возвращает {index: {"path", "timestamp", "landmarks"}}.
    """
    result = {}
    errors = (feedback or {}).get("errors") or []
    if not errors or not frames_with_poses:
        return result

    os.makedirs(frames_dir, exist_ok=True)

    # Кадры с одной (целевой) позой — самая крупная гимнастка
    target_frames = []
    for idx, (frame, pose_list, _) in enumerate(frames_with_poses):
        if not pose_list:
            continue
        best = max(pose_list, key=_skeleton_size)
        target_frames.append((idx, frame, best))

    if not target_frames:
        return result

    # Кадр с максимальным наклоном корпуса (момент кувырка/прогиба)
    peak_idx = max(
        range(len(target_frames)),
        key=lambda i: compute_torso_tilt(target_frames[i][2]),
        default=None,
    )

    total = len(errors)
    for i, _err in enumerate(errors):
        # Первая (самая серьёзная) ошибка — на пиковом кадре,
        # остальные — равномерно по видео
        if peak_idx is not None and i == 0:
            pick = peak_idx
        else:
            pick = int(len(target_frames) * (i + 1) / (total + 1)) % len(target_frames)
        idx, frame, landmarks = target_frames[pick]

        timestamp = round(idx * (1.0 / max(fps, 1e-6)), 2)
        frame_png = os.path.join(frames_dir, f"error_{i}.png")
        cv2.imwrite(frame_png, frame)

        result[i] = {
            "path": frame_png,
            "timestamp": timestamp,
            "landmarks": {
                j: {"x": float(lm.x), "y": float(lm.y)}
                for j, lm in enumerate(landmarks)
                if getattr(lm, "visibility", 1.0) > 0.5
            },
        }
    return result


def _build_heatmap(frames_with_poses, heatmap_path: str, width_cells: int = 20, height_cells: int = 12):
    """
    Строит тепловую карту позиций гимнасток по площадке (групповой формат).

    Собирает центры скелетов всех гимнасток по кадрам, агрегирует в сетку
    width_cells × height_cells и сохраняет PNG. Возвращает stats или None.
    """
    if not frames_with_poses:
        return None

    centers = []
    for _, pose_list, _ in frames_with_poses:
        for landmarks in pose_list:
            centers.append(_skeleton_center(landmarks))

    if not centers:
        return None

    grid = np.zeros((height_cells, width_cells), dtype=float)
    for cx, cy in centers:
        col = min(width_cells - 1, max(0, int(cx * width_cells)))
        row = min(height_cells - 1, max(0, int(cy * height_cells)))
        grid[row, col] += 1

    total = float(grid.sum())
    if total <= 0:
        return None

    # Доля попаданий в «плотные» ячейки, где одновременно сидят >= 4 гимнастки
    dense_cells = grid >= 4
    overlap_ratio = float(grid[dense_cells].sum() / total) if dense_cells.any() else 0.0
    stats = {
        "peak_density": int(grid.max()),
        "overlap_ratio": round(overlap_ratio, 3),
    }

    # Рисуем тепловую карту: JET-палитра, пустые ячейки — тёмный фон.
    # Маска накладывается ДО resize, иначе размеры массива не совпадут.
    normalized = (grid / grid.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[normalized == 0] = (30, 30, 30)
    colored = cv2.resize(colored, (480, 270), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(heatmap_path, colored)
    return stats


def _open_video_with_rotation(video_path: str):
    """
    Открывает VideoCapture и учитывает rotation metadata вертикальных видео
    (смартфонная съемка). OpenCV игнорирует метаданные поворота: кадр приходит
    «лежащим» на 90/270° и детекция позы ломается. Возвращает (cap, rotate_code),
    где rotate_code — cv2.ROTATE_90_CLOCKWISE / ROTATE_180 / ROTATE_90_COUNTERCLOCKWISE
    или None, если поворот не нужен.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    try:
        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except (cv2.error, TypeError, ValueError):
        orientation = 0

    rotate_code = None
    if orientation in (90, -90):
        # -90 и 90 в OpenCV трактуются одинаково как 90° по часовой;
        # -180/180 не встречаются в CAP_PROP_ORIENTATION_META (только 0/90/180/270)
        rotate_code = cv2.ROTATE_90_CLOCKWISE if orientation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
    elif orientation == 180:
        rotate_code = cv2.ROTATE_180
    return cap, rotate_code


def analyze_video(video_path: str, exercise: str = "unknown", format_type: str = "individual") -> dict:
    """
    Анализирует видео с помощью MediaPipe PoseLandmarker (Tasks API):
    определяет наличие человека/группы, собирает статистику, вычисляет
    метрики углублённого анализа движений и формирует
    рекомендации по технике для выбранного упражнения.

    format_type: "individual" — одна гимнастка (самая крупная в кадре),
                 "group" — группа из GROUP_SIZE гимнасток.
    """
    cap, rotate_code = _open_video_with_rotation(video_path)
    if cap is None:
        return {"error": "Не удалось открыть видеофайл"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Анализируем примерно 2 кадра в секунду для экономии ресурсов
    frame_interval = max(1, int(fps / 2))

    frames_analyzed = 0
    pose_detections = 0
    # Замер латентности обработки кадра (мс): для контроля требований realtime
    frame_latencies_ms = []

    # Кадры с позами всех обнаруженных людей (для построения треков гимнасток)
    frames_with_poses = []
    # Кадры для трекинга булав (анализируем те же кадры, что и позу)
    frames_for_clubs = []
    # Трекер булав: использует кастомную ONNX-модель, если она есть, иначе HSV
    club_tracker = ClubTracker(model_path=CLUB_MODEL_PATH)

    landmarker = get_pose_landmarker()
    # Смещение таймстампов: инстанс PoseLandmarker singleton, и detect_for_video
    # требует монотонно растущие таймстампы между видео; каждому видео — свой диапазон
    with _pose_lock:
        ts_offset = _pose_ts_offset[0]
        _pose_ts_offset[0] += _VIDEO_TS_OFFSET_MS

    frame_idx_holder = [0]  # мутируемый счетчик кадров (int в замыкании не мутируется)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_holder[0] % frame_interval == 0:
            frame_start = time.perf_counter()
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_idx = frame_idx_holder[0]
            timestamp_ms = (
                ts_offset + int(frame_idx * 1000 / fps) if fps else ts_offset
            )
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Сохраняем кадр для трекинга булав (если выбрано упражнение с булавами)
            if exercise == "risk-buhavy":
                frames_for_clubs.append(frame)

            pose_list = list(results.pose_landmarks) if results.pose_landmarks else []
            # world-ландмарки: имя атрибута менялось между версиями MediaPipe
            # (0.10.x: pose_world_landmarks; в 1.1.x убран из результата)
            world_list = list(getattr(results, "world_landmarks", None) or getattr(results, "pose_world_landmarks", None) or [])
            if pose_list:
                pose_detections += 1
                frames_with_poses.append((frame, pose_list, world_list))

            # Латентность обработки кадра (детекция позы + булавы), мс
            frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
            frames_analyzed += 1

        frame_idx_holder[0] += 1

    cap.release()

    # Строим треки всех гимнасток и выбираем целевых
    all_tracks = track_athletes(frames_with_poses, fps)
    target_tracks = select_target_tracks(all_tracks, format_type)

    if not target_tracks:
        return {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "frames_analyzed": frames_analyzed,
            "pose_detections": pose_detections,
            "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
            "latency": _latency_stats(frame_latencies_ms),
            "movement": None,
            "athletes": [],
            "clubs": None,
            "feedback": {
                "overall_score": None,
                "errors": [
                    {"severity": "critical", "message": "Не удалось обнаружить гимнастку в кадре."}
                ],
                "recommendations": [
                    "Убедитесь, что спортсменка полностью в кадре и хорошо освещена."
                ],
                "summary": "Анализ невозможен: человек не обнаружен.",
            },
            "format": format_type,
        }

    # Для каждой целевой гимнастки вычисляем агрегированные метрики
    athlete_results = []
    for i, track in enumerate(target_tracks):
        movement = compute_deep_metrics(
            track.torso_angles,
            track.wrist_heights,
            track.com_positions,
            fps,
            knee_angles_left=track.knee_angles_left,
            knee_angles_right=track.knee_angles_right,
            leg_opening_angles=track.leg_opening_angles,
            hip_heights=track.hip_heights,
            signed_torso_angles=track.signed_torso_angles,
            ankle_heights=track.ankle_heights,
            ankle_y_left=track.ankle_y_left,
            ankle_y_right=track.ankle_y_right,
            arm_symmetries=track.arm_symmetries,
            rotation_angles_world=track.rotation_angles_world,
            wrist_y_left=track.wrist_y_left,
            wrist_y_right=track.wrist_y_right,
        )
        athlete_results.append({
            "index": i,
            "movement": movement,
            "frames_seen": track.frames_seen,
        })

    # Трекинг булав (только для упражнения с булавами)
    clubs = None
    if exercise == "risk-buhavy" and frames_for_clubs:
        trajectories = club_tracker.track(frames_for_clubs)

        # Для индивидуального формата связываем булавы с единственной гимнасткой
        if format_type == "individual" and target_tracks:
            wrist_positions = [target_tracks[0].wrist_positions]
            assigned = assign_clubs_to_athletes(trajectories, wrist_positions)
            if assigned:
                trajectories = assigned[0]
        # Для группового формата связываем булавы с каждой гимнасткой
        elif format_type == "group" and target_tracks:
            wrist_positions = [t.wrist_positions for t in target_tracks]
            assigned_groups = assign_clubs_to_athletes(trajectories, wrist_positions)
            # Сохраняем траектории по гимнасткам для дальнейшего анализа
            clubs_per_athlete = [analyze_club_flight(group, fps) for group in assigned_groups]
            clubs = {
                "format": "group",
                "athletes": clubs_per_athlete,
                "clubs_detected_total": len(trajectories),
            }
        else:
            clubs = analyze_club_flight(trajectories, fps)
    else:
        clubs = None

    # Командная обратная связь (только групповой формат): сопоставление событий
    # гимнасток по кадрам — рассинхрон старта, потеря ритма после падения булавы
    team_feedback = None
    if format_type == "group" and len(target_tracks) >= 2:
        clubs_per_athlete_for_team = None
        if isinstance(clubs, dict) and clubs.get("format") == "group":
            clubs_per_athlete_for_team = clubs.get("athletes")
        team_feedback = detect_team_feedback(target_tracks, fps, clubs_per_athlete_for_team)

    # Детекция ошибок для первой гимнастки (для обратной совместимости ответа)
    first_movement = athlete_results[0]["movement"] if athlete_results else {}
    if isinstance(clubs, dict) and clubs.get("format") == "group":
        feedback = detect_errors_and_recommendations(exercise, first_movement, None)
    else:
        feedback = detect_errors_and_recommendations(exercise, first_movement, clubs)

    # Для группового формата — отдельный feedback для каждой гимнастки
    athletes_output = []
    for res in athlete_results:
        athletes_output.append({
            **res,
            "feedback": detect_errors_and_recommendations(exercise, res["movement"], None),
        })

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames_analyzed": frames_analyzed,
        "pose_detections": pose_detections,
        "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
        "latency": _latency_stats(frame_latencies_ms),
        "movement": first_movement if first_movement else None,
        "athletes": athletes_output,
        "clubs": clubs,
        "feedback": feedback,
        "team_feedback": team_feedback,
        "format": format_type,
        # Кадры с позами (numpy-массивы): используются для создания отчёта
        # (скриншоты ошибок, тепловая карта) и удаляются до сериализации ответа
        "_frames_with_poses": frames_with_poses,
    }


def create_report(
    analysis: dict,
    video_path: str,
    frames_with_poses: list,
    fps: float,
    reports_dir: str = storage.REPORTS_DIR,
) -> dict:
    """
    Сохраняет отчёт о тренировке: видео, метаданные, кадры ошибок
    и тепловую карту (для группового формата). Возвращает документ отчёта.

    Синхронная функция: выполняется в threadpool FastAPI вместе с analyze_video,
    чтобы не блокировать event loop (см. upload_video).
    """
    report_id = storage.create_report_id()
    report_dir = storage.get_report_dir(report_id, reports_dir)
    frames_dir = os.path.join(report_dir, "frames")

    analysis["report_id"] = report_id

    # Кадры моментов ошибок (для карточки PDF)
    error_frames = _capture_error_frames(
        video_path, fps, frames_with_poses,
        analysis.get("feedback") or {}, frames_dir,
    )
    analysis["error_frames"] = {
        str(i): {
            "timestamp": f.get("timestamp"),
            "landmarks": f.get("landmarks"),
        }
        for i, f in error_frames.items()
    }

    # Тепловая карта (групповой формат): где гимнастки сбиваются в кучу
    heatmap_stats = None
    if analysis.get("format") == "group":
        heatmap_path = os.path.join(frames_dir, "heatmap.png")
        heatmap_stats = _build_heatmap(frames_with_poses, heatmap_path)
        if heatmap_stats:
            analysis["heatmap"] = heatmap_path
            analysis["heatmap_stats"] = heatmap_stats

    doc = storage.save_report(report_id, analysis, video_path, reports_dir)
    if heatmap_stats:
        doc["analysis"]["heatmap"] = os.path.join(
            storage.get_report_dir(report_id, reports_dir), "frames", "heatmap.png"
        )
    doc["error_frames_full"] = error_frames
    storage.cleanup_old_reports(reports_dir)
    return doc


@app.post("/upload")
def upload_video(
    request: Request,
    video_file: UploadFile = File(...),
    exercise: str = Form("unknown"),
    format_type: str = Form("individual"),
    _api_key: str = Depends(require_api_key),
):
    """
    Загрузка видео и синхронный анализ.

    Обычный def (не async): FastAPI выполняет эндпоинт в threadpool, поэтому
    CPU-интенсивный analyze_video не блокирует event loop и конкурентные запросы.
    """
    filename = video_file.filename or "video.mp4"

    # 1. Валидация файла (fast-path по Content-Length). Ошибка клиента ->
    # корректный HTTP-статус вместо 200 с {"error": ...} в теле.
    try:
        validate_video_file(filename, video_file.size or 0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Сохранение файла во временную директорию (с жестким лимитом размера:
    #    video_file.size может быть неизвестен при стриминговой загрузке)
    unique_name = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        written = 0
        with open(save_path, "wb") as buffer:
            while chunk := video_file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_FILE_SIZE_BYTES:
                    break
                buffer.write(chunk)

        if written > MAX_FILE_SIZE_BYTES:
            os.remove(save_path)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Файл слишком большой. Максимальный размер — "
                    f"{MAX_FILE_SIZE_MB} МБ."
                ),
            )

        # 3. Анализ видео
        analysis = analyze_video(save_path, exercise, format_type)

        # Логирование латентности: контроль бюджета обработки кадра
        latency = analysis.get("latency")
        if latency:
            print(
                f"[latency] file={unique_name} exercise={exercise} format={format_type} "
                f"avg={latency['avg_ms']}ms p95={latency['p95_ms']}ms "
                f"max={latency['max_ms']}ms samples={latency['samples']}"
            )

        # 4. Отчёт: видео сохраняется, кадры ошибок и тепловая карта строятся,
        #    если анализ был успешным; иначе видео просто удаляется
        report_id = None
        if not analysis.get("error") and analysis.get("feedback"):
            # Кадры ошибок снимаются с тех же кадров, что использовались в анализе
            frames_with_poses = analysis.pop("_frames_with_poses", [])
            fps = analysis.get("fps") or 30.0
            doc = create_report(analysis, save_path, frames_with_poses, fps)
            report_id = doc.get("report_id")
            analysis = doc["analysis"]
            # Абсолютная ссылка на плеер — используется в PDF для таймкодов
            analysis["video_url"] = str(request.base_url).rstrip("/") + f"/video-player/{report_id}"
            storage.update_report_analysis(report_id, analysis)
        else:
            if os.path.exists(save_path):
                os.remove(save_path)

        return {
            "message": "Видео получено",
            "file": unique_name,
            "exercise": exercise,
            "format": format_type,
            "report_id": report_id,
            "analysis": analysis,
        }
    finally:
        # Временный файл мог быть перемещён в отчёт; удаляем только если остался
        if os.path.exists(save_path):
            os.remove(save_path)


@app.get("/stats/{exercise}")
async def get_stats(exercise: str, format_type: str = "individual", k: int = 3):
    """
    Тренд критических ошибок по данным детектора (pre/post-контроль).

    Сравнивает среднее число critical-ошибок в первых k и последних k сессиях
    по этому упражнению и формату. Возвращает None-тренд, если сессий < 2*k.
    """
    trend = storage.error_trend(exercise, format_type, k=k)
    return {
        "exercise": exercise,
        "format": format_type,
        "trend": trend,
        "note": (
            "Тренд отражает снижение срабатываний детектора ошибок. "
            "Валидация правил против экспертизы тренера не проводилась."
        ),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health-check для деплой-проверок и мониторинга: статус сервиса
    и готовность модели PoseLandmarker (наличие + совпадение pin-хеша).
    """
    return {
        "status": "ok" if POSE_MODEL_OK else "degraded",
        "pose_model_ok": POSE_MODEL_OK,
    }
