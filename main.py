"""
Тонкая обёртка для обратной совместимости после Шага 1 рефакторинга.

Код приложения перенесён в src/app/main.py (см. комментарий в файле).
Этот модуль реэкспортирует всё, на что ссылаются внешние точки входа:

- uvicorn main:app (Dockerfile, docker-compose, локальный запуск);
- test_main.py: 21 имя из списка `from main import (...)` (тесты лежат в `tests/`);
- test_agent.py: переключён на `import src.app.main as main` (патчи атрибутов
  rate-limit/LLM должны попадать в модуль эндпоинтов; с Шага 3 лимит агента
  патчится в src.app.auth);
- scripts/download_pose_model.py: MODEL_PATH, MODEL_SHA256_PATH, _file_sha256,
  verify_pose_model.

Важно: эндпоинты читают get_llm_client из модуля src.app.main,
поэтому тесты патчат именно его; настоящие привязки в этом файле нужны
scripts/download_pose_model.py и точкам входа uvicorn.
"""

from src.app.auth import AGENT_RATE_LIMIT_PER_MINUTE  # noqa: F401  (Шаг 3: auth вынесен)
from src.app.main import (  # noqa: F401
    AGENT_MAX_HISTORY_MESSAGES,
    AthleteTrack,
    MAX_FILE_SIZE_BYTES,
    app,
    compute_ankle_height,
    compute_ankle_y,
    compute_arm_symmetry,
    compute_deep_metrics,
    compute_hip_height,
    compute_knee_angle,
    compute_leg_opening_angle,
    compute_rotation_angle_world,
    compute_torso_tilt_signed,
    detect_errors_and_recommendations,
    detect_team_feedback,
    get_llm_client,
    track_athletes,
    select_target_tracks,
    _build_heatmap,
    _file_sha256,
    _latency_stats,
    _movement_start_frame,
    _pause_frames,
    _suggested_drill,
    MODEL_PATH,
    MODEL_SHA256_PATH,
    validate_video_file,
    verify_pose_model,
)
