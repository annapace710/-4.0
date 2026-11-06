"""Роуты отчётов (Шаг 6в рефакторинга).

Эндпоинты /reports/{report_id}/pdf, /reports/{report_id}/video,
/reports/{report_id}/frames/{frame_name} и /video-player/{report_id}
перенесены из src/app/main.py без изменения логики, вместе с приватными
хелперами _get_report_or_none и _save_video_thumbnail.

Особенность (важно для тестов): модуль storage резолвится ЛЕНИВО, в момент
запроса, через корневую обёртку storage.py — тесты патчат storage.REPORTS_DIR
на корневой обёртке (import storage → src.app.storage → src.app.reports.storage),
и ленивое чтение атрибута в момент запроса всегда видит актуальное значение.
"""

import os

import cv2

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

import storage
import report_generator

router = APIRouter()


def _get_report_or_none(report_id: str):
    """Загружает отчёт или возвращает None (для FastAPI 404)."""
    return storage.get_report(report_id)


def _save_video_thumbnail(video_path: str, thumb_path: str) -> None:
    """Сохраняет первый кадр видео как PNG-миниатюру (кэшируется на диске)."""
    if os.path.exists(thumb_path) or not video_path or not os.path.exists(video_path):
        return
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if ret:
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            cv2.imwrite(thumb_path, frame)
    finally:
        cap.release()


@router.get("/reports/{report_id}/pdf")
async def get_report_pdf(report_id: str):
    """Возвращает PDF-карточку тренировки («Сохранить отчёт»)."""
    doc = _get_report_or_none(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")

    analysis = doc.get("analysis") or {}
    report_dir = storage.get_report_dir(report_id)
    frames_dir = os.path.join(report_dir, "frames")

    # Миниатюра видео: первый кадр, сохранённый во время анализа
    thumb_path = os.path.join(frames_dir, "thumb.png")
    if not os.path.exists(thumb_path):
        video_full_path = os.path.join(report_dir, doc.get("video_file") or "")
        _save_video_thumbnail(video_full_path, thumb_path)

    # Кадры ошибок (PNG + координаты скелета для обводки)
    saved_frames = analysis.get("error_frames") or {}
    error_frames = {}
    for key, f in saved_frames.items():
        png_path = os.path.join(frames_dir, f"error_{key}.png")
        if os.path.exists(png_path):
            error_frames[int(key)] = {
                "path": png_path,
                "timestamp": f.get("timestamp"),
                "landmarks": f.get("landmarks"),
            }

    # Предыдущий отчёт по тому же упражнению и формату — для блока динамики
    previous = storage.find_previous_report(
        doc.get("exercise"), doc.get("format"), doc.get("created_at") or ""
    )

    pdf_path = os.path.join(report_dir, "report.pdf")
    report_generator.build_report_pdf(
        doc,
        video_thumb_path=thumb_path if os.path.exists(thumb_path) else None,
        error_frames=error_frames,
        previous_meta=previous,
        output_path=pdf_path,
    )

    if not os.path.exists(pdf_path):
        return {"error": "Не удалось сформировать PDF"}

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"training-report-{report_id[:8]}.pdf",
    )


@router.get("/reports/{report_id}/video")
async def get_report_video(report_id: str, request: Request, t: float = None):
    """
    Раздаёт видео отчёта. Параметр t (секунды) используется фронтенд-плеером
    для перескока к моменту ошибки по кликабельным таймкодам из PDF.
    """
    doc = _get_report_or_none(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    video_path = os.path.join(storage.get_report_dir(report_id), doc.get("video_file") or "")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Видео отчёта не найдено")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )


@router.get("/reports/{report_id}/frames/{frame_name}")
async def get_report_frame(report_id: str, frame_name: str):
    """Раздаёт PNG-кадры отчёта (скриншоты ошибок, тепловая карта)."""
    if not storage.is_valid_report_id(report_id):
        return {"error": "Некорректный report_id"}
    # Имя файла строго ограничено (защита от path traversal)
    allowed = {"error_0.png", "error_1.png", "error_2.png", "error_3.png", "error_4.png", "heatmap.png", "thumb.png"}
    if frame_name not in allowed:
        return {"error": "Кадр не найден"}
    frame_path = os.path.join(storage.get_report_dir(report_id), "frames", frame_name)
    if not os.path.exists(frame_path):
        return {"error": "Кадр не найден"}
    return FileResponse(frame_path, media_type="image/png")


@router.get("/video-player/{report_id}")
async def video_player(report_id: str, t: float = 0.0):
    """HTML-плеер видео с автопереходом к таймкоду ошибки (?t=<секунды>)."""
    doc = _get_report_or_none(report_id)
    if doc is None:
        return HTMLResponse("<h1>Отчёт не найден</h1>", status_code=404)
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>Видео тренировки</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
  <h2 style="padding:12px">Видео тренировки — переход к моменту ошибки</h2>
  <video id="player" width="960" controls autoplay>
    <source src="/reports/{report_id}/video" type="video/mp4">
  </video>
  <p><a href="/reports/{report_id}/pdf" style="color:#7ab7ff">Скачать PDF-карточку</a></p>
  <script>
    const player = document.getElementById("player");
    player.currentTime = {float(t)};
  </script>
  </body>
</html>"""
    return HTMLResponse(html)
