"""Домен reports: хранение и PDF-отчёты.

Шаг 6б рефакторинга: выделен из storage.py и report_generator.py.
"""

from .storage import (
    REPORTS_DIR,
    RETENTION_DAYS,
    cleanup_old_reports,
    create_report_id,
    error_trend,
    find_previous_report,
    get_report,
    get_report_dir,
    is_valid_report_id,
    save_report,
    update_report_analysis,
)
from .pdf import build_report_pdf

__all__ = [
    "REPORTS_DIR",
    "RETENTION_DAYS",
    "build_report_pdf",
    "cleanup_old_reports",
    "create_report_id",
    "error_trend",
    "find_previous_report",
    "get_report",
    "get_report_dir",
    "is_valid_report_id",
    "save_report",
    "update_report_analysis",
]
