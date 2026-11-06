"""
Генерация PDF-карточки тренировки — обёртка совместимости (Шаг 6б рефакторинга).

Реализация перенесена в src/app/reports/pdf.py. Этот модуль оставлен,
чтобы существующие импорты в src/app/main.py (`import report_generator`)
продолжали работать без изменений.
"""

from .reports.pdf import *  # noqa: F401,F403
from .reports.pdf import (  # noqa: F401
    COMPARABLE_METRICS,
    EXERCISE_TITLES,
    build_report_pdf,
    heatmap_notes,
)
