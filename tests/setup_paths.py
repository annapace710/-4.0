"""
Настройка путей для тестов (Шаг 4 рефакторинга).

Тесты лежат в tests/, а импортируемые модули (main, storage, agent) —
в корне проекта. Этот модуль добавляет корень проекта в sys.path.

Важно: имя conftest.py не используется сознательно — это соглашение pytest
(файл, подхватываемый автоматически), а тесты проекта работают на unittest,
где модуль импортируется явно.

Использование (первой строкой в тестовом файле, до остальных импортов):

    import setup_paths  # noqa: F401
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
