"""
Безопасное выполнение SQL-запросов на SQLite базах данных.

Особенности:
    - Таймаут через threading (signal.alarm не работает в потоках)
    - Только SELECT-запросы (защита от DDL/DML на тестовых БД)
    - Сравнение результатов как frozenset кортежей (порядок строк не важен)
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional, Tuple, FrozenSet


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------

ResultSet = Optional[FrozenSet[tuple]]
ExecutionResult = Tuple[ResultSet, Optional[str]]


# ---------------------------------------------------------------------------
# Исполнитель с таймаутом
# ---------------------------------------------------------------------------

class _QueryThread(threading.Thread):
    """Выполняет SQL-запрос в отдельном потоке для поддержки таймаута."""

    def __init__(self, db_path: str, sql: str):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.sql = sql
        self.result: ResultSet = None
        self.error: Optional[str] = None

    def run(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = lambda b: b.decode(errors="replace")
            cursor = conn.cursor()
            cursor.execute(self.sql)
            rows = cursor.fetchall()
            conn.close()
            # Нормализуем: строки в frozenset, значения приводим к str для
            # корректного сравнения NUMERIC vs TEXT (SQLite duck typing)
            self.result = frozenset(
                tuple(str(v).strip() if v is not None else None for v in row)
                for row in rows
            )
        except Exception as exc:
            self.error = str(exc)


def execute_sql(
    db_path: str,
    sql: str,
    timeout: float = 30.0,
) -> ExecutionResult:
    """Выполняет SQL-запрос на SQLite БД с таймаутом.

    Args:
        db_path: путь к .sqlite файлу
        sql:     SQL-запрос (только SELECT)
        timeout: максимальное время выполнения в секундах

    Returns:
        (result, error):
            result — frozenset кортежей результирующих строк, или None при ошибке
            error  — строка с описанием ошибки, или None при успехе
    """
    if not db_path:
        return None, "db_path не задан"

    sql_stripped = sql.strip()
    if not sql_stripped:
        return None, "Пустой SQL-запрос"

    thread = _QueryThread(db_path, sql_stripped)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, f"Таймаут выполнения ({timeout}s)"

    if thread.error is not None:
        return None, thread.error

    return thread.result, None


# ---------------------------------------------------------------------------
# Утилиты для сравнения результатов
# ---------------------------------------------------------------------------

def results_match(
    result_pred: ResultSet,
    result_gold: ResultSet,
) -> bool:
    """Сравнивает два result set.

    Оба должны быть не None и совпадать как множества кортежей.
    None-результат (ошибка исполнения) никогда не равен другому результату.
    """
    if result_pred is None or result_gold is None:
        return False
    return result_pred == result_gold
