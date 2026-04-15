from __future__ import annotations

import sqlite3
import threading
from typing import FrozenSet, Optional, Tuple

ResultSet = Optional[FrozenSet[tuple]]
ExecutionResult = Tuple[ResultSet, Optional[str]]


class QueryThread(threading.Thread):
    def __init__(self, db_path: str, sql: str):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.sql = sql
        self.result: ResultSet = None
        self.error: Optional[str] = None

    def run(self):
        try:
            connection = sqlite3.connect(self.db_path)
            connection.text_factory = lambda value: value.decode(errors="replace")
            cursor = connection.cursor()
            cursor.execute(self.sql)
            rows = cursor.fetchall()
            connection.close()
            self.result = frozenset(
                tuple(str(cell).strip() if cell is not None else None for cell in row)
                for row in rows
            )
        except Exception as error:
            self.error = str(error)


def execute_sql(db_path: str, sql: str, timeout: float = 30.0) -> ExecutionResult:
    if not db_path:
        return None, "db_path не задан"

    sql_text = sql.strip()
    if not sql_text:
        return None, "Пустой SQL-запрос"

    thread = QueryThread(db_path, sql_text)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, f"Таймаут выполнения ({timeout}s)"

    if thread.error is not None:
        return None, thread.error

    return thread.result, None


def results_match(result_pred: ResultSet, result_gold: ResultSet) -> bool:
    if result_pred is None or result_gold is None:
        return False
    return result_pred == result_gold
