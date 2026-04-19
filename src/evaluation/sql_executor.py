from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from typing import FrozenSet, Optional, Tuple

ResultSet = Optional[FrozenSet[tuple]]
ExecutionResult = Tuple[ResultSet, Optional[str]]


class _QueryThread(threading.Thread):
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
    if not db_path:
        return None, "db_path is empty"

    sql_stripped = sql.strip()
    if not sql_stripped:
        return None, "empty SQL query"

    thread = _QueryThread(db_path, sql_stripped)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, f"execution timeout ({timeout}s)"
    if thread.error is not None:
        return None, thread.error
    return thread.result, None


def results_match(result_pred: ResultSet, result_gold: ResultSet) -> bool:
    if result_pred is None or result_gold is None:
        return False
    return result_pred == result_gold


def _row_multiset(row: tuple) -> frozenset:
    return frozenset(Counter(row).items())


def results_match_permuted(result_pred: ResultSet, result_gold: ResultSet) -> bool:
    if result_pred is None or result_gold is None:
        return False
    pred_norm = Counter(_row_multiset(r) for r in result_pred)
    gold_norm = Counter(_row_multiset(r) for r in result_gold)
    return pred_norm == gold_norm
