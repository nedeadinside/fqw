from __future__ import annotations

import re
import sqlite3
import threading
from collections import Counter
from typing import Any, List, Optional, Tuple

Row = Tuple[Any, ...]
Rows = Optional[List[Row]]
ExecutionResult = Tuple[Rows, Optional[str]]


class _QueryRunner:
    def __init__(self, db_path: str, sql: str):
        self.db_path = db_path
        self.sql = sql
        self.conn: Optional[sqlite3.Connection] = None
        self.result: Rows = None
        self.error: Optional[str] = None
        self.done = threading.Event()

    def run(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.text_factory = lambda b: b.decode(errors="replace")
            try:
                cursor = self.conn.cursor()
                cursor.execute(self.sql)
                self.result = list(cursor.fetchall())
            finally:
                try:
                    self.conn.close()
                except Exception:
                    pass
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done.set()


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

    runner = _QueryRunner(db_path, sql_stripped)
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    finished = runner.done.wait(timeout=timeout)

    if not finished:
        if runner.conn is not None:
            try:
                runner.conn.interrupt()
            except Exception:
                pass
        thread.join(timeout=2.0)
        return None, f"execution timeout ({timeout}s)"

    if runner.error is not None:
        return None, runner.error
    return runner.result, None


_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


def has_order_by(sql: str) -> bool:
    stripped = _STRING_LITERAL_RE.sub("", sql)
    return bool(_ORDER_BY_RE.search(stripped))


def match_ordered(pred: Rows, gold: Rows) -> bool:
    if pred is None or gold is None:
        return False
    return pred == gold


def match_multiset(pred: Rows, gold: Rows) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return Counter(pred) == Counter(gold)
    except TypeError:
        return sorted(pred, key=repr) == sorted(gold, key=repr)


def results_match(pred: Rows, gold: Rows, gold_sql: str) -> bool:
    if has_order_by(gold_sql):
        return match_ordered(pred, gold)
    return match_multiset(pred, gold)


def results_match_permuted(pred: Rows, gold: Rows) -> bool:
    return match_multiset(pred, gold)
