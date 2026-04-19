from __future__ import annotations

from typing import Dict, List, Tuple

from src.evaluation.sql_executor import execute_sql, results_match


def normalize_sql(sql: str) -> str:
    sql = sql.lower().strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return " ".join(sql.split())


def execution_accuracy(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    correct_flags = []
    for pred in predictions:
        db_path = db_paths.get(pred["db_id"], "")
        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)
        correct_flags.append(
            pred_err is None
            and gold_err is None
            and results_match(pred_result, gold_result)
        )

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    correct_flags = [
        normalize_sql(pred["predicted_sql"]) == normalize_sql(pred["gold_sql"])
        for pred in predictions
    ]
    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def valid_sql_rate(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    valid_flags = []
    for pred in predictions:
        db_path = db_paths.get(pred["db_id"], "")
        _, err = execute_sql(db_path, pred["predicted_sql"], timeout)
        valid_flags.append(err is None)

    score = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    return score, valid_flags


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> dict:
    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    em, _ = exact_match(predictions)
    vsr, _ = valid_sql_rate(predictions, db_paths, timeout)

    return {
        "ex": ex,
        "em": em,
        "vsr": vsr,
        "n_examples": len(predictions),
        "n_correct_ex": sum(ex_flags),
    }
