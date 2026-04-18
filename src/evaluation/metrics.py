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
        db_id = pred["db_id"]
        db_path = db_paths.get(db_id, "")

        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)

        is_correct = (
            pred_err is None
            and gold_err is None
            and results_match(pred_result, gold_result)
        )
        correct_flags.append(is_correct)

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    correct_flags = []
    for pred in predictions:
        norm_pred = normalize_sql(pred["predicted_sql"])
        norm_gold = normalize_sql(pred["gold_sql"])
        correct_flags.append(norm_pred == norm_gold)

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def valid_sql_rate(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    valid_flags = []
    for pred in predictions:
        db_id = pred["db_id"]
        db_path = db_paths.get(db_id, "")
        _, err = execute_sql(db_path, pred["predicted_sql"], timeout)
        valid_flags.append(err is None)

    score = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    return score, valid_flags


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> dict:
    print(f"[metrics] Вычисление метрик для {len(predictions)} примеров...")

    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    print(f"  EX  = {ex:.4f}")

    em, _ = exact_match(predictions)
    print(f"  EM  = {em:.4f}")

    vsr, _ = valid_sql_rate(predictions, db_paths, timeout)
    print(f"  VSR = {vsr:.4f}")

    results = {
        "ex": ex,
        "em": em,
        "vsr": vsr,
        "n_examples": len(predictions),
        "n_correct_ex": sum(ex_flags),
    }

    return results
