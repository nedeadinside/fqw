from __future__ import annotations

from typing import Dict, List, Tuple

from src.evaluation.sql_executor import (
    execute_sql,
    results_match,
    results_match_permuted,
)


def normalize_sql(sql: str) -> str:
    sql = sql.lower().strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return " ".join(sql.split())


def _ex_with_matcher(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float,
    matcher,
) -> Tuple[float, List[bool]]:
    correct_flags = []
    for pred in predictions:
        db_path = db_paths.get(pred["db_id"], "")
        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)
        correct_flags.append(
            pred_err is None
            and gold_err is None
            and matcher(pred_result, gold_result)
        )

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def execution_accuracy(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    return _ex_with_matcher(predictions, db_paths, timeout, results_match)


def execution_accuracy_permuted(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    return _ex_with_matcher(predictions, db_paths, timeout, results_match_permuted)


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


def _breakdown_by_source(
    predictions: List[dict],
    flags: List[bool],
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[bool]] = {}
    for pred, flag in zip(predictions, flags):
        src = pred.get("source", "unknown")
        buckets.setdefault(src, []).append(flag)
    return {
        src: {
            "score": sum(fs) / len(fs) if fs else 0.0,
            "n": len(fs),
            "n_correct": sum(fs),
        }
        for src, fs in buckets.items()
    }


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> dict:
    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    ex_perm, ex_perm_flags = execution_accuracy_permuted(predictions, db_paths, timeout)
    em, em_flags = exact_match(predictions)
    vsr, vsr_flags = valid_sql_rate(predictions, db_paths, timeout)

    return {
        "ex_permuted": ex_perm,
        "ex_strict": ex,
        "em": em,
        "vsr": vsr,
        "n_examples": len(predictions),
        "n_correct_ex_permuted": sum(ex_perm_flags),
        "n_correct_ex_strict": sum(ex_flags),
        "by_source": {
            "ex_permuted": _breakdown_by_source(predictions, ex_perm_flags),
            "ex_strict": _breakdown_by_source(predictions, ex_flags),
            "em": _breakdown_by_source(predictions, em_flags),
            "vsr": _breakdown_by_source(predictions, vsr_flags),
        },
    }
