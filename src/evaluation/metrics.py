from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.evaluation.spider_eval_utils import (
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match,
    rebuild_sql_col,
    rebuild_sql_val,
)
from src.evaluation.spider_process_sql import Schema, get_schema, get_sql
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
    desc: str = "execution accuracy",
) -> Tuple[float, List[bool]]:
    correct_flags = []
    for pred in tqdm(predictions, desc=desc, unit="q"):
        db_path = db_paths.get(pred["db_id"], "")
        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)
        correct_flags.append(
            pred_err is None and gold_err is None and matcher(pred_result, gold_result)
        )

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def execution_accuracy(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    return _ex_with_matcher(
        predictions, db_paths, timeout, results_match, desc="ex_strict"
    )


def execution_accuracy_permuted(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    return _ex_with_matcher(
        predictions, db_paths, timeout, results_match_permuted, desc="ex_permuted"
    )


def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    correct_flags = [
        normalize_sql(pred["predicted_sql"]) == normalize_sql(pred["gold_sql"])
        for pred in tqdm(predictions, desc="exact_match", unit="q")
    ]
    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def valid_sql_rate(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    valid_flags = []
    for pred in tqdm(predictions, desc="valid_sql_rate", unit="q"):
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


def compute_spider_component_metrics(
    predictions: List[dict],
    db_dir: str,
    tables_json: str,
) -> dict:
    spider_preds = [p for p in predictions if p.get("source") == "spider"]
    if not spider_preds:
        return {}

    kmaps = build_foreign_key_map_from_json(tables_json)

    levels = ["easy", "medium", "hard", "extra", "all"]
    partial_types = [
        "select",
        "select(no AGG)",
        "where",
        "where(no OP)",
        "group(no Having)",
        "group",
        "order",
        "and/or",
        "IUEN",
        "keywords",
    ]

    scores: dict = {}
    for level in levels:
        scores[level] = {"count": 0, "exact": 0.0, "exec": 0.0, "partial": {}}
        for t in partial_types:
            scores[level]["partial"][t] = {
                "acc": 0.0,
                "rec": 0.0,
                "f1": 0.0,
                "acc_count": 0,
                "rec_count": 0,
            }

    evaluator = Evaluator()
    for pred in tqdm(spider_preds, desc="spider_official", unit="q"):
        db_name = pred["db_id"]
        db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
        schema = Schema(get_schema(db_path))

        g_sql = get_sql(schema, pred["gold_sql"])
        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]["count"] += 1
        scores["all"]["count"] += 1

        try:
            p_sql = get_sql(schema, pred["predicted_sql"])
        except Exception:
            p_sql = {
                "except": None,
                "intersect": None,
                "union": None,
                "from": {"conds": [], "table_units": []},
                "groupBy": [],
                "having": [],
                "orderBy": [],
                "select": [False, []],
                "where": [],
                "limit": None,
            }

        kmap = kmaps[db_name]
        g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_col(g_valid, rebuild_sql_val(g_sql), kmap)
        p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_col(p_valid, rebuild_sql_val(p_sql), kmap)

        exec_score = eval_exec_match(
            db_path, pred["predicted_sql"], pred["gold_sql"], p_sql, g_sql
        )
        if exec_score:
            scores[hardness]["exec"] += 1.0
            scores["all"]["exec"] += 1.0

        exact_score = evaluator.eval_exact_match(p_sql, g_sql)
        partial_scores = evaluator.partial_scores
        scores[hardness]["exact"] += exact_score
        scores["all"]["exact"] += exact_score

        for t in partial_types:
            for level in (hardness, "all"):
                if partial_scores[t]["pred_total"] > 0:
                    scores[level]["partial"][t]["acc"] += partial_scores[t]["acc"]
                    scores[level]["partial"][t]["acc_count"] += 1
                if partial_scores[t]["label_total"] > 0:
                    scores[level]["partial"][t]["rec"] += partial_scores[t]["rec"]
                    scores[level]["partial"][t]["rec_count"] += 1
                scores[level]["partial"][t]["f1"] += partial_scores[t]["f1"]

    for level in levels:
        cnt = scores[level]["count"]
        if cnt == 0:
            continue
        scores[level]["exec"] /= cnt
        scores[level]["exact"] /= cnt
        for t in partial_types:
            ac = scores[level]["partial"][t]["acc_count"]
            rc = scores[level]["partial"][t]["rec_count"]
            scores[level]["partial"][t]["acc"] = (
                scores[level]["partial"][t]["acc"] / ac if ac > 0 else 0.0
            )
            scores[level]["partial"][t]["rec"] = (
                scores[level]["partial"][t]["rec"] / rc if rc > 0 else 0.0
            )
            a = scores[level]["partial"][t]["acc"]
            r = scores[level]["partial"][t]["rec"]
            scores[level]["partial"][t]["f1"] = (
                2.0 * a * r / (a + r) if (a + r) > 0 else 0.0
            )

    return scores


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
    spider_db_dir: Optional[str] = None,
    spider_tables_json: Optional[str] = None,
) -> dict:
    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    ex_perm, ex_perm_flags = execution_accuracy_permuted(predictions, db_paths, timeout)
    em, em_flags = exact_match(predictions)
    vsr, vsr_flags = valid_sql_rate(predictions, db_paths, timeout)

    result = {
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

    if spider_db_dir and spider_tables_json:
        result["spider_official"] = compute_spider_component_metrics(
            predictions, spider_db_dir, spider_tables_json
        )

    return result
