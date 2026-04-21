from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.evaluation.logging_utils import (
    close_metrics_error_log,
    configure_metrics_error_log,
    log_metric_error,
)
from src.evaluation.spider_eval_utils import (
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match_from_rows,
    rebuild_sql_col,
    rebuild_sql_val,
)
from src.evaluation.spider_process_sql import Schema, get_schema, get_sql
from src.evaluation.sql_executor import (
    ExecutionResult,
    Rows,
    execute_sql,
    results_match,
    results_match_permuted,
)

ExecCache = Dict[Tuple[str, str], ExecutionResult]


_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


def normalize_sql(sql: str) -> str:
    placeholders: List[str] = []

    def _stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    masked = _STRING_LITERAL_RE.sub(_stash, sql)
    masked = masked.lower().strip()
    if masked.endswith(";"):
        masked = masked[:-1].rstrip()
    masked = " ".join(masked.split())

    def _restore(match: re.Match) -> str:
        return placeholders[int(match.group(1))]

    return re.sub(r"\x00(\d+)\x00", _restore, masked)


def _cache_key(db_id: str, sql: str) -> Tuple[str, str]:
    return db_id, (sql or "").strip()


def _run_cached(
    cache: ExecCache,
    db_id: str,
    db_path: str,
    sql: str,
    timeout: float,
    error_stage: str,
    error_kind: str,
) -> ExecutionResult:
    key = _cache_key(db_id, sql)
    if key in cache:
        return cache[key]
    result, err = execute_sql(db_path, sql, timeout)
    cache[key] = (result, err)
    if err is not None:
        log_metric_error(
            "stage=%s | kind=%s | db_id=%s | error=%s | sql=%s",
            error_stage,
            error_kind,
            db_id,
            err,
            sql,
        )
    return cache[key]


def _build_exec_cache(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float,
) -> ExecCache:
    cache: ExecCache = {}
    for pred in tqdm(predictions, desc="exec_sql", unit="q"):
        db_id = pred.get("db_id", "")
        db_path = db_paths.get(db_id, "")
        _run_cached(
            cache, db_id, db_path, pred["predicted_sql"], timeout,
            error_stage="exec", error_kind="pred_sql_exec_error",
        )
        _run_cached(
            cache, db_id, db_path, pred["gold_sql"], timeout,
            error_stage="exec", error_kind="gold_sql_exec_error",
        )
    return cache


def _pair_from_cache(cache: ExecCache, pred: dict) -> Tuple[Rows, Optional[str], Rows, Optional[str]]:
    db_id = pred.get("db_id", "")
    p_rows, p_err = cache.get(_cache_key(db_id, pred["predicted_sql"]), (None, "missing"))
    g_rows, g_err = cache.get(_cache_key(db_id, pred["gold_sql"]), (None, "missing"))
    return p_rows, p_err, g_rows, g_err


def execution_accuracy(
    predictions: List[dict],
    cache: ExecCache,
) -> Tuple[float, List[bool]]:
    flags: List[bool] = []
    for pred in predictions:
        p_rows, p_err, g_rows, g_err = _pair_from_cache(cache, pred)
        ok = p_err is None and g_err is None and results_match(p_rows, g_rows, pred["gold_sql"])
        flags.append(ok)
    score = sum(flags) / len(flags) if flags else 0.0
    return score, flags


def execution_accuracy_permuted(
    predictions: List[dict],
    cache: ExecCache,
) -> Tuple[float, List[bool]]:
    flags: List[bool] = []
    for pred in predictions:
        p_rows, p_err, g_rows, g_err = _pair_from_cache(cache, pred)
        ok = p_err is None and g_err is None and results_match_permuted(p_rows, g_rows)
        flags.append(ok)
    score = sum(flags) / len(flags) if flags else 0.0
    return score, flags


def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    correct_flags = [
        normalize_sql(pred["predicted_sql"]) == normalize_sql(pred["gold_sql"])
        for pred in tqdm(predictions, desc="exact_match", unit="q")
    ]
    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def valid_sql_rate(
    predictions: List[dict],
    cache: ExecCache,
) -> Tuple[float, List[bool]]:
    flags: List[bool] = []
    for pred in predictions:
        db_id = pred.get("db_id", "")
        _, err = cache.get(_cache_key(db_id, pred["predicted_sql"]), (None, "missing"))
        flags.append(err is None)
    score = sum(flags) / len(flags) if flags else 0.0
    return score, flags


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


def _load_spider_fk_maps(tables_json: str) -> tuple[Dict[str, dict], List[str]]:
    primary = Path(tables_json)
    candidates = [primary]

    test_tables = primary.with_name("test_tables.json")
    if test_tables not in candidates and test_tables.exists():
        candidates.append(test_tables)

    kmaps: Dict[str, dict] = {}
    loaded_files: List[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        kmaps.update(build_foreign_key_map_from_json(str(candidate)))
        loaded_files.append(str(candidate))

    if not kmaps:
        raise FileNotFoundError(
            f"No Spider table metadata files found. Expected at least: {primary}"
        )

    return kmaps, loaded_files


def compute_spider_component_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    tables_json: str,
    cache: ExecCache,
) -> dict:
    spider_preds = [p for p in predictions if p.get("source") == "spider"]
    if not spider_preds:
        return {}

    kmaps, loaded_table_files = _load_spider_fk_maps(tables_json)

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

    schema_cache: Dict[str, Schema] = {}
    evaluator = Evaluator()
    for pred in tqdm(spider_preds, desc="spider_official", unit="q"):
        db_name = pred["db_id"]
        db_path = db_paths.get(db_name, "")
        if not db_path:
            raise FileNotFoundError(
                f"SQLite file for db_id '{db_name}' was not found in configured database directories"
            )
        schema = schema_cache.get(db_name)
        if schema is None:
            schema = Schema(get_schema(db_path))
            schema_cache[db_name] = schema

        g_sql = get_sql(schema, pred["gold_sql"])
        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]["count"] += 1
        scores["all"]["count"] += 1

        try:
            p_sql = get_sql(schema, pred["predicted_sql"])
        except Exception as e:
            log_metric_error(
                "parse_pred_sql_failed | db_id=%s | error=%s | sql=%s",
                db_name,
                e,
                pred["predicted_sql"],
            )
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

        kmap = kmaps.get(db_name)
        if kmap is None:
            raise KeyError(
                f"db_id '{db_name}' is missing in tables metadata files: {loaded_table_files}"
            )
        g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_col(g_valid, rebuild_sql_val(g_sql), kmap)
        p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_col(p_valid, rebuild_sql_val(p_sql), kmap)

        p_rows, p_err, g_rows, g_err = _pair_from_cache(cache, pred)
        if p_err is not None or g_err is not None:
            exec_score = False
        else:
            exec_score = eval_exec_match_from_rows(p_rows, g_rows, p_sql, g_sql)
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
    metrics_errors_log_path: Optional[str] = None,
) -> dict:
    configure_metrics_error_log(metrics_errors_log_path)
    try:
        cache = _build_exec_cache(predictions, db_paths, timeout)

        ex, ex_flags = execution_accuracy(predictions, cache)
        ex_perm, ex_perm_flags = execution_accuracy_permuted(predictions, cache)
        em, em_flags = exact_match(predictions)
        vsr, vsr_flags = valid_sql_rate(predictions, cache)

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

        if metrics_errors_log_path:
            result["metrics_errors_log_path"] = metrics_errors_log_path

        if spider_tables_json:
            result["spider_official"] = compute_spider_component_metrics(
                predictions, db_paths, spider_tables_json, cache
            )

        return result
    finally:
        close_metrics_error_log()
