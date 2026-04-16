from __future__ import annotations

import time
from collections import defaultdict
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


def ves_score(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
    time_ratio_clip: float = 100.0,
) -> float:
    scores = []

    for pred in predictions:
        db_id = pred["db_id"]
        db_path = db_paths.get(db_id, "")
        if not db_path:
            continue

        t0 = time.perf_counter()
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)
        t_gold = (time.perf_counter() - t0) * 1000

        if gold_err is not None:
            continue

        t0 = time.perf_counter()
        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        t_pred = (time.perf_counter() - t0) * 1000

        if pred_err is not None:
            scores.append(0.0)
            continue

        if not results_match(pred_result, gold_result):
            scores.append(0.0)
            continue

        t_pred = max(t_pred, 1.0)
        ratio = min(t_gold / t_pred, time_ratio_clip)
        scores.append(ratio**0.5)

    return sum(scores) / len(scores) if scores else 0.0


def bleu_score(predictions: List[dict]) -> float:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError("Установите: pip install sacrebleu")

    bleu = BLEU(tokenize="char")
    hypotheses = [p["predicted_sql"] for p in predictions]
    references = [[p["gold_sql"] for p in predictions]]
    result = bleu.corpus_score(hypotheses, references)
    return result.score / 100.0


def rouge_l_score(predictions: List[dict]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Установите: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred in predictions:
        result = scorer.score(pred["gold_sql"], pred["predicted_sql"])
        scores.append(result["rougeL"].fmeasure)

    return sum(scores) / len(scores) if scores else 0.0


def stratify_by_source(
    predictions: List[dict],
    correct_flags: List[bool],
) -> Dict[str, float]:
    by_source: Dict[str, List[bool]] = defaultdict(list)
    for pred, flag in zip(predictions, correct_flags):
        source = pred.get("source", "unknown")
        by_source[source].append(flag)

    return {
        src: sum(flags) / len(flags) if flags else 0.0
        for src, flags in by_source.items()
    }


def stratify_by_complexity(
    predictions: List[dict],
    correct_flags: List[bool],
) -> Dict[str, float]:
    by_complexity: Dict[str, List[bool]] = defaultdict(list)
    for pred, flag in zip(predictions, correct_flags):
        complexity = pred.get("complexity", "unknown")
        by_complexity[complexity].append(flag)

    return {
        lvl: sum(flags) / len(flags) if flags else 0.0
        for lvl, flags in by_complexity.items()
    }


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    compute_bleu: bool = True,
    compute_rouge: bool = True,
    compute_ves: bool = True,
    timeout: float = 30.0,
) -> dict:
    print(f"[metrics] Вычисление метрик для {len(predictions)} примеров...")

    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    print(f"  EX  = {ex:.4f}")

    em, _ = exact_match(predictions)
    print(f"  EM  = {em:.4f}")

    vsr, _ = valid_sql_rate(predictions, db_paths, timeout)
    print(f"  VSR = {vsr:.4f}")

    ex_by_source = stratify_by_source(predictions, ex_flags)
    ex_by_complexity = stratify_by_complexity(predictions, ex_flags)

    results = {
        "ex": ex,
        "em": em,
        "vsr": vsr,
        "ex_by_source": ex_by_source,
        "ex_by_complexity": ex_by_complexity,
        "n_examples": len(predictions),
        "n_correct_ex": sum(ex_flags),
    }

    if compute_ves:
        bird_preds = [p for p in predictions if p.get("source") == "bird"]
        if bird_preds:
            ves = ves_score(bird_preds, db_paths, timeout)
            results["ves"] = ves
            print(f"  VES = {ves:.4f}  (BIRD, {len(bird_preds)} примеров)")

    if compute_bleu:
        try:
            bleu = bleu_score(predictions)
            results["bleu"] = bleu
            print(f"  BLEU = {bleu:.4f}")
        except ImportError as e:
            print(f"  BLEU: {e}")

    if compute_rouge:
        try:
            rouge = rouge_l_score(predictions)
            results["rouge_l"] = rouge
            print(f"  ROUGE-L = {rouge:.4f}")
        except ImportError as e:
            print(f"  ROUGE-L: {e}")

    return results
