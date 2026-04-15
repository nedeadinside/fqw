from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Tuple

from src.evaluation.sql_executor import execute_sql, results_match


def normalize_sql(sql: str) -> str:
    sql_text = sql.lower().strip()
    if sql_text.endswith(";"):
        sql_text = sql_text[:-1].rstrip()
    return " ".join(sql_text.split())


def execution_accuracy(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    correct_flags: List[bool] = []
    for prediction in predictions:
        db_id = prediction["db_id"]
        db_path = db_paths.get(db_id, "")

        pred_result, pred_err = execute_sql(
            db_path, prediction["predicted_sql"], timeout
        )
        gold_result, gold_err = execute_sql(db_path, prediction["gold_sql"], timeout)

        is_correct = (
            pred_err is None
            and gold_err is None
            and results_match(pred_result, gold_result)
        )
        correct_flags.append(is_correct)

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    flags: List[bool] = []
    for prediction in predictions:
        norm_pred = normalize_sql(prediction["predicted_sql"])
        norm_gold = normalize_sql(prediction["gold_sql"])
        flags.append(norm_pred == norm_gold)
    score = sum(flags) / len(flags) if flags else 0.0
    return score, flags


def valid_sql_rate(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    flags: List[bool] = []
    for prediction in predictions:
        db_id = prediction["db_id"]
        db_path = db_paths.get(db_id, "")
        _, error = execute_sql(db_path, prediction["predicted_sql"], timeout)
        flags.append(error is None)
    score = sum(flags) / len(flags) if flags else 0.0
    return score, flags


def ves_score(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
    time_ratio_clip: float = 100.0,
) -> float:
    scores: List[float] = []
    for prediction in predictions:
        db_id = prediction["db_id"]
        db_path = db_paths.get(db_id, "")
        if not db_path:
            continue

        start = time.perf_counter()
        gold_result, gold_err = execute_sql(db_path, prediction["gold_sql"], timeout)
        gold_time = (time.perf_counter() - start) * 1000
        if gold_err is not None:
            continue

        start = time.perf_counter()
        pred_result, pred_err = execute_sql(
            db_path, prediction["predicted_sql"], timeout
        )
        pred_time = (time.perf_counter() - start) * 1000

        if pred_err is not None:
            scores.append(0.0)
            continue

        if not results_match(pred_result, gold_result):
            scores.append(0.0)
            continue

        pred_time = max(pred_time, 1.0)
        ratio = min(gold_time / pred_time, time_ratio_clip)
        scores.append(ratio**0.5)

    return sum(scores) / len(scores) if scores else 0.0


def bleu_score(predictions: List[dict]) -> float:
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="char")
    hypotheses = [prediction["predicted_sql"] for prediction in predictions]
    references = [[prediction["gold_sql"] for prediction in predictions]]
    result = bleu.corpus_score(hypotheses, references)
    return result.score / 100.0


def rouge_l_score(predictions: List[dict]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    values: List[float] = []
    for prediction in predictions:
        score = scorer.score(prediction["gold_sql"], prediction["predicted_sql"])
        values.append(score["rougeL"].fmeasure)
    return sum(values) / len(values) if values else 0.0


def stratify_by_source(
    predictions: List[dict], correct_flags: List[bool]
) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for prediction, flag in zip(predictions, correct_flags):
        grouped[prediction.get("source", "unknown")].append(flag)
    return {
        source: sum(flags) / len(flags) if flags else 0.0
        for source, flags in grouped.items()
    }


def stratify_by_complexity(
    predictions: List[dict],
    correct_flags: List[bool],
) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for prediction, flag in zip(predictions, correct_flags):
        grouped[prediction.get("complexity", "unknown")].append(flag)
    return {
        level: sum(flags) / len(flags) if flags else 0.0
        for level, flags in grouped.items()
    }


def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    compute_bleu: bool = True,
    compute_rouge: bool = True,
    compute_ves: bool = True,
    timeout: float = 30.0,
) -> dict:
    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    em, _ = exact_match(predictions)
    vsr, _ = valid_sql_rate(predictions, db_paths, timeout)

    results = {
        "ex": ex,
        "em": em,
        "vsr": vsr,
        "ex_by_source": stratify_by_source(predictions, ex_flags),
        "ex_by_complexity": stratify_by_complexity(predictions, ex_flags),
        "n_examples": len(predictions),
        "n_correct_ex": sum(ex_flags),
    }

    if compute_ves:
        bird_predictions = [p for p in predictions if p.get("source") == "bird"]
        if bird_predictions:
            results["ves"] = ves_score(bird_predictions, db_paths, timeout)

    if compute_bleu:
        try:
            results["bleu"] = bleu_score(predictions)
        except ImportError:
            pass

    if compute_rouge:
        try:
            results["rouge_l"] = rouge_l_score(predictions)
        except ImportError:
            pass

    return results
