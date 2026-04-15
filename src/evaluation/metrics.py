"""
Метрики оценки качества NL-to-SQL систем.

Реализованные метрики:
    execution_accuracy (EX)  — основная: совпадение result sets
    exact_match (EM)         — нормализованное строковое сравнение SQL
    valid_sql_rate (VSR)     — доля синтаксически корректных запросов
    ves_score (VES)          — BIRD Valid Efficiency Score
    bleu_score               — BLEU (для справки, ограниченная применимость)
    rouge_l_score            — ROUGE-L (для справки)
    compute_all_metrics      — вычисляет все метрики разом

Формат записи предсказания (dict):
    {
        "example_id":    str,
        "db_id":         str,
        "question":      str,
        "predicted_sql": str,
        "gold_sql":      str,
        "source":        "spider" | "bird",
        "complexity":    "easy" | "medium" | "hard" | "extra"  (опционально)
    }
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.evaluation.sql_executor import execute_sql, results_match


# ---------------------------------------------------------------------------
# Нормализация SQL для Exact Match
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """Нормализует SQL для строкового сравнения.

    Шаги:
        1. Нижний регистр
        2. Удаление trailing ';'
        3. Схлопывание пробелов
    """
    sql = sql.lower().strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return " ".join(sql.split())


# ---------------------------------------------------------------------------
# Execution Accuracy
# ---------------------------------------------------------------------------

def execution_accuracy(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    """Execution Accuracy (EX).

    Для каждого примера выполняет pred_sql и gold_sql на БД,
    сравнивает result sets как frozenset.

    Returns:
        (score, per_example_correct)
        score — float [0, 1]
        per_example_correct — список bool по каждому примеру
    """
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


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(predictions: List[dict]) -> Tuple[float, List[bool]]:
    """Exact Match (EM): нормализованное строковое сравнение.

    Returns:
        (score, per_example_correct)
    """
    correct_flags = []
    for pred in predictions:
        norm_pred = normalize_sql(pred["predicted_sql"])
        norm_gold = normalize_sql(pred["gold_sql"])
        correct_flags.append(norm_pred == norm_gold)

    score = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    return score, correct_flags


# ---------------------------------------------------------------------------
# Valid SQL Rate
# ---------------------------------------------------------------------------

def valid_sql_rate(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
) -> Tuple[float, List[bool]]:
    """Valid SQL Rate (VSR): доля синтаксически и семантически валидных запросов.

    Запрос считается валидным, если выполняется без исключения SQLite.

    Returns:
        (score, per_example_valid)
    """
    valid_flags = []
    for pred in predictions:
        db_id = pred["db_id"]
        db_path = db_paths.get(db_id, "")
        _, err = execute_sql(db_path, pred["predicted_sql"], timeout)
        valid_flags.append(err is None)

    score = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    return score, valid_flags


# ---------------------------------------------------------------------------
# Valid Efficiency Score (BIRD)
# ---------------------------------------------------------------------------

def ves_score(
    predictions: List[dict],
    db_paths: Dict[str, str],
    timeout: float = 30.0,
    time_ratio_clip: float = 100.0,
) -> float:
    """Valid Efficiency Score (VES) — метрика BIRD.

    Формула (упрощённая):
        VES = mean( sqrt(min(R_gold / R_pred, clip)) )
        где R = время выполнения запроса (мс), R_pred > 0

    Если pred_sql невалиден — contributes 0 к VES.
    Работает только на примерах с source="bird".
    """
    scores = []

    for pred in predictions:
        db_id = pred["db_id"]
        db_path = db_paths.get(db_id, "")
        if not db_path:
            continue

        # Время выполнения gold SQL
        t0 = time.perf_counter()
        gold_result, gold_err = execute_sql(db_path, pred["gold_sql"], timeout)
        t_gold = (time.perf_counter() - t0) * 1000  # мс

        if gold_err is not None:
            continue

        # Время выполнения pred SQL
        t0 = time.perf_counter()
        pred_result, pred_err = execute_sql(db_path, pred["predicted_sql"], timeout)
        t_pred = (time.perf_counter() - t0) * 1000  # мс

        if pred_err is not None:
            scores.append(0.0)
            continue

        # Корректность (pred должен вернуть тот же результат)
        if not results_match(pred_result, gold_result):
            scores.append(0.0)
            continue

        # Отношение времён
        t_pred = max(t_pred, 1.0)  # защита от деления на 0
        ratio = min(t_gold / t_pred, time_ratio_clip)
        scores.append(ratio ** 0.5)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# BLEU и ROUGE-L (для полноты теоретического обзора)
# ---------------------------------------------------------------------------

def bleu_score(predictions: List[dict]) -> float:
    """BLEU между предсказанными и золотыми SQL-запросами.

    Примечание: BLEU плохо коррелирует с семантической корректностью SQL.
    Включён для теоретического сравнения.
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError("Установите: pip install sacrebleu")

    bleu = BLEU(tokenize="char")
    hypotheses = [p["predicted_sql"] for p in predictions]
    references = [[p["gold_sql"] for p in predictions]]
    result = bleu.corpus_score(hypotheses, references)
    return result.score / 100.0  # Нормализуем в [0, 1]


def rouge_l_score(predictions: List[dict]) -> float:
    """ROUGE-L между предсказанными и золотыми SQL-запросами.

    Примечание: аналогично BLEU, имеет ограниченную применимость для SQL.
    """
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


# ---------------------------------------------------------------------------
# Стратификация по источнику и сложности
# ---------------------------------------------------------------------------

def stratify_by_source(
    predictions: List[dict],
    correct_flags: List[bool],
) -> Dict[str, float]:
    """EX по источнику: Spider vs BIRD."""
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
    """EX по уровню сложности SQL (easy/medium/hard/extra).

    Поле 'complexity' должно быть в записи предсказания.
    """
    by_complexity: Dict[str, List[bool]] = defaultdict(list)
    for pred, flag in zip(predictions, correct_flags):
        complexity = pred.get("complexity", "unknown")
        by_complexity[complexity].append(flag)

    return {
        lvl: sum(flags) / len(flags) if flags else 0.0
        for lvl, flags in by_complexity.items()
    }


# ---------------------------------------------------------------------------
# Вычисление всех метрик разом
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: List[dict],
    db_paths: Dict[str, str],
    compute_bleu: bool = True,
    compute_rouge: bool = True,
    compute_ves: bool = True,
    timeout: float = 30.0,
) -> dict:
    """Вычисляет EX, EM, VSR, VES, BLEU, ROUGE-L и стратификации.

    Args:
        predictions: список предсказаний с полями из формата выше
        db_paths:    словарь {db_id: path_to_sqlite}
        compute_bleu: считать ли BLEU (требует sacrebleu)
        compute_rouge: считать ли ROUGE-L (требует rouge-score)
        compute_ves:   считать ли VES (только для BIRD-примеров)
        timeout:     таймаут выполнения одного SQL-запроса

    Returns:
        dict со всеми метриками
    """
    print(f"[metrics] Вычисление метрик для {len(predictions)} примеров...")

    # EX
    ex, ex_flags = execution_accuracy(predictions, db_paths, timeout)
    print(f"  EX  = {ex:.4f}")

    # EM
    em, _ = exact_match(predictions)
    print(f"  EM  = {em:.4f}")

    # VSR
    vsr, _ = valid_sql_rate(predictions, db_paths, timeout)
    print(f"  VSR = {vsr:.4f}")

    # Стратификация EX
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

    # VES (только BIRD)
    if compute_ves:
        bird_preds = [p for p in predictions if p.get("source") == "bird"]
        if bird_preds:
            ves = ves_score(bird_preds, db_paths, timeout)
            results["ves"] = ves
            print(f"  VES = {ves:.4f}  (BIRD, {len(bird_preds)} примеров)")

    # BLEU
    if compute_bleu:
        try:
            bleu = bleu_score(predictions)
            results["bleu"] = bleu
            print(f"  BLEU = {bleu:.4f}")
        except ImportError as e:
            print(f"  BLEU: {e}")

    # ROUGE-L
    if compute_rouge:
        try:
            rouge = rouge_l_score(predictions)
            results["rouge_l"] = rouge
            print(f"  ROUGE-L = {rouge:.4f}")
        except ImportError as e:
            print(f"  ROUGE-L: {e}")

    return results
