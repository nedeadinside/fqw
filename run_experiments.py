"""
Скрипт запуска всех экспериментов последовательно.

Порядок:
    1. E0-ZS  — zero-shot baseline (без обучения, inference only)
    2. E0-FS  — 3-shot baseline   (без обучения, inference only)
    3. E1     — QLoRA r=8,  custom tokens
    4. E2     — QLoRA r=16, custom tokens  (основной эксперимент)
    5. E3     — QLoRA r=32, custom tokens
    6. E4     — QLoRA r=16, без custom tokens  (ablation)
    7. E5     — QLoRA r=16, attention only, custom tokens  (ablation)

Использование:
    python run_experiments.py [--config configs/training_config.yaml]
                              [--only E2]        # запустить только один
                              [--skip E3,E5]     # пропустить некоторые
                              [--eval_only]      # только оценка (не обучать)
                              [--split test]     # сплит для оценки
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIG = "configs/training_config.yaml"
RESULTS_DIR = "results"

# Эксперименты, которые требуют обучения (fine-tuning)
FINETUNE_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5"]

# Эксперименты baseline (только инференс)
BASELINE_EXPERIMENTS = ["E0-ZS", "E0-FS"]


def run_training(experiment_id: str, config: str):
    """Запускает обучение для заданного эксперимента."""
    print(f"\n{'#'*60}")
    print(f"# ОБУЧЕНИЕ: {experiment_id}")
    print(f"{'#'*60}")
    cmd = [
        sys.executable, "-m", "src.training.train",
        "--config", config,
        "--experiment", experiment_id,
    ]
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_evaluation(
    experiment_id: str,
    config: str,
    split: str,
    model_path: str | None = None,
    baseline_mode: str | None = None,
):
    """Запускает оценку для заданного эксперимента."""
    print(f"\n{'#'*60}")
    print(f"# ОЦЕНКА: {experiment_id} | split={split}")
    print(f"{'#'*60}")

    cmd = [
        sys.executable, "-m", "src.evaluation.evaluate",
        "--config", config,
        "--experiment", experiment_id,
        "--split", split,
        "--output_dir", RESULTS_DIR,
    ]

    if model_path:
        cmd += ["--model_path", model_path]

    if baseline_mode:
        cmd += ["--baseline", baseline_mode]

    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def get_model_path(experiment_id: str) -> str:
    """Возвращает путь к лучшему checkpoint для эксперимента."""
    import yaml
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg["output_dir"]) / experiment_id / "best"
    return str(output_dir)


def collect_results(experiments: list[str], split: str) -> dict:
    """Собирает метрики всех экспериментов в одну таблицу."""
    metrics_dir = Path(RESULTS_DIR) / "metrics"
    all_results = {}
    for exp_id in experiments:
        metrics_file = metrics_dir / f"{exp_id}_{split}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_results[exp_id] = json.load(f)
    return all_results


def print_summary_table(results: dict):
    """Выводит итоговую сводную таблицу."""
    print(f"\n{'='*70}")
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'='*70}")
    header = f"{'Эксперимент':<15} {'EX':>8} {'EM':>8} {'VSR':>8} {'EX Spider':>10} {'EX BIRD':>10}"
    print(header)
    print("-" * 70)
    for exp_id, m in sorted(results.items()):
        ex = m.get("ex", 0.0)
        em = m.get("em", 0.0)
        vsr = m.get("vsr", 0.0)
        ex_spider = m.get("ex_by_source", {}).get("spider", 0.0)
        ex_bird = m.get("ex_by_source", {}).get("bird", 0.0)
        row = (
            f"{exp_id:<15} {ex:>8.4f} {em:>8.4f} {vsr:>8.4f} "
            f"{ex_spider:>10.4f} {ex_bird:>10.4f}"
        )
        print(row)
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Запуск всех экспериментов ВКР")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--only",
        default=None,
        help="Запустить только указанные эксперименты (через запятую), напр. E2,E4",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Пропустить эксперименты (через запятую), напр. E3,E5",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Только оценка, без обучения",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "test_spider_held_out"],
    )
    args = parser.parse_args()

    all_exp = BASELINE_EXPERIMENTS + FINETUNE_EXPERIMENTS

    if args.only:
        experiments = [e.strip() for e in args.only.split(",")]
    else:
        experiments = all_exp

    if args.skip:
        skip_set = {e.strip() for e in args.skip.split(",")}
        experiments = [e for e in experiments if e not in skip_set]

    print(f"Эксперименты к запуску: {experiments}")
    print(f"Сплит оценки: {args.split}")
    print(f"Только оценка: {args.eval_only}")

    for exp_id in experiments:
        try:
            if exp_id in BASELINE_EXPERIMENTS:
                # Baseline: только инференс
                mode = "zero_shot" if exp_id == "E0-ZS" else "few_shot"
                run_evaluation(
                    exp_id, args.config, args.split, baseline_mode=mode
                )
            else:
                # Fine-tuning эксперимент
                if not args.eval_only:
                    run_training(exp_id, args.config)

                model_path = get_model_path(exp_id)
                if Path(model_path).exists():
                    run_evaluation(
                        exp_id, args.config, args.split, model_path=model_path
                    )
                else:
                    print(f"[WARNING] Checkpoint не найден: {model_path}")
                    print(f"  Пропускаем оценку для {exp_id}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Эксперимент {exp_id} завершился с ошибкой: {e}")
            print("  Продолжаем со следующим экспериментом...")

    # Финальная сводка
    all_results = collect_results(experiments, args.split)
    if all_results:
        print_summary_table(all_results)

        summary_path = Path(RESULTS_DIR) / "metrics" / f"summary_{args.split}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Сводные результаты сохранены: {summary_path}")


if __name__ == "__main__":
    main()
