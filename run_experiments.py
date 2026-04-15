from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG = "configs/training_config.yaml"
RESULTS_DIR = "results"
EXPERIMENT_ID = "E2"


def run_training(config_path: str) -> bool:
    command = [
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        config_path,
    ]
    result = subprocess.run(command, check=True)
    return result.returncode == 0


def run_evaluation(config_path: str, split: str, model_path: str) -> bool:
    command = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--config",
        config_path,
        "--split",
        split,
        "--output_dir",
        RESULTS_DIR,
        "--model_path",
        model_path,
    ]
    result = subprocess.run(command, check=True)
    return result.returncode == 0


def get_model_path(config_path: str) -> str:
    with open(config_path, encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    return str(Path(config["output_dir"]) / EXPERIMENT_ID / "best")


def load_metrics(split: str) -> dict:
    metrics_path = (
        Path(RESULTS_DIR) / "metrics" / f"{EXPERIMENT_ID}_{split}_metrics.json"
    )
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def print_summary(split: str, metrics: dict):
    if not metrics:
        print("[summary] no metrics file")
        return
    print(f"[summary] experiment={EXPERIMENT_ID} split={split}")
    print(f"EX={metrics.get('ex', 0):.4f}")
    print(f"EM={metrics.get('em', 0):.4f}")
    print(f"VSR={metrics.get('vsr', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run main experiment")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "test_spider_held_out"],
    )
    args = parser.parse_args()

    print(f"[run] experiment={EXPERIMENT_ID} split={args.split}")

    try:
        if not args.eval_only:
            run_training(args.config)

        model_path = get_model_path(args.config)
        if not Path(model_path).exists():
            print(f"[run] checkpoint not found: {model_path}")
            return

        run_evaluation(args.config, args.split, model_path)
        metrics = load_metrics(args.split)
        print_summary(args.split, metrics)
    except subprocess.CalledProcessError as error:
        print(f"[run] failed: {error}")


if __name__ == "__main__":
    main()
