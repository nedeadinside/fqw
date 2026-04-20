from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.dataset import build_db_path_index, load_jsonl
from src.evaluation._config import (
    load_config,
    resolve_config_path,
    resolve_optional_path,
)
from src.evaluation.metrics import compute_all_metrics

REQUIRED_CONFIG_KEYS = (
    "predictions_path",
    "spider_db_dir",
    "spider_test_db_dir",
)


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _metrics_output_path(cfg: dict[str, Any], predictions_path: Path) -> Path:
    if "metrics_path" in cfg:
        return resolve_optional_path(str(cfg["metrics_path"]))
    results_dir = resolve_optional_path(str(cfg.get("results_dir", "./results")))
    stem = predictions_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return results_dir / "metrics" / f"{stem}_metrics.json"


def evaluate(
    config_path: str | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> dict:
    if cfg_override is not None:
        cfg = dict(cfg_override)
    else:
        if config_path is None:
            raise ValueError("Either config_path or cfg_override must be provided")
        cfg = load_config(resolve_config_path(config_path))

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Eval config missing required keys: {missing}")

    predictions_path = resolve_optional_path(cfg["predictions_path"])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {cfg['predictions_path']}")

    predictions = load_jsonl(predictions_path)

    db_paths = build_db_path_index(
        predictions,
        spider_db_dir=cfg["spider_db_dir"],
        spider_test_db_dir=cfg["spider_test_db_dir"],
    )

    metrics = compute_all_metrics(
        predictions=predictions,
        db_paths=db_paths,
        timeout=cfg.get("execution_timeout", 30.0),
        spider_db_dir=cfg.get("spider_db_dir"),
        spider_tables_json=cfg.get("spider_tables_json"),
    )
    metrics["predictions_path"] = str(predictions_path)
    metrics["n_predictions"] = len(predictions)

    out_path = _metrics_output_path(cfg, predictions_path)
    _save_metrics(metrics, out_path)
    return metrics


if __name__ == "__main__":
    EVAL_CONFIG_PATH = "configs/eval_qwen.yaml"
    metrics = evaluate(config_path=EVAL_CONFIG_PATH)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
