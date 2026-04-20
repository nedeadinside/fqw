from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_CONFIG_PATH = "configs/train_qwen_2-5-3b.yaml"
GENERATE_CONFIG_PATH = "configs/generate_qwen.yaml"
EVAL_CONFIG_PATH = "configs/eval_qwen.yaml"
CHAT_TEMPLATE_PATH = "templates/qwen_chat_template.jinja"
RUN_ID = "E2"


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    return candidate


def _require_file(path: str, label: str) -> Path:
    resolved = _resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return resolved


def _build_metrics_path(eval_cfg: dict[str, Any], predictions_path: Path) -> Path:
    if "metrics_path" in eval_cfg:
        return _resolve_project_path(str(eval_cfg["metrics_path"]))

    results_dir = _resolve_project_path(str(eval_cfg.get("results_dir", "./results")))
    stem = predictions_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return results_dir / "metrics" / f"{stem}_metrics.json"


def _preflight_train() -> None:
    _require_file(TRAIN_CONFIG_PATH, "Train config")
    _require_file(CHAT_TEMPLATE_PATH, "Chat template")


def _preflight_generate() -> None:
    _require_file(GENERATE_CONFIG_PATH, "Generate config")
    _require_file(CHAT_TEMPLATE_PATH, "Chat template")


def _preflight_test() -> tuple[dict[str, Any], Path, Path]:
    from src.evaluation._config import load_config

    config_file = _require_file(EVAL_CONFIG_PATH, "Eval config")
    cfg = load_config(config_file)

    predictions_raw = cfg.get("predictions_path")
    if not predictions_raw:
        raise ValueError("Eval config must contain 'predictions_path'")

    predictions_path = _resolve_project_path(str(predictions_raw))
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Predictions file configured for test command was not found: "
            f"{predictions_raw}"
        )

    metrics_path = _build_metrics_path(cfg, predictions_path)
    return cfg, predictions_path, metrics_path


def _run_train() -> int:
    _preflight_train()

    from src.training.train import train

    best_checkpoint = train(
        config_path=TRAIN_CONFIG_PATH,
        chat_template_path=CHAT_TEMPLATE_PATH,
        run_id=RUN_ID,
    )
    print(f"Train finished. Best checkpoint: {best_checkpoint}")
    return 0


def _run_generate() -> int:
    _preflight_generate()

    from src.evaluation.generate import generate

    predictions_path = generate(
        config_path=GENERATE_CONFIG_PATH,
        chat_template_path=CHAT_TEMPLATE_PATH,
        run_id=RUN_ID,
    )
    print(f"Generate finished. Predictions saved to: {predictions_path}")
    return 0


def _run_test() -> int:
    cfg, predictions_path, metrics_path = _preflight_test()

    from src.evaluation.evaluate import evaluate

    metrics = evaluate(config_path=EVAL_CONFIG_PATH)

    summary = {
        "ex_strict": metrics.get("ex_strict"),
        "ex_permuted": metrics.get("ex_permuted"),
        "em": metrics.get("em"),
        "vsr": metrics.get("vsr"),
        "n_examples": metrics.get("n_examples"),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }

    split_hint = cfg.get("split")
    if split_hint is not None:
        summary["split"] = split_hint

    print("Test finished. Metrics summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for FQW pipeline. "
            "Uses default YAML configs and run id."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "train",
        help="Run training with default train config.",
    )
    subparsers.add_parser(
        "generate",
        help="Generate predictions with default generate config.",
    )
    subparsers.add_parser(
        "test",
        help="Run evaluation metrics with default eval config.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "train": _run_train,
        "generate": _run_generate,
        "test": _run_test,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")

    try:
        return handler()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
