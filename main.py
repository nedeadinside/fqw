from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.config import (
    parse_profile_overrides,
    resolve_pipeline,
    resolve_project_path,
    save_effective_config,
    update_manifest,
)
from src.config.pipeline import ExperimentPaths

DEFAULT_PIPELINE_PATH = "configs/pipeline.yaml"


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_chat_template(effective_cfg: dict[str, Any]) -> str | None:
    template_value = effective_cfg.get("paths", {}).get("chat_template")
    if not template_value:
        return None
    template_path = resolve_project_path(str(template_value))
    _require_file(template_path, "Chat template")
    return str(template_path)


def _resolve_processed_data_dir(effective_cfg: dict[str, Any]) -> Path:
    processed_data_dir = effective_cfg.get("paths", {}).get(
        "processed_data_dir", "./processed_data"
    )
    path = resolve_project_path(str(processed_data_dir))
    if not path.exists():
        raise FileNotFoundError(f"Processed data dir not found: {path}")
    return path


def _build_train_cfg(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
) -> dict[str, Any]:
    train_cfg = dict(effective_cfg.get("train", {}))
    model_cfg = effective_cfg.get("model", {})

    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("Model profile must define model.name")

    train_cfg["model_name"] = model_name
    train_cfg["max_seq_length"] = model_cfg.get(
        "max_seq_length", train_cfg.get("max_seq_length", 2048)
    )
    train_cfg["load_in_4bit"] = model_cfg.get(
        "load_in_4bit", train_cfg.get("load_in_4bit", True)
    )
    train_cfg["processed_data_dir"] = str(_resolve_processed_data_dir(effective_cfg))
    train_cfg["checkpoint_dir"] = str(experiment_paths.checkpoints_dir)
    train_cfg["logging_dir"] = str(experiment_paths.logs_dir)

    return train_cfg


def _build_generate_cfg(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
    split_override: str | None,
) -> tuple[dict[str, Any], str]:
    generate_cfg = dict(effective_cfg.get("generate", {}))
    model_cfg = effective_cfg.get("model", {})
    train_cfg = dict(effective_cfg.get("train", {}))

    split = split_override or str(generate_cfg.get("split", "test"))
    generate_cfg["split"] = split

    generate_cfg["processed_data_dir"] = str(_resolve_processed_data_dir(effective_cfg))
    generate_cfg["load_in_4bit"] = model_cfg.get(
        "load_in_4bit", generate_cfg.get("load_in_4bit", False)
    )

    custom_tokens = train_cfg.get("custom_special_tokens")
    if custom_tokens:
        generate_cfg["custom_special_tokens"] = custom_tokens

    if "model_path" not in generate_cfg:
        generate_cfg["best_model_dir"] = str(experiment_paths.best_model_dir)

    generate_cfg["predictions_path"] = str(experiment_paths.prediction_path(split))

    if effective_cfg.get("profiles", {}).get("training") == "evidence":
        generate_cfg["strip_evidence"] = True

    return generate_cfg, split


def _build_evaluate_cfg(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
    split: str,
) -> dict[str, Any]:
    evaluate_cfg = dict(effective_cfg.get("evaluate", {}))
    paths_cfg = effective_cfg.get("paths", {})

    spider_db_dir = resolve_project_path(
        str(paths_cfg.get("spider_db_dir", "./raw_data/Spider/database"))
    )
    spider_test_db_dir = resolve_project_path(
        str(paths_cfg.get("spider_test_db_dir", "./raw_data/Spider/test_database"))
    )

    if not spider_db_dir.exists():
        raise FileNotFoundError(f"Spider db dir not found: {spider_db_dir}")
    if not spider_test_db_dir.exists():
        raise FileNotFoundError(f"Spider test db dir not found: {spider_test_db_dir}")

    evaluate_cfg["predictions_path"] = str(experiment_paths.prediction_path(split))
    evaluate_cfg["metrics_path"] = str(experiment_paths.metrics_path(split))
    evaluate_cfg["spider_db_dir"] = str(spider_db_dir)
    evaluate_cfg["spider_test_db_dir"] = str(spider_test_db_dir)

    spider_tables_json = paths_cfg.get("spider_tables_json")
    if spider_tables_json:
        evaluate_cfg["spider_tables_json"] = str(
            resolve_project_path(str(spider_tables_json))
        )

    return evaluate_cfg


def _run_train_stage(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
    chat_template_path: str | None,
) -> Path:
    from src.training.train import train

    train_cfg = _build_train_cfg(effective_cfg, experiment_paths)
    best_checkpoint = Path(
        train(
            chat_template_path=chat_template_path,
            cfg_override=train_cfg,
        )
    )

    update_manifest(
        experiment_paths,
        "train",
        {
            "status": "completed",
            "checkpoint_dir": str(experiment_paths.checkpoints_dir),
            "best_model_dir": str(best_checkpoint),
        },
    )
    return best_checkpoint


def _run_generate_stage(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
    chat_template_path: str | None,
    split_override: str | None,
) -> tuple[Path, str]:
    from src.evaluation.generate import generate

    generate_cfg, split = _build_generate_cfg(
        effective_cfg,
        experiment_paths,
        split_override=split_override,
    )

    if (
        "model_path" not in generate_cfg
        and not experiment_paths.best_model_dir.exists()
    ):
        raise FileNotFoundError(
            "Best model checkpoint not found for generation. "
            f"Expected: {experiment_paths.best_model_dir}"
        )

    predictions_path = generate(
        chat_template_path=chat_template_path,
        cfg_override=generate_cfg,
    )

    update_manifest(
        experiment_paths,
        "generate",
        {
            "status": "completed",
            "split": split,
            "predictions_path": str(predictions_path),
        },
    )
    return predictions_path, split


def _run_test_stage(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
    split: str,
) -> dict[str, Any]:
    from src.evaluation.evaluate import evaluate

    evaluate_cfg = _build_evaluate_cfg(effective_cfg, experiment_paths, split)
    predictions_path = Path(evaluate_cfg["predictions_path"])
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found for evaluation. Expected: {predictions_path}"
        )

    metrics = evaluate(cfg_override=evaluate_cfg)

    update_manifest(
        experiment_paths,
        "evaluate",
        {
            "status": "completed",
            "split": split,
            "metrics_path": str(experiment_paths.metrics_path(split)),
            "summary": {
                "ex_strict": metrics.get("ex_strict"),
                "ex_permuted": metrics.get("ex_permuted"),
                "em": metrics.get("em"),
                "vsr": metrics.get("vsr"),
                "n_examples": metrics.get("n_examples"),
            },
        },
    )

    return metrics


def _print_dry_run(
    effective_cfg: dict[str, Any],
    experiment_paths: ExperimentPaths,
) -> None:
    payload = {
        "experiment_id": experiment_paths.experiment_id,
        "paths": {
            "root": str(experiment_paths.root),
            "checkpoints": str(experiment_paths.checkpoints_dir),
            "best_model": str(experiment_paths.best_model_dir),
            "predictions": str(experiment_paths.predictions_dir),
            "metrics": str(experiment_paths.metrics_dir),
            "manifest": str(experiment_paths.manifest_path),
        },
        "profiles": effective_cfg.get("profiles", {}),
        "meta": effective_cfg.get("_meta", {}),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified pipeline launcher with pipeline+profiles configuration model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--pipeline",
            default=DEFAULT_PIPELINE_PATH,
            help="Path to pipeline YAML config.",
        )
        target.add_argument(
            "--profile",
            action="append",
            default=[],
            help=(
                "Profile override in form kind=name. "
                "Can be repeated, e.g. --profile model=qwen25_7b_coder"
            ),
        )
        target.add_argument(
            "--experiment-id",
            default=None,
            help="Force experiment ID. Recommended for standalone generate/test.",
        )
        target.add_argument(
            "--id-mode",
            choices=["auto", "manual"],
            default=None,
            help="Override experiment id mode from pipeline config.",
        )
        target.add_argument(
            "--split",
            default=None,
            help="Override generation/evaluation split.",
        )
        target.add_argument(
            "--dry-run-config",
            action="store_true",
            help="Resolve pipeline and print effective runtime context without running stages.",
        )

    train_parser = subparsers.add_parser("train", help="Run training stage.")
    generate_parser = subparsers.add_parser("generate", help="Run generation stage.")
    test_parser = subparsers.add_parser("test", help="Run evaluation stage.")
    all_parser = subparsers.add_parser("all", help="Run train -> generate -> test.")

    add_common_options(train_parser)
    add_common_options(generate_parser)
    add_common_options(test_parser)
    add_common_options(all_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        profile_overrides = parse_profile_overrides(args.profile)
        effective_cfg, experiment_paths = resolve_pipeline(
            pipeline_path=args.pipeline,
            profile_overrides=profile_overrides,
            experiment_id_override=args.experiment_id,
            id_mode_override=args.id_mode,
        )

        experiment_paths.ensure_directories()
        save_effective_config(effective_cfg, experiment_paths)
        chat_template_path = _resolve_chat_template(effective_cfg)

        if args.dry_run_config:
            _print_dry_run(effective_cfg, experiment_paths)
            return 0

        if args.command == "train":
            best_checkpoint = _run_train_stage(
                effective_cfg,
                experiment_paths,
                chat_template_path,
            )
            print(f"Train finished. Best checkpoint: {best_checkpoint}")
            return 0

        if args.command == "generate":
            predictions_path, split = _run_generate_stage(
                effective_cfg,
                experiment_paths,
                chat_template_path,
                split_override=args.split,
            )
            print(
                f"Generate finished for split '{split}'. Predictions: {predictions_path}"
            )
            return 0

        if args.command == "test":
            split = args.split or str(
                effective_cfg.get("generate", {}).get("split", "test")
            )
            metrics = _run_test_stage(
                effective_cfg,
                experiment_paths,
                split=split,
            )
            print("Test finished. Metrics summary:")
            print(
                json.dumps(
                    {
                        "split": split,
                        "ex_strict": metrics.get("ex_strict"),
                        "ex_permuted": metrics.get("ex_permuted"),
                        "em": metrics.get("em"),
                        "vsr": metrics.get("vsr"),
                        "n_examples": metrics.get("n_examples"),
                        "metrics_path": str(experiment_paths.metrics_path(split)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "all":
            _run_train_stage(effective_cfg, experiment_paths, chat_template_path)
            predictions_path, split = _run_generate_stage(
                effective_cfg,
                experiment_paths,
                chat_template_path,
                split_override=args.split,
            )
            metrics = _run_test_stage(effective_cfg, experiment_paths, split=split)

            print("Pipeline finished.")
            print(
                json.dumps(
                    {
                        "experiment_id": experiment_paths.experiment_id,
                        "split": split,
                        "predictions_path": str(predictions_path),
                        "metrics_path": str(experiment_paths.metrics_path(split)),
                        "ex_strict": metrics.get("ex_strict"),
                        "ex_permuted": metrics.get("ex_permuted"),
                        "em": metrics.get("em"),
                        "vsr": metrics.get("vsr"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
