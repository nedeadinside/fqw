from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_KINDS = ("model", "training", "inference", "evaluation")


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_id: str
    root: Path
    checkpoints_dir: Path
    best_model_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    logs_dir: Path
    config_dir: Path
    manifest_path: Path

    def prediction_path(self, split: str) -> Path:
        return self.predictions_dir / f"{split}_predictions.jsonl"

    def metrics_path(self, split: str) -> Path:
        return self.metrics_dir / f"{split}_metrics.json"

    def ensure_directories(self) -> None:
        for path in (
            self.root,
            self.checkpoints_dir,
            self.predictions_dir,
            self.metrics_dir,
            self.logs_dir,
            self.config_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    token = token.strip("-")
    return token or "exp"


def parse_profile_overrides(raw_items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --profile value '{item}'. Expected format: kind=name"
            )
        kind, name = item.split("=", 1)
        kind = kind.strip()
        name = name.strip()
        if kind not in PROFILE_KINDS:
            raise ValueError(
                f"Unknown profile kind '{kind}'. Allowed: {', '.join(PROFILE_KINDS)}"
            )
        if not name:
            raise ValueError(f"Profile name cannot be empty for kind '{kind}'")
        overrides[kind] = name
    return overrides


def resolve_project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def _resolve_pipeline_path(pipeline_path: str | Path) -> Path:
    candidate = Path(pipeline_path)
    if candidate.exists():
        return candidate.resolve()

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")


def _resolve_profile_path(
    profile_name: str,
    kind: str,
    profiles_dir: Path,
) -> Path:
    explicit = Path(profile_name)
    if explicit.suffix in {".yaml", ".yml"} or "/" in profile_name:
        candidate = resolve_project_path(explicit)
    else:
        candidate = profiles_dir / kind / f"{profile_name}.yaml"

    if not candidate.exists():
        raise FileNotFoundError(
            f"Profile for kind '{kind}' not found: {profile_name} (resolved to {candidate})"
        )
    return candidate


def _build_auto_experiment_id(
    experiment_cfg: dict[str, Any],
    selected_profiles: dict[str, str],
) -> str:
    prefix = _slugify(str(experiment_cfg.get("prefix", "nl2sql")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [prefix, timestamp]
    if experiment_cfg.get("include_profiles_in_id", True):
        model_profile = selected_profiles.get("model")
        training_profile = selected_profiles.get("training")
        if model_profile:
            parts.append(_slugify(model_profile))
        if training_profile:
            parts.append(_slugify(training_profile))

    return "_".join(parts)


def _build_paths(artifacts_root: Path, experiment_id: str) -> ExperimentPaths:
    root = artifacts_root / experiment_id
    checkpoints_dir = root / "checkpoints"
    best_model_dir = checkpoints_dir / "best"
    predictions_dir = root / "predictions"
    metrics_dir = root / "metrics"
    logs_dir = root / "logs"
    config_dir = root / "config"
    manifest_path = root / "manifest.json"
    return ExperimentPaths(
        experiment_id=experiment_id,
        root=root,
        checkpoints_dir=checkpoints_dir,
        best_model_dir=best_model_dir,
        predictions_dir=predictions_dir,
        metrics_dir=metrics_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        manifest_path=manifest_path,
    )


def resolve_pipeline(
    pipeline_path: str,
    profile_overrides: dict[str, str] | None = None,
    experiment_id_override: str | None = None,
    id_mode_override: str | None = None,
) -> tuple[dict[str, Any], ExperimentPaths]:
    pipeline_file = _resolve_pipeline_path(pipeline_path)
    base_cfg = _load_yaml(pipeline_file)

    profiles_dir_raw = str(base_cfg.get("profiles_dir", "configs/profiles"))
    profiles_dir = resolve_project_path(profiles_dir_raw)

    selected_profiles: dict[str, str] = {
        key: str(value)
        for key, value in base_cfg.get("profiles", {}).items()
        if key in PROFILE_KINDS and value
    }
    selected_profiles.update(profile_overrides or {})

    merged = copy.deepcopy(base_cfg)
    selected_profile_paths: dict[str, str] = {}
    for kind in PROFILE_KINDS:
        profile_name = selected_profiles.get(kind)
        if not profile_name:
            continue

        profile_path = _resolve_profile_path(profile_name, kind, profiles_dir)
        profile_cfg = _load_yaml(profile_path)
        merged = _deep_merge(merged, profile_cfg)
        selected_profile_paths[kind] = str(profile_path)

    merged["profiles"] = selected_profiles

    experiment_cfg = merged.setdefault("experiment", {})
    id_mode = (
        (id_mode_override or experiment_cfg.get("id_mode", "auto")).strip().lower()
    )
    if id_mode not in {"auto", "manual"}:
        raise ValueError("experiment.id_mode must be either 'auto' or 'manual'")

    if experiment_id_override:
        experiment_id = experiment_id_override
    elif id_mode == "manual":
        experiment_id = str(experiment_cfg.get("id", "")).strip()
        if not experiment_id:
            raise ValueError("Manual id mode requires experiment.id or --experiment-id")
    else:
        experiment_id = _build_auto_experiment_id(experiment_cfg, selected_profiles)

    if not experiment_id:
        raise ValueError("Experiment ID cannot be empty")

    artifacts_root = resolve_project_path(
        str(experiment_cfg.get("artifacts_root", "./artifacts/experiments"))
    )
    paths = _build_paths(artifacts_root, experiment_id)

    merged["_meta"] = {
        "pipeline_path": str(pipeline_file),
        "profiles": selected_profiles,
        "profile_paths": selected_profile_paths,
        "project_root": str(PROJECT_ROOT),
        "experiment_id": experiment_id,
        "id_mode": id_mode,
    }

    return merged, paths


def save_effective_config(cfg: dict[str, Any], paths: ExperimentPaths) -> Path:
    paths.ensure_directories()
    snapshot_path = paths.config_dir / "effective_pipeline.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return snapshot_path


def update_manifest(
    paths: ExperimentPaths,
    stage: str,
    payload: dict[str, Any],
) -> Path:
    paths.ensure_directories()

    manifest: dict[str, Any] = {}
    if paths.manifest_path.exists():
        with open(paths.manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    manifest.setdefault("experiment_id", paths.experiment_id)
    manifest.setdefault(
        "paths",
        {
            "root": str(paths.root),
            "checkpoints": str(paths.checkpoints_dir),
            "best_model": str(paths.best_model_dir),
            "predictions": str(paths.predictions_dir),
            "metrics": str(paths.metrics_dir),
            "logs": str(paths.logs_dir),
            "config": str(paths.config_dir),
        },
    )
    manifest.setdefault("stages", {})
    manifest["stages"][stage] = payload
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(paths.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return paths.manifest_path
