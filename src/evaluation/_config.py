from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_optional_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_candidate = Path(__file__).resolve().parents[2] / path
    if project_candidate.exists():
        return project_candidate

    return candidate


def resolve_config_path(config_path: str) -> str:
    resolved = resolve_optional_path(config_path)
    if resolved.exists():
        return str(resolved)
    raise FileNotFoundError(f"Config not found: {config_path}")


def merge_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg_path = cfg.get("train_config_path")
    if not train_cfg_path:
        return cfg

    resolved = resolve_optional_path(str(train_cfg_path))
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced train config not found: {train_cfg_path}")

    train_cfg = load_config(resolved)
    for key in ("output_dir", "processed_data_dir", "load_in_4bit"):
        if key not in cfg and key in train_cfg:
            cfg[key] = train_cfg[key]
    return cfg
