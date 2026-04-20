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
