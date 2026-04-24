from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Generator

from huggingface_hub import snapshot_download


def _iter_json_array(
    path: Path, chunk_size: int = 1 << 20
) -> Generator[dict[str, Any], None, None]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        idx = 0
        started = False
        eof = False

        while True:
            if idx >= len(buffer) and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1

            if not started:
                if idx >= len(buffer):
                    if eof:
                        break
                    continue
                if buffer[idx] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started = True
                idx += 1
                continue

            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1

            if idx >= len(buffer):
                if eof:
                    break
                continue

            if buffer[idx] == "]":
                break

            if buffer[idx] == ",":
                idx += 1
                continue

            try:
                value, next_idx = decoder.raw_decode(buffer, idx)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            if isinstance(value, dict):
                yield value

            idx = next_idx
            if idx > (1 << 20):
                buffer = buffer[idx:]
                idx = 0


def _build_field_profile(data_file: Path, max_rows: int) -> dict[str, Any]:
    key_freq: Counter[str] = Counter()
    first_example: dict[str, Any] | None = None
    sampled_rows = 0

    for row in _iter_json_array(data_file):
        if first_example is None:
            first_example = row
        for k in row.keys():
            key_freq[k] += 1
        sampled_rows += 1
        if sampled_rows >= max_rows:
            break

    return {
        "sampled_rows": sampled_rows,
        "detected_keys_sorted": sorted(key_freq.keys()),
        "key_frequency_sample": dict(key_freq),
        "first_row_keys": sorted(first_example.keys()) if first_example else [],
        "first_row_preview": first_example,
    }


def download_synsql(dataset_id: str, output_dir: str, profile_rows: int) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset snapshot: {dataset_id}", file=sys.stderr)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["data.json", "tables.json", "databases/**", "README*"],
    )

    data_file = out_dir / "data.json"
    tables_file = out_dir / "tables.json"
    if not data_file.exists() or not tables_file.exists():
        raise FileNotFoundError(
            "Expected data.json and tables.json after download, but one of them is missing"
        )

    profile = {
        "dataset_id": dataset_id,
        "output_dir": str(out_dir),
        "files": {
            "data.json": data_file.stat().st_size,
            "tables.json": tables_file.stat().st_size,
        },
        "field_profile": _build_field_profile(data_file, profile_rows),
    }

    profile_path = out_dir / "field_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"Saved field profile to: {profile_path}", file=sys.stderr)
    return profile_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SynSQL dataset snapshot from HuggingFace and profile fields"
    )
    parser.add_argument(
        "--dataset-id",
        default="seeklhy/SynSQL-2.5M",
        help="HuggingFace dataset id",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/matvey/projects/fqw/raw_data/SynSQL",
        help="Where to save downloaded dataset files",
    )
    parser.add_argument(
        "--profile-rows",
        type=int,
        default=1000,
        help="How many rows from data.json to sample for field profile",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    download_synsql(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        profile_rows=args.profile_rows,
    )


if __name__ == "__main__":
    main()
