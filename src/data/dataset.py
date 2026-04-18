from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a SQL expert. Given the database schema and a question "
    "in natural language, generate the corresponding SQL query for SQLite."
)

CUSTOM_SPECIAL_TOKENS = ["<schema>", "</schema>", "<question>", "</question>"]


def format_example(example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<schema>\n{example['schema']}\n</schema>\n\n"
                f"<question>\n{example['question']}\n</question>"
            ),
        },
        {"role": "assistant", "content": example["sql"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def load_jsonl(path: str | Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_dev_split(
    records: List[dict],
    val_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[List[dict], List[dict]]:
    rng = random.Random(seed)

    by_db: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_db[rec["db_id"]].append(rec)

    db_ids = sorted(by_db.keys())
    rng.shuffle(db_ids)

    split_point = int(len(db_ids) * val_ratio)
    val_dbs = set(db_ids[:split_point])

    val, test = [], []
    for db_id, recs in by_db.items():
        if db_id in val_dbs:
            val.extend(recs)
        else:
            test.extend(recs)

    return val, test


def load_splits(
    processed_data_dir: str | Path,
    tokenizer,
    seed: int = 42,
) -> Dict[str, Dataset]:
    data_dir = Path(processed_data_dir)

    spider_train = load_jsonl(data_dir / "spider_train.jsonl")
    bird_train = load_jsonl(data_dir / "bird_train.jsonl")
    for r in spider_train:
        r.setdefault("source", "spider")
    for r in bird_train:
        r.setdefault("source", "bird")
    train_records = spider_train + bird_train
    random.Random(seed).shuffle(train_records)

    spider_dev = load_jsonl(data_dir / "spider_dev.jsonl")
    bird_dev = load_jsonl(data_dir / "bird_dev.jsonl")
    for r in spider_dev:
        r.setdefault("source", "spider")
    for r in bird_dev:
        r.setdefault("source", "bird")

    spider_val, spider_test_dev = stratified_dev_split(spider_dev, seed=seed)
    bird_val, bird_test_dev = stratified_dev_split(bird_dev, seed=seed)

    val_records = spider_val + bird_val
    test_records = spider_test_dev + bird_test_dev

    spider_held_out = load_jsonl(data_dir / "spider_test.jsonl")
    for r in spider_held_out:
        r.setdefault("source", "spider")

    def _to_dataset(records: List[dict]) -> Dataset:
        texts = [format_example(r, tokenizer) for r in records]
        return Dataset.from_dict(
            {
                "text": texts,
                "db_id": [r["db_id"] for r in records],
                "question": [r["question"] for r in records],
                "sql": [r["sql"] for r in records],
                "source": [r.get("source", "unknown") for r in records],
                "complexity": [r.get("complexity", "unknown") for r in records],
            }
        )

    splits = {
        "train": _to_dataset(train_records),
        "val": _to_dataset(val_records),
        "test": _to_dataset(test_records),
        "test_spider_held_out": _to_dataset(spider_held_out),
    }

    print(
        f"[dataset] train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}, "
        f"held_out={len(splits['test_spider_held_out'])}"
    )

    return splits


def build_db_path_index(
    records: List[dict],
    spider_db_dir: str | Path,
    spider_test_db_dir: str | Path,
    bird_train_db_dir: str | Path,
    bird_dev_db_dir: str | Path,
) -> Dict[str, str]:
    index: Dict[str, str] = {}
    spider_db_dir = Path(spider_db_dir)
    spider_test_db_dir = Path(spider_test_db_dir)
    bird_train_db_dir = Path(bird_train_db_dir)
    bird_dev_db_dir = Path(bird_dev_db_dir)

    for rec in records:
        db_id = rec["db_id"]
        if db_id in index:
            continue

        found = None
        for base in [
            spider_db_dir,
            spider_test_db_dir,
            bird_train_db_dir,
            bird_dev_db_dir,
        ]:
            candidate = base / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                found = str(candidate)
                break

        if found:
            index[db_id] = found
        else:
            index[db_id] = ""

    return index
