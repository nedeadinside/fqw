from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a SQL expert. Given the database schema and a question "
    "in natural language, generate the corresponding SQL query for SQLite."
)

CUSTOM_SPECIAL_TOKENS = [
    "<schema>",
    "</schema>",
    "<question>",
    "</question>",
    "<evidence>",
    "</evidence>",
]


def format_example(example: dict, tokenizer) -> str:
    evidence = example.get("evidence", "")
    if evidence:
        assistant_content = f"<evidence>\n{evidence}\n</evidence>\n\n{example['sql']}"
    else:
        assistant_content = example["sql"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<schema>\n{example['schema']}\n</schema>\n\n"
                f"<question>\n{example['question']}\n</question>"
            ),
        },
        {"role": "assistant", "content": assistant_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_jsonl(path: str | Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _tag_source(records: List[dict], source: str) -> List[dict]:
    for r in records:
        r.setdefault("source", source)
    return records


def _to_dataset(records: List[dict], tokenizer) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [format_example(r, tokenizer) for r in records],
            "db_id": [r["db_id"] for r in records],
            "question": [r["question"] for r in records],
            "sql": [r["sql"] for r in records],
            "source": [r.get("source", "unknown") for r in records],
            "complexity": [r.get("complexity", "unknown") for r in records],
        }
    )


def load_splits(
    processed_data_dir: str | Path,
    tokenizer,
) -> Dict[str, Dataset]:
    data_dir = Path(processed_data_dir)

    train_records = _tag_source(load_jsonl(data_dir / "train.jsonl"), "spider")
    val_records = _tag_source(load_jsonl(data_dir / "val.jsonl"), "spider")
    test_records = _tag_source(load_jsonl(data_dir / "test.jsonl"), "spider")

    return {
        "train": _to_dataset(train_records, tokenizer),
        "val": _to_dataset(val_records, tokenizer),
        "test": _to_dataset(test_records, tokenizer),
    }


def build_db_path_index(
    records: List[dict],
    spider_db_dir: str | Path,
    spider_test_db_dir: str | Path,
) -> Dict[str, str]:
    bases = [
        Path(spider_db_dir),
        Path(spider_test_db_dir),
    ]

    index: Dict[str, str] = {}
    for rec in records:
        db_id = rec["db_id"]
        if db_id in index:
            continue

        found = ""
        for base in bases:
            candidate = base / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                found = str(candidate)
                break
        index[db_id] = found

    return index
