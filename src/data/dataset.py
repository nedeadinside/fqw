"""
Загрузка JSONL-датасетов, форматирование промптов и разбиение на сплиты.

Каждый JSONL-файл содержит записи с полями:
    example_id, db_id, question, sql, schema

Стратегия сплитов (из PLAN.md):
    train  : spider_train + bird_train  (18 087 примеров)
    val    : 50% spider_dev + 50% bird_dev, стратификация по db_id  (~1 284)
    test   : оставшиеся 50% dev                                       (~1 284)
    test_spider_held_out : spider_test                                (2 147)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a SQL expert. Given the database schema and a question "
    "in natural language, generate the corresponding SQL query for SQLite."
)

CUSTOM_SPECIAL_TOKENS = ["<schema>", "</schema>", "<question>", "</question>"]


# ---------------------------------------------------------------------------
# Форматирование одного примера в текст для SFT
# ---------------------------------------------------------------------------

def format_example(example: dict, tokenizer) -> str:
    """Превращает JSONL-запись в строку для SFTTrainer.

    Использует нативный apply_chat_template от Qwen (ChatML).
    Кастомные токены <schema> / <question> уже в словаре к этому моменту.
    """
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


def format_example_plain(example: dict, tokenizer) -> str:
    """То же, но БЕЗ кастомных special tokens (для ablation E4).

    Теги <schema>/<question> присутствуют как plain text — токенизируются
    в подтокены, т.к. они не добавлены в словарь токенизатора.
    """
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


# ---------------------------------------------------------------------------
# Загрузка одного JSONL-файла
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Стратифицированное разбиение dev-сплита по db_id
# ---------------------------------------------------------------------------

def stratified_dev_split(
    records: List[dict],
    val_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[List[dict], List[dict]]:
    """Делит список записей на (val, test) стратифицированно по db_id.

    Все примеры одной БД попадают целиком в один сплит — исключает
    data leakage внутри БД.
    """
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


# ---------------------------------------------------------------------------
# Публичный интерфейс: загрузка всех сплитов
# ---------------------------------------------------------------------------

def load_splits(
    processed_data_dir: str | Path,
    tokenizer,
    max_seq_length: int = 2048,
    use_custom_tokens: bool = True,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Загружает все сплиты и возвращает HuggingFace Dataset-ы.

    Returns dict с ключами:
        "train", "val", "test", "test_spider_held_out"
    """
    data_dir = Path(processed_data_dir)

    fmt = format_example if use_custom_tokens else format_example_plain

    # --- Train ---
    spider_train = load_jsonl(data_dir / "spider_train.jsonl")
    bird_train = load_jsonl(data_dir / "bird_train.jsonl")
    # Inject source для JSONL-файлов, сгенерированных до добавления этого поля
    for r in spider_train:
        r.setdefault("source", "spider")
    for r in bird_train:
        r.setdefault("source", "bird")
    train_records = spider_train + bird_train
    random.Random(seed).shuffle(train_records)

    # --- Dev → val + test ---
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

    # --- Spider held-out test ---
    spider_held_out = load_jsonl(data_dir / "spider_test.jsonl")
    for r in spider_held_out:
        r.setdefault("source", "spider")

    def _to_dataset(records: List[dict]) -> Dataset:
        texts = [fmt(r, tokenizer) for r in records]
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

    print(f"[dataset] train={len(splits['train'])}, "
          f"val={len(splits['val'])}, "
          f"test={len(splits['test'])}, "
          f"held_out={len(splits['test_spider_held_out'])}")

    return splits


# ---------------------------------------------------------------------------
# Вспомогательная функция: путь к .sqlite по db_id
# ---------------------------------------------------------------------------

def resolve_db_path(
    db_id: str,
    source: str,
    spider_db_dir: str | Path,
    spider_test_db_dir: str | Path,
    bird_train_db_dir: str | Path,
    bird_dev_db_dir: str | Path,
    split: Optional[str] = None,
) -> Optional[Path]:
    """Возвращает Path к .sqlite-файлу для заданного db_id.

    source: "spider" | "bird"
    split:  "train" | "dev" | "test"  (для определения директории у BIRD)
    """
    if source == "spider":
        for base in [Path(spider_db_dir), Path(spider_test_db_dir)]:
            candidate = base / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                return candidate
    elif source == "bird":
        for base in [Path(bird_train_db_dir), Path(bird_dev_db_dir)]:
            candidate = base / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                return candidate
    return None


def build_db_path_index(
    records: List[dict],
    spider_db_dir: str | Path,
    spider_test_db_dir: str | Path,
    bird_train_db_dir: str | Path,
    bird_dev_db_dir: str | Path,
) -> Dict[str, str]:
    """Строит словарь {db_id: /path/to/db.sqlite} для списка записей.

    Определяет источник (Spider vs BIRD) по наличию db_id в соответствующих
    директориях — не требует поля 'source' в JSONL.
    """
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
        for base in [spider_db_dir, spider_test_db_dir,
                     bird_train_db_dir, bird_dev_db_dir]:
            candidate = base / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                found = str(candidate)
                break

        if found:
            index[db_id] = found
        else:
            # Запись без БД — оценка будет пропущена для этого db_id
            index[db_id] = ""

    return index
