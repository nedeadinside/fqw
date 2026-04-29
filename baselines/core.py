from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.lora_config import get_bnb_config

MODEL_NAME = "Qwen/Qwen2.5-3B"
PROCESSED_DATA_DIR = "processed_data"
ARTIFACTS_ROOT = "artifacts/experiments"
CHAT_TEMPLATE_PATH = "templates/qwen_chat_template.jinja"
SPIDER_DB_DIR = "raw_data/Spider/database"
SPIDER_TEST_DB_DIR = "raw_data/Spider/test_database"
SPIDER_TABLES_JSON = "raw_data/Spider/tables.json"
SEED = 42
N_SHOTS = 3
MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 3584
BATCH_SIZE = 4
LOAD_IN_4BIT = True
EXECUTION_TIMEOUT = 30.0

SQL_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>")
SQL_START_KEYWORDS = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")

SYSTEM_ZS = (
    "You are a SQL expert specialized in SQLite. Given the database schema "
    "and a question in natural language, output ONLY the SQL query that "
    "answers the question, wrapped in a fenced code block with the `sql` tag. "
    "Do not include explanations, comments, or any other text."
)

SYSTEM_FS = (
    "You are a SQL expert specialized in SQLite. Given the database schema "
    "and a question in natural language, output the SQL query that answers "
    "the question. Follow exactly the format shown in the examples: respond "
    "with ONLY the SQL query wrapped in a fenced code block with the `sql` "
    "tag, no extra text."
)


def build_user_turn(example: dict) -> str:
    return (
        "Database schema (SQLite DDL):\n"
        f"{example['schema']}\n\n"
        f"Question: {example['question']}"
    )


def _wrap_sql(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def build_zs_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_ZS},
        {"role": "user", "content": build_user_turn(example)},
    ]


def build_fs_messages(example: dict, shots: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_FS}]
    for shot in shots:
        messages.append({"role": "user", "content": build_user_turn(shot)})
        messages.append({"role": "assistant", "content": _wrap_sql(shot["sql"])})
    messages.append({"role": "user", "content": build_user_turn(example)})
    return messages


def sample_few_shot(val_records: list[dict], k: int = N_SHOTS, seed: int = SEED) -> list[dict]:
    spider_only = [r for r in val_records if r.get("source") == "spider"]
    if not spider_only:
        raise ValueError("No spider records in val.jsonl for few-shot pool")

    by_len = sorted(spider_only, key=lambda r: len(r["sql"]))
    n = len(by_len)
    buckets = [
        by_len[: n // 3],
        by_len[n // 3 : 2 * n // 3],
        by_len[2 * n // 3 :],
    ]

    rng = random.Random(seed)
    chosen: list[dict] = []
    used_dbs: set[str] = set()

    for bucket in buckets[:k]:
        candidates = [r for r in bucket if r["db_id"] not in used_dbs]
        if not candidates:
            candidates = bucket
        pick = rng.choice(candidates)
        chosen.append(pick)
        used_dbs.add(pick["db_id"])

    return chosen


def _extract_assistant_text(generated_text: str) -> str:
    text = generated_text
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    for stop in SQL_STOP_TOKENS:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def _looks_like_sql(s: str) -> bool:
    head = s.lstrip().upper()
    return any(head.startswith(kw) for kw in SQL_START_KEYWORDS)


def extract_sql_plain(generated_text: str) -> str:
    text = _extract_assistant_text(generated_text)

    m = re.search(r"```sql\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m and _looks_like_sql(m.group(1)):
        return m.group(1).strip()

    m = re.search(r"\b(SELECT|WITH)\b.*?;", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip().rstrip(";").strip()

    m = re.search(r"\b(SELECT|WITH)\b.*", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()

    return ""


def setup_tokenizer_base(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    template_path = Path(CHAT_TEMPLATE_PATH)
    if not template_path.exists():
        raise FileNotFoundError(f"Chat template not found: {template_path}")
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(model_name: str = MODEL_NAME, load_in_4bit: bool = LOAD_IN_4BIT):
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model


def _build_prompt(example: dict, shots: list[dict] | None, tokenizer) -> str:
    messages = build_fs_messages(example, shots) if shots else build_zs_messages(example)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_predictions_baseline(
    records: list[dict],
    shots: list[dict] | None,
    model,
    tokenizer,
) -> list[dict]:
    torch.manual_seed(SEED)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generation_config = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    predictions: list[dict] = []
    pbar = tqdm(
        range(0, len(records), BATCH_SIZE),
        total=(len(records) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="generate",
        unit="batch",
    )
    for start in pbar:
        batch = records[start : start + BATCH_SIZE]
        prompts = [_build_prompt(ex, shots, tokenizer) for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[:, input_len:]

        for j, ex in enumerate(batch):
            decoded = tokenizer.decode(gen_tokens[j], skip_special_tokens=False)
            predictions.append(
                {
                    "example_id": ex.get("example_id", f"{start + j}"),
                    "source": ex.get("source", "unknown"),
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "gold_sql": ex["sql"],
                    "predicted_evidence": "",
                    "predicted_sql": extract_sql_plain(decoded),
                }
            )

    return predictions


def save_predictions(preds: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def save_manifest(experiment_dir: Path, payload: dict) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def evaluate_predictions(experiment_dir: Path) -> dict:
    from src.evaluation.evaluate import evaluate

    predictions_path = experiment_dir / "predictions" / "test_predictions.jsonl"
    metrics_path = experiment_dir / "metrics" / "test_metrics.json"

    cfg: dict[str, Any] = {
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "spider_db_dir": SPIDER_DB_DIR,
        "spider_test_db_dir": SPIDER_TEST_DB_DIR,
        "execution_timeout": EXECUTION_TIMEOUT,
    }
    if Path(SPIDER_TABLES_JSON).exists():
        cfg["spider_tables_json"] = SPIDER_TABLES_JSON

    return evaluate(cfg_override=cfg)


def run_baseline(
    mode: str,
    experiment_id: str,
    test_records: list[dict],
    val_records: list[dict],
    model,
    tokenizer,
) -> dict:
    if mode not in {"zero-shot", "few-shot"}:
        raise ValueError(f"Unknown mode: {mode}")

    experiment_dir = Path(ARTIFACTS_ROOT) / experiment_id
    (experiment_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)

    shots: list[dict] | None = None
    shots_meta: list[dict] = []
    if mode == "few-shot":
        shots = sample_few_shot(val_records, N_SHOTS, SEED)
        shots_meta = [
            {
                "example_id": s.get("example_id"),
                "db_id": s["db_id"],
                "complexity": s.get("complexity", "unknown"),
                "sql_len": len(s["sql"]),
            }
            for s in shots
        ]

    preds = generate_predictions_baseline(test_records, shots, model, tokenizer)

    predictions_path = experiment_dir / "predictions" / "test_predictions.jsonl"
    save_predictions(preds, predictions_path)

    save_manifest(
        experiment_dir,
        {
            "experiment_id": experiment_id,
            "kind": "baseline",
            "mode": mode,
            "model": MODEL_NAME,
            "split": "test",
            "n_examples": len(preds),
            "shots": shots_meta,
            "shots_source": f"{PROCESSED_DATA_DIR}/val.jsonl",
            "predictions_path": str(predictions_path),
            "generation": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_input_length": MAX_INPUT_LENGTH,
                "batch_size": BATCH_SIZE,
                "do_sample": False,
                "num_beams": 1,
                "load_in_4bit": LOAD_IN_4BIT,
                "seed": SEED,
            },
        },
    )

    return evaluate_predictions(experiment_dir)
