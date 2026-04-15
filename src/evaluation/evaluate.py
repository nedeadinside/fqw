from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    SYSTEM_PROMPT,
    apply_project_chat_template,
    build_db_path_index,
    load_jsonl,
    stratified_dev_split,
)
from src.evaluation.metrics import compute_all_metrics

EXPERIMENT_ID = "E2"


def make_inference_prompt(example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": {
                "schema": example["schema"],
                "question": example["question"],
            },
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_sql(generated_text: str, prompt: str) -> str:
    if generated_text.startswith(prompt):
        sql_text = generated_text[len(prompt) :]
    else:
        sql_text = generated_text

    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        if stop_token in sql_text:
            sql_text = sql_text[: sql_text.index(stop_token)]

    sql_text = sql_text.strip()

    if sql_text.startswith("```"):
        lines = sql_text.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        sql_text = "\n".join(inner).strip()

    return sql_text


def generate_predictions(
    records: List[dict],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    source_label: Optional[str] = None,
) -> List[dict]:
    model.eval()

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "repetition_penalty": 1.1,
        "eos_token_id": eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    predictions: List[dict] = []
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        prompts = [make_inference_prompt(record, tokenizer) for record in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        input_lengths = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_lengths:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for index, (record, text_output, prompt) in enumerate(
            zip(batch, decoded, prompts)
        ):
            _ = prompt
            sql_text = extract_sql(text_output, "")
            predictions.append(
                {
                    "example_id": record.get("example_id", f"{start + index}"),
                    "db_id": record["db_id"],
                    "question": record["question"],
                    "predicted_sql": sql_text,
                    "gold_sql": record["sql"],
                    "source": source_label or infer_source(record),
                    "complexity": record.get("complexity", "unknown"),
                }
            )

        if (start // batch_size) % 10 == 0:
            print(f"[eval] {start + len(batch)}/{total}")

    return predictions


def infer_source(example: dict) -> str:
    source = example.get("source", "")
    if source in ("spider", "bird"):
        return source
    return "unknown"


def load_model_for_inference(model_path: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = apply_project_chat_template(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_SPECIAL_TOKENS})

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        config_data = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_name = config_data["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **load_kwargs
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    return model, tokenizer


def build_complexity_lookup(json_path: Path, field: str) -> dict:
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as file_obj:
            examples = json.load(file_obj)
    except Exception:
        return {}
    return {
        index: example.get(field, "unknown") for index, example in enumerate(examples)
    }


def resolve_model_path(config: dict, model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    return str(Path(config["output_dir"]) / EXPERIMENT_ID / "best")


def evaluate(
    model_path: Optional[str],
    split: str,
    config_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    output_dir: str = "results",
):
    with open(config_path, encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    data_dir = Path(config["processed_data_dir"])

    spider_complexity = build_complexity_lookup(
        Path(config["spider_db_dir"]).parent / "dev.json", "hardness"
    )
    bird_complexity = build_complexity_lookup(
        Path(config["bird_dev_db_dir"]).parent / "dev.json", "difficulty"
    )

    if split == "test_spider_held_out":
        records = load_jsonl(data_dir / "spider_test.jsonl")
        for record in records:
            record.setdefault("source", "spider")
        source_label = "spider"
    else:
        spider_dev = load_jsonl(data_dir / "spider_dev.jsonl")
        bird_dev = load_jsonl(data_dir / "bird_dev.jsonl")

        for record in spider_dev:
            record.setdefault("source", "spider")
            record.setdefault(
                "complexity", spider_complexity.get(record["example_id"], "unknown")
            )
        for record in bird_dev:
            record.setdefault("source", "bird")
            record.setdefault(
                "complexity", bird_complexity.get(record["example_id"], "unknown")
            )

        spider_val, spider_test = stratified_dev_split(spider_dev)
        bird_val, bird_test = stratified_dev_split(bird_dev)

        if split == "val":
            records = spider_val + bird_val
        else:
            records = spider_test + bird_test
        source_label = None

    print(f"[eval] split={split} size={len(records)}")

    db_paths = build_db_path_index(
        records,
        spider_db_dir=Path(config["spider_db_dir"]),
        spider_test_db_dir=Path(config["spider_test_db_dir"]),
        bird_train_db_dir=Path(config["bird_train_db_dir"]),
        bird_dev_db_dir=Path(config["bird_dev_db_dir"]),
    )

    resolved_model_path = resolve_model_path(config, model_path)
    model, tokenizer = load_model_for_inference(resolved_model_path)

    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        source_label=source_label,
    )

    out_dir = Path(output_dir)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{EXPERIMENT_ID}_{split}_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as file_obj:
        for prediction in predictions:
            file_obj.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics = compute_all_metrics(predictions=predictions, db_paths=db_paths)
    metrics["experiment_id"] = EXPERIMENT_ID
    metrics["split"] = split
    metrics["n_predictions"] = len(predictions)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{EXPERIMENT_ID}_{split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    print_results(split, metrics)
    return metrics


def print_results(split: str, metrics: dict):
    print(f"[metrics] split={split}")
    print(f"EX={metrics.get('ex', 0):.4f}")
    print(f"EM={metrics.get('em', 0):.4f}")
    print(f"VSR={metrics.get('vsr', 0):.4f}")
    if "ves" in metrics:
        print(f"VES={metrics['ves']:.4f}")
    if "bleu" in metrics:
        print(f"BLEU={metrics['bleu']:.4f}")
    if "rouge_l" in metrics:
        print(f"ROUGE-L={metrics['rouge_l']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main evaluation run")
    parser.add_argument("--model_path", default=None)
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "test_spider_held_out"],
    )
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        split=args.split,
        config_path=args.config,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )
