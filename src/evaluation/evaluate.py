from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    SYSTEM_PROMPT,
    build_db_path_index,
    load_jsonl,
    stratified_dev_split,
)
from src.evaluation.metrics import compute_all_metrics

ALLOWED_SPLITS = {"val", "test", "test_spider_held_out"}
REQUIRED_CONFIG_KEYS = (
    "processed_data_dir",
    "spider_db_dir",
    "spider_test_db_dir",
    "bird_train_db_dir",
    "bird_dev_db_dir",
)
QWEN_TEMPLATE_MARKERS = (
    "<|im_start|>system",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "<|im_end|>",
    "add_generation_prompt",
)
SQL_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve_optional_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_candidate = Path(__file__).resolve().parents[2] / path
    if project_candidate.exists():
        return project_candidate

    return candidate


def _resolve_config_path(config_path: str) -> str:
    resolved = _resolve_optional_path(config_path)
    if resolved.exists():
        return str(resolved)
    raise FileNotFoundError(f"Eval config not found: {config_path}")


def _validate_qwen_template_tokens(template_text: str, template_path: Path) -> None:
    missing = [m for m in QWEN_TEMPLATE_MARKERS if m not in template_text]
    if missing:
        raise ValueError(
            f"Chat template is missing required Qwen markers {missing} in {template_path}"
        )


def merge_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg_path = cfg.get("train_config_path")
    if not train_cfg_path:
        return cfg

    resolved = _resolve_optional_path(str(train_cfg_path))
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced train config not found: {train_cfg_path}")

    train_cfg = load_config(resolved)
    for key in ("output_dir", "processed_data_dir", "load_in_4bit"):
        if key not in cfg and key in train_cfg:
            cfg[key] = train_cfg[key]
    return cfg


def make_inference_prompt(example: dict, tokenizer) -> str:
    user_content = (
        f"<schema>\n{example['schema']}\n</schema>\n\n"
        f"<question>\n{example['question']}\n</question>"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_sql(generated_text: str) -> str:
    sql = generated_text
    if "<|im_start|>assistant" in sql:
        sql = sql.split("<|im_start|>assistant")[-1]

    for stop in SQL_STOP_TOKENS:
        if stop in sql:
            sql = sql[: sql.index(stop)]

    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        sql = "\n".join(inner).strip()

    return sql


def setup_tokenizer(
    model_path: str,
    chat_template_path: str | None = None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if chat_template_path:
        template_file = _resolve_optional_path(chat_template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
        template_text = template_file.read_text(encoding="utf-8")
        _validate_qwen_template_tokens(template_text, template_file)
        tokenizer.chat_template = template_text

    tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_SPECIAL_TOKENS})
    return tokenizer


def load_model(
    model_path: str,
    tokenizer,
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        from src.training.lora_config import get_bnb_config

        load_kwargs["quantization_config"] = get_bnb_config()

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model_name = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    return model


def generate_predictions(
    records: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    max_input_length: int = 3072,
    do_sample: bool = False,
    num_beams: int = 1,
    seed: int = 42,
) -> list[dict]:
    model.eval()
    torch.manual_seed(seed)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    predictions = []
    for i, ex in enumerate(records):
        prompt = make_inference_prompt(ex, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        predictions.append(
            {
                "example_id": ex.get("example_id", f"{i}"),
                "db_id": ex["db_id"],
                "question": ex["question"],
                "predicted_sql": extract_sql(decoded),
                "gold_sql": ex["sql"],
            }
        )

    return predictions


def select_records(processed_data_dir: str | Path, split: str) -> list[dict]:
    data_dir = Path(processed_data_dir)

    if split == "test_spider_held_out":
        return load_jsonl(data_dir / "spider_test.jsonl")

    spider_dev = load_jsonl(data_dir / "spider_dev.jsonl")
    bird_dev = load_jsonl(data_dir / "bird_dev.jsonl")
    spider_val, spider_test = stratified_dev_split(spider_dev)
    bird_val, bird_test = stratified_dev_split(bird_dev)

    if split == "val":
        return spider_val + bird_val
    return spider_test + bird_test


def _resolve_model_path(
    cfg: dict[str, Any],
    run_id: str,
    model_path_override: str | None,
) -> str:
    if model_path_override:
        return model_path_override
    if "output_dir" not in cfg:
        raise ValueError("output_dir is required in eval config when model_path is not set")
    return str(Path(cfg["output_dir"]) / run_id / "best")


def _save_predictions(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def evaluate(
    config_path: str,
    chat_template_path: str | None = None,
    run_id: str = "E2",
    model_path_override: str | None = None,
) -> dict:
    cfg = merge_train_config(load_config(_resolve_config_path(config_path)))

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Eval config missing required keys: {missing}")

    split = cfg.get("split", "test")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split: {split}. Allowed: {sorted(ALLOWED_SPLITS)}")

    model_path = _resolve_model_path(cfg, run_id, model_path_override)
    records = select_records(cfg["processed_data_dir"], split)

    db_paths = build_db_path_index(
        records,
        spider_db_dir=cfg["spider_db_dir"],
        spider_test_db_dir=cfg["spider_test_db_dir"],
        bird_train_db_dir=cfg["bird_train_db_dir"],
        bird_dev_db_dir=cfg["bird_dev_db_dir"],
    )

    tokenizer = setup_tokenizer(model_path, chat_template_path=chat_template_path)
    model = load_model(model_path, tokenizer, load_in_4bit=cfg.get("load_in_4bit", False))

    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.get("max_new_tokens", 512),
        max_input_length=cfg.get("max_input_length", 3072),
        do_sample=cfg.get("do_sample", False),
        num_beams=cfg.get("num_beams", 1),
        seed=cfg.get("seed", 42),
    )

    results_dir = Path(cfg.get("results_dir", "./results"))
    _save_predictions(predictions, results_dir / "predictions" / f"{run_id}_{split}_predictions.jsonl")

    metrics = compute_all_metrics(
        predictions=predictions,
        db_paths=db_paths,
        timeout=cfg.get("execution_timeout", 30.0),
    )
    metrics.update({
        "run_id": run_id,
        "split": split,
        "n_predictions": len(predictions),
    })

    _save_metrics(metrics, results_dir / "metrics" / f"{run_id}_{split}_metrics.json")
    return metrics


if __name__ == "__main__":
    EVAL_CONFIG_PATH = "configs/eval_qwen.yaml"
    EVAL_CHAT_TEMPLATE_PATH: str | None = "templates/qwen_chat_template.jinja"
    EVAL_RUN_ID = "E2"
    EVAL_MODEL_PATH: str | None = None

    metrics = evaluate(
        config_path=EVAL_CONFIG_PATH,
        chat_template_path=EVAL_CHAT_TEMPLATE_PATH,
        run_id=EVAL_RUN_ID,
        model_path_override=EVAL_MODEL_PATH,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
