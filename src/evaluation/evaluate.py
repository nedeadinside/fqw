from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    SYSTEM_PROMPT,
    build_db_path_index,
    load_jsonl,
    stratified_dev_split,
)
from src.evaluation.metrics import compute_all_metrics

DEFAULT_CONFIG_PATH = "configs/training_config.yaml"
RUN_ID = "E2"


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

    for stop in ["<|im_end|>", "<|endoftext|>", "</s>"]:
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


def generate_predictions(
    records: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[dict]:
    model.eval()

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        repetition_penalty=1.1,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    predictions = []
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        prompts = []
        for ex in batch:
            prompts.append(make_inference_prompt(ex, tokenizer))

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

        for j, (ex, gen_text) in enumerate(zip(batch, decoded)):
            sql = extract_sql(gen_text)
            pred = {
                "example_id": ex.get("example_id", f"{i + j}"),
                "db_id": ex["db_id"],
                "question": ex["question"],
                "predicted_sql": sql,
                "gold_sql": ex["sql"],
            }
            predictions.append(pred)

        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{total}] примеров обработано")

    return predictions


def load_model_for_inference(
    model_path: str,
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": CUSTOM_SPECIAL_TOKENS}
    )
    print(f"[inference] Кастомных токенов добавлено/найдено: {num_added}")

    load_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if load_in_4bit:
        from src.training.lora_config import get_bnb_config

        load_kwargs["quantization_config"] = get_bnb_config()
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    try:
        from peft import PeftModel

        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            base_model_name = json.loads(adapter_config.read_text())[
                "base_model_name_or_path"
            ]
            base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
            base.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base, model_path)
            print(f"[inference] Загружена PeftModel (LoRA adapter) из {model_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            print(f"[inference] Загружена merged модель из {model_path}")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        print(f"[inference] Загружена модель из {model_path}")

    model.eval()
    return model, tokenizer


def evaluate(
    model_path: str | None = None,
    split: str = "test",
    config_path: str = DEFAULT_CONFIG_PATH,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    output_dir: str = "results",
) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if model_path is None:
        model_path = str(Path(cfg["output_dir"]) / RUN_ID / "best")

    data_dir = Path(cfg["processed_data_dir"])
    allowed_splits = {"val", "test", "test_spider_held_out"}
    if split not in allowed_splits:
        raise ValueError(
            f"Unsupported split: {split}. Allowed: {sorted(allowed_splits)}"
        )

    if split == "test_spider_held_out":
        records = load_jsonl(data_dir / "spider_test.jsonl")
    else:
        spider_dev = load_jsonl(data_dir / "spider_dev.jsonl")
        bird_dev = load_jsonl(data_dir / "bird_dev.jsonl")

        spider_val, spider_test = stratified_dev_split(spider_dev)
        bird_val, bird_test = stratified_dev_split(bird_dev)

        if split == "val":
            records = spider_val + bird_val
        else:
            records = spider_test + bird_test

    print(f"[eval] Сплит: {split}, примеров: {len(records)}")

    db_paths = build_db_path_index(
        records,
        spider_db_dir=cfg["spider_db_dir"],
        spider_test_db_dir=cfg["spider_test_db_dir"],
        bird_train_db_dir=cfg["bird_train_db_dir"],
        bird_dev_db_dir=cfg["bird_dev_db_dir"],
    )

    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        load_in_4bit=cfg.get("load_in_4bit", False),
    )

    print("[eval] Генерация SQL предсказаний...")
    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    out_dir = Path(output_dir)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{RUN_ID}_{split}_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"[eval] Предсказания сохранены: {preds_path}")

    metrics = compute_all_metrics(
        predictions=predictions,
        db_paths=db_paths,
    )
    metrics["run_id"] = RUN_ID
    metrics["split"] = split
    metrics["n_predictions"] = len(predictions)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{RUN_ID}_{split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] Метрики сохранены: {metrics_path}")

    _print_results_table(RUN_ID, split, metrics)

    return metrics


def _print_results_table(run_id: str, split: str, metrics: dict):
    print(f"\n{'=' * 50}")
    print(f"Результаты: {run_id} | split={split}")
    print(f"{'=' * 50}")
    print(f"  EX  (Execution Accuracy): {metrics.get('ex', 0):.4f}")
    print(f"  EM  (Exact Match):        {metrics.get('em', 0):.4f}")
    print(f"  VSR (Valid SQL Rate):     {metrics.get('vsr', 0):.4f}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    evaluate()
