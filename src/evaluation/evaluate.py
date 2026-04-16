from __future__ import annotations

import argparse
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


def make_inference_prompt(
    example: dict, tokenizer, use_custom_tokens: bool = True
) -> str:
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


def make_few_shot_prompt(
    example: dict,
    few_shot_examples: list[dict],
    tokenizer,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for fs_ex in few_shot_examples:
        messages.append(
            {
                "role": "user",
                "content": f"Schema:\n{fs_ex['schema']}\n\nQuestion: {fs_ex['question']}",
            }
        )
        messages.append({"role": "assistant", "content": fs_ex["sql"]})

    messages.append(
        {
            "role": "user",
            "content": f"Schema:\n{example['schema']}\n\nQuestion: {example['question']}",
        }
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_sql(generated_text: str, prompt: str) -> str:
    if generated_text.startswith(prompt):
        sql = generated_text[len(prompt) :]
    else:
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
    use_custom_tokens: bool = True,
    few_shot_examples: list[dict] | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    source_label: str | None = None,
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
            if few_shot_examples:
                prompt = make_few_shot_prompt(ex, few_shot_examples, tokenizer)
            else:
                prompt = make_inference_prompt(ex, tokenizer, use_custom_tokens)
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,  # оставляем место для генерации
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        input_lengths = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_lengths:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for j, (ex, gen_text, prompt) in enumerate(zip(batch, decoded, prompts)):
            sql = extract_sql(gen_text, "")
            pred = {
                "example_id": ex.get("example_id", f"{i + j}"),
                "db_id": ex["db_id"],
                "question": ex["question"],
                "predicted_sql": sql,
                "gold_sql": ex["sql"],
                "source": source_label or _infer_source(ex),
                "complexity": ex.get("complexity", "unknown"),
            }
            predictions.append(pred)

        if (i // batch_size) % 10 == 0:
            print(f"  [{i + len(batch)}/{total}] примеров обработано")

    return predictions


def _infer_source(example: dict) -> str:
    source = example.get("source", "")
    if source in ("spider", "bird"):
        return source
    return "unknown"


def load_model_for_inference(
    model_path: str,
    use_custom_tokens: bool = True,
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_custom_tokens:
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
    model_path: str,
    split: str,
    config_path: str,
    experiment_id: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    baseline_mode: str | None = None,
    output_dir: str = "results",
):
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["processed_data_dir"])
    use_custom_tokens = (baseline_mode is None) and (experiment_id != "E4")

    spider_complexity: dict = {}
    bird_complexity: dict = {}
    try:
        spider_complexity = _build_complexity_lookup(
            Path(cfg["spider_db_dir"]).parent / "dev.json", "hardness"
        )
        bird_complexity = _build_complexity_lookup(
            Path(cfg["bird_dev_db_dir"]).parent / "dev.json", "difficulty"
        )
    except Exception:
        pass

    if split == "test_spider_held_out":
        records = load_jsonl(data_dir / "spider_test.jsonl")
        for r in records:
            r.setdefault("source", "spider")
        source_label = "spider"
    else:
        spider_dev = load_jsonl(data_dir / "spider_dev.jsonl")
        bird_dev = load_jsonl(data_dir / "bird_dev.jsonl")

        for r in spider_dev:
            r.setdefault("source", "spider")
            r.setdefault(
                "complexity", spider_complexity.get(r["example_id"], "unknown")
            )
        for r in bird_dev:
            r.setdefault("source", "bird")
            r.setdefault("complexity", bird_complexity.get(r["example_id"], "unknown"))

        spider_val, spider_test = stratified_dev_split(spider_dev)
        bird_val, bird_test = stratified_dev_split(bird_dev)

        if split == "val":
            records = spider_val + bird_val
        else:
            records = spider_test + bird_test
        source_label = None

    print(f"[eval] Сплит: {split}, примеров: {len(records)}")

    db_paths = build_db_path_index(
        records,
        spider_db_dir=cfg["spider_db_dir"],
        spider_test_db_dir=cfg["spider_test_db_dir"],
        bird_train_db_dir=cfg["bird_train_db_dir"],
        bird_dev_db_dir=cfg["bird_dev_db_dir"],
    )

    if baseline_mode is None and model_path:
        model, tokenizer = load_model_for_inference(
            model_path, use_custom_tokens=use_custom_tokens
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = cfg["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        use_custom_tokens = False

    few_shot_examples = None
    if baseline_mode == "few_shot":
        train_records = load_jsonl(data_dir / "spider_train.jsonl")[:1000]
        few_shot_examples = train_records[:3]

    print("[eval] Генерация SQL предсказаний...")
    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        use_custom_tokens=use_custom_tokens,
        few_shot_examples=few_shot_examples,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        source_label=source_label,
    )

    out_dir = Path(output_dir)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{experiment_id}_{split}_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"[eval] Предсказания сохранены: {preds_path}")

    metrics = compute_all_metrics(
        predictions=predictions,
        db_paths=db_paths,
    )
    metrics["experiment_id"] = experiment_id
    metrics["split"] = split
    metrics["n_predictions"] = len(predictions)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{experiment_id}_{split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] Метрики сохранены: {metrics_path}")

    _print_results_table(experiment_id, split, metrics)

    return metrics


def _build_complexity_lookup(json_path: Path, field: str) -> dict:
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            examples = json.load(f)
        return {i: ex.get(field, "unknown") for i, ex in enumerate(examples)}
    except Exception:
        return {}


def _print_results_table(experiment_id: str, split: str, metrics: dict):
    print(f"\n{'=' * 50}")
    print(f"Результаты: {experiment_id} | split={split}")
    print(f"{'=' * 50}")
    print(f"  EX  (Execution Accuracy): {metrics.get('ex', 0):.4f}")
    print(f"  EM  (Exact Match):        {metrics.get('em', 0):.4f}")
    print(f"  VSR (Valid SQL Rate):     {metrics.get('vsr', 0):.4f}")
    if "ves" in metrics:
        print(f"  VES (Valid Eff. Score):   {metrics['ves']:.4f}")
    if "bleu" in metrics:
        print(f"  BLEU:                     {metrics['bleu']:.4f}")
    if "rouge_l" in metrics:
        print(f"  ROUGE-L:                  {metrics['rouge_l']:.4f}")

    if metrics.get("ex_by_source"):
        print("\n  EX по источнику:")
        for src, score in sorted(metrics["ex_by_source"].items()):
            print(f"    {src:12s}: {score:.4f}")

    if metrics.get("ex_by_complexity"):
        print("\n  EX по сложности:")
        order = ["easy", "medium", "hard", "extra", "unknown"]
        for lvl in order:
            if lvl in metrics["ex_by_complexity"]:
                print(f"    {lvl:8s}: {metrics['ex_by_complexity'][lvl]:.4f}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка NL-to-SQL модели")
    parser.add_argument("--model_path", default=None, help="Путь к дообученной модели")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "test_spider_held_out"],
        help="Сплит для оценки",
    )
    parser.add_argument(
        "--experiment",
        default="E2",
        help="ID эксперимента (E0-ZS, E0-FS, E1, E2, E3, E4, E5)",
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Путь к конфигу",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        choices=["zero_shot", "few_shot"],
        help="Режим baseline без fine-tuning",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        split=args.split,
        config_path=args.config,
        experiment_id=args.experiment,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        baseline_mode=args.baseline,
        output_dir=args.output_dir,
    )
