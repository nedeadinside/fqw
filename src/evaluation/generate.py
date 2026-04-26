from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    SYSTEM_PROMPT,
    load_jsonl,
)
from src.evaluation._config import (
    load_config,
    resolve_config_path,
    resolve_optional_path,
)

ALLOWED_SPLITS = {"val", "test"}
REQUIRED_CONFIG_KEYS = ("processed_data_dir", "predictions_path")
QWEN_TEMPLATE_MARKERS = (
    "<|im_start|>system",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "<|im_end|>",
    "add_generation_prompt",
)
SQL_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>")


def _validate_qwen_template_tokens(template_text: str, template_path: Path) -> None:
    missing = [m for m in QWEN_TEMPLATE_MARKERS if m not in template_text]
    if missing:
        raise ValueError(
            f"Chat template is missing required Qwen markers {missing} in {template_path}"
        )


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


def _extract_assistant_text(generated_text: str) -> str:
    text = generated_text
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    for stop in SQL_STOP_TOKENS:
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def extract_evidence(assistant_text: str) -> str:
    m = re.search(r"<evidence>(.*?)</evidence>", assistant_text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_sql(generated_text: str) -> str:
    text = _extract_assistant_text(generated_text)
    m = re.search(r"<sql>(.*?)</sql>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def setup_tokenizer(
    model_path: str,
    chat_template_path: str | None = None,
    custom_tokens: list[str] | None = None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if chat_template_path:
        template_file = resolve_optional_path(chat_template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
        template_text = template_file.read_text(encoding="utf-8")
        _validate_qwen_template_tokens(template_text, template_file)
        tokenizer.chat_template = template_text

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    tokenizer.add_special_tokens(
        {"additional_special_tokens": custom_tokens or CUSTOM_SPECIAL_TOKENS}
    )
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

        base_model_name = json.loads(adapter_config.read_text())[
            "base_model_name_or_path"
        ]
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
    batch_size: int = 1,
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

    predictions: list[dict] = []
    pbar = tqdm(
        range(0, len(records), batch_size),
        total=(len(records) + batch_size - 1) // batch_size,
        desc="generate",
        unit="batch",
    )
    for start in pbar:
        batch = records[start : start + batch_size]
        prompts = [make_inference_prompt(ex, tokenizer) for ex in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[:, input_len:]

        for j, ex in enumerate(batch):
            decoded = tokenizer.decode(gen_tokens[j], skip_special_tokens=False)
            assistant_text = _extract_assistant_text(decoded)
            predictions.append(
                {
                    "example_id": ex.get("example_id", f"{start + j}"),
                    "source": ex.get("source", "unknown"),
                    "db_id": ex["db_id"],
                    "question": ex["question"],
                    "gold_sql": ex["sql"],
                    "predicted_evidence": extract_evidence(assistant_text),
                    "predicted_sql": extract_sql(decoded),
                }
            )

    return predictions


def _tag(records: list[dict], source: str) -> list[dict]:
    for r in records:
        r.setdefault("source", source)
    return records


def select_records(processed_data_dir: str | Path, split: str) -> list[dict]:
    data_dir = Path(processed_data_dir)
    return _tag(load_jsonl(data_dir / f"{split}.jsonl"), "spider")


def _resolve_model_path(
    cfg: dict[str, Any],
    model_path_override: str | None,
) -> str:
    if model_path_override:
        return str(resolve_optional_path(model_path_override))
    if "model_path" in cfg:
        return str(resolve_optional_path(str(cfg["model_path"])))
    if "best_model_dir" in cfg:
        return str(resolve_optional_path(str(cfg["best_model_dir"])))
    raise ValueError("Config must contain one of: model_path, best_model_dir")


def _save_predictions(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def generate(
    config_path: str | None = None,
    chat_template_path: str | None = None,
    model_path_override: str | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> Path:
    if cfg_override is not None:
        cfg = dict(cfg_override)
    else:
        if config_path is None:
            raise ValueError("Either config_path or cfg_override must be provided")
        cfg = load_config(resolve_config_path(config_path))

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Generate config missing required keys: {missing}")

    split = cfg.get("split", "test")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unsupported split: {split}. Allowed: {sorted(ALLOWED_SPLITS)}"
        )

    model_path = _resolve_model_path(cfg, model_path_override)
    records = select_records(cfg["processed_data_dir"], split)
    custom_tokens = cfg.get("custom_special_tokens", CUSTOM_SPECIAL_TOKENS)

    tokenizer = setup_tokenizer(
        model_path,
        chat_template_path=chat_template_path,
        custom_tokens=custom_tokens,
    )
    model = load_model(
        model_path, tokenizer, load_in_4bit=cfg.get("load_in_4bit", False)
    )

    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.get("max_new_tokens", 512),
        max_input_length=cfg.get("max_input_length", 3072),
        do_sample=cfg.get("do_sample", False),
        num_beams=cfg.get("num_beams", 1),
        seed=cfg.get("seed", 42),
        batch_size=cfg.get("batch_size", 1),
    )

    out_path = Path(str(cfg["predictions_path"]))
    _save_predictions(predictions, out_path)
    return out_path
