from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

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
REQUIRED_CONFIG_KEYS = ("processed_data_dir",)
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


def extract_sql(generated_text: str, strip_evidence: bool = False) -> str:
    sql = _extract_assistant_text(generated_text)

    if strip_evidence:
        sql = re.sub(r"<evidence>.*?</evidence>\s*", "", sql, flags=re.DOTALL).strip()

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
    strip_evidence: bool = False,
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
        assistant_text = _extract_assistant_text(decoded)
        predictions.append(
            {
                "example_id": ex.get("example_id", f"{i}"),
                "source": ex.get("source", "unknown"),
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["sql"],
                "predicted_evidence": extract_evidence(assistant_text),
                "predicted_sql": extract_sql(decoded, strip_evidence=strip_evidence),
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
    run_id: str,
    model_path_override: str | None,
) -> str:
    if model_path_override:
        return str(resolve_optional_path(model_path_override))
    if "model_path" in cfg:
        return str(resolve_optional_path(str(cfg["model_path"])))
    if "best_model_dir" in cfg:
        return str(resolve_optional_path(str(cfg["best_model_dir"])))
    if "checkpoint_dir" in cfg:
        candidate = Path(str(cfg["checkpoint_dir"])) / "best"
        return str(resolve_optional_path(str(candidate)))
    if "output_dir" in cfg:
        root = Path(str(cfg["output_dir"]))
        candidate = root / run_id / "best" if run_id else root / "best"
        return str(resolve_optional_path(str(candidate)))
    raise ValueError(
        "Config must contain one of: model_path, best_model_dir, checkpoint_dir, output_dir"
    )


def _save_predictions(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def generate(
    config_path: str | None = None,
    chat_template_path: str | None = None,
    run_id: str = "E2",
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

    model_path = _resolve_model_path(cfg, run_id, model_path_override)
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
        strip_evidence=cfg.get("strip_evidence", False),
    )

    if "predictions_path" in cfg:
        out_path = Path(str(cfg["predictions_path"]))
    else:
        results_dir = Path(str(cfg.get("results_dir", "./results")))
        filename = (
            f"{run_id}_{split}_predictions.jsonl"
            if run_id
            else f"{split}_predictions.jsonl"
        )
        out_path = results_dir / "predictions" / filename
    _save_predictions(predictions, out_path)
    return out_path


if __name__ == "__main__":
    GENERATE_CONFIG_PATH = "configs/generate_qwen.yaml"
    GENERATE_CHAT_TEMPLATE_PATH: str | None = "templates/qwen_chat_template.jinja"
    GENERATE_RUN_ID = "E2"
    GENERATE_MODEL_PATH: str | None = None

    out = generate(
        config_path=GENERATE_CONFIG_PATH,
        chat_template_path=GENERATE_CHAT_TEMPLATE_PATH,
        run_id=GENERATE_RUN_ID,
        model_path_override=GENERATE_MODEL_PATH,
    )
    print(f"Predictions saved to: {out}")
