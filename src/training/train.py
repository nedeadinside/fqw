from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.data.dataset import CUSTOM_SPECIAL_TOKENS, load_splits
from src.training.lora_config import (
    get_bnb_config,
    get_lora_config,
)


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_optional_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_candidate = Path(__file__).resolve().parents[2] / path
    if project_candidate.exists():
        return project_candidate

    return candidate


def _resolve_config_path(
    config_path: str,
) -> str:
    resolved = _resolve_optional_path(config_path)
    if resolved.exists():
        return str(resolved)

    raise FileNotFoundError(f"Train config not found: {config_path}")


def _build_output_dir(cfg: dict[str, Any]) -> Path:
    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("Train config requires 'checkpoint_dir'")
    return Path(str(checkpoint_dir))


def _validate_qwen_template_tokens(template_text: str, template_path: Path) -> None:
    required_snippets = [
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "add_generation_prompt",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in template_text]
    if missing:
        raise ValueError(
            "Chat template is missing required Qwen system markers "
            f"{missing} in {template_path}"
        )


def setup_tokenizer(
    model_name: str,
    custom_tokens: list[str],
    chat_template_path: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"

    if chat_template_path:
        template_file = _resolve_optional_path(chat_template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")

        template_text = template_file.read_text(encoding="utf-8")
        _validate_qwen_template_tokens(template_text, template_file)
        tokenizer.chat_template = template_text

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})
    for tok in custom_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1

    return tokenizer


def setup_model(model_name: str, tokenizer, use_quantization: bool = True):
    bnb_config = get_bnb_config() if use_quantization else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.resize_token_embeddings(len(tokenizer))
    return model


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer, response_template: str):
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def _find_last_subsequence(self, sequence, subsequence):
        last_pos = None
        limit = len(sequence) - len(subsequence) + 1
        for i in range(limit):
            if sequence[i : i + len(subsequence)] == subsequence:
                last_pos = i
        return last_pos

    def __call__(self, features):
        features = [
            {k: v for k, v in feature.items() if k != "labels"} for feature in features
        ]

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
        labels = input_ids.clone()

        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            start = self._find_last_subsequence(seq, self.response_token_ids)

            if start is None:
                labels[i, :] = -100
                continue

            end = start + len(self.response_token_ids)
            labels[i, :end] = -100
            labels[i, attention_mask[i] == 0] = -100

        batch["labels"] = labels
        return batch


class JsonlTrainLogCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
        }
        payload.update(logs)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return control


def train(
    config_path: str | None = None,
    chat_template_path: str | None = None,
    model_name_override: str | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> str:
    if cfg_override is not None:
        cfg = dict(cfg_override)
    else:
        if config_path is None:
            raise ValueError("Either config_path or cfg_override must be provided")
        cfg = load_config(_resolve_config_path(config_path))

    model_name = model_name_override or cfg["model_name"]
    lora_r = cfg.get("lora_r", 16)
    custom_tokens = cfg.get("custom_special_tokens", CUSTOM_SPECIAL_TOKENS)

    output_dir = _build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(
        model_name,
        custom_tokens,
        chat_template_path=chat_template_path,
    )

    splits = load_splits(
        processed_data_dir=cfg["processed_data_dir"],
        tokenizer=tokenizer,
    )
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    model = setup_model(
        model_name,
        tokenizer,
        use_quantization=cfg.get("load_in_4bit", True),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    lora_config = get_lora_config(
        r=lora_r,
        lora_dropout=cfg.get("lora_dropout", 0.05),
    )

    model = get_peft_model(model, lora_config)

    response_template = "<|im_start|>assistant\n"
    collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    logging_dir_raw = cfg.get("logging_dir")
    logging_dir = None
    if logging_dir_raw:
        logging_dir = _resolve_optional_path(str(logging_dir_raw))
        logging_dir.mkdir(parents=True, exist_ok=True)

    bf16_flag = cfg.get("bf16", True)
    fp16_flag = not bf16_flag

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        bf16=bf16_flag,
        fp16=fp16_flag,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=cfg.get("use_liger_kernel", True),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 200),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        logging_steps=cfg.get("logging_steps", 10),
        report_to=cfg.get("report_to", "wandb"),
        run_name=cfg.get("run_name", "nl2sql-main"),
        dataset_text_field="text",
        max_length=cfg.get("max_seq_length", 2048),
        **({"logging_dir": str(logging_dir)} if logging_dir is not None else {}),
    )

    callbacks = []
    if logging_dir is not None:
        callbacks.append(JsonlTrainLogCallback(logging_dir / "train_logs.jsonl"))

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    return str(best_dir)
