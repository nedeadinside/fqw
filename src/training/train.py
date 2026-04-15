from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import CUSTOM_SPECIAL_TOKENS, load_splits
from src.training.lora_config import (
    get_bnb_config,
    get_lora_config,
    get_lora_config_attention_only,
)

EXPERIMENT_CONFIGS = {
    "E1": {"lora_r": 8, "use_custom_tokens": True, "attention_only": False},
    "E2": {"lora_r": 16, "use_custom_tokens": True, "attention_only": False},
    "E3": {"lora_r": 32, "use_custom_tokens": True, "attention_only": False},
    "E4": {"lora_r": 16, "use_custom_tokens": False, "attention_only": False},
    "E5": {"lora_r": 16, "use_custom_tokens": True, "attention_only": True},
}


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_tokenizer(model_name: str, use_custom_tokens: bool, custom_tokens: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_custom_tokens:
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


def train(config_path: str, experiment_id: str = "E2"):
    cfg = load_config(config_path)
    exp_cfg = EXPERIMENT_CONFIGS[experiment_id]

    model_name = cfg["model_name"]
    use_custom_tokens = exp_cfg["use_custom_tokens"]
    lora_r = exp_cfg["lora_r"]
    attention_only = exp_cfg["attention_only"]
    custom_tokens = cfg.get("custom_special_tokens", CUSTOM_SPECIAL_TOKENS)

    output_dir = Path(cfg["output_dir"]) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(model_name, use_custom_tokens, custom_tokens)

    splits = load_splits(
        processed_data_dir=cfg["processed_data_dir"],
        tokenizer=tokenizer,
        max_seq_length=cfg["max_seq_length"],
        use_custom_tokens=use_custom_tokens,
    )
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    model = setup_model(model_name, tokenizer, use_quantization=True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    if attention_only:
        lora_config = get_lora_config_attention_only(
            r=lora_r,
            use_custom_tokens=use_custom_tokens,
            lora_dropout=cfg.get("lora_dropout", 0.05),
        )
    else:
        lora_config = get_lora_config(
            r=lora_r,
            use_custom_tokens=use_custom_tokens,
            lora_dropout=cfg.get("lora_dropout", 0.05),
        )

    model = get_peft_model(model, lora_config)

    response_template = "<|im_start|>assistant\n"
    collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    bf16_flag = cfg.get("bf16", True)
    fp16_flag = not bf16_flag

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
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
        run_name=f"{cfg.get('run_name', 'nl2sql')}-{experiment_id}",
        dataset_text_field="text",
        max_length=cfg.get("max_seq_length", 2048),
        packing=cfg.get("packing", False),
        completion_only_loss=False,
        assistant_only_loss=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    return str(best_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument(
        "--experiment",
        default="E2",
        choices=list(EXPERIMENT_CONFIGS.keys()),
    )
    args = parser.parse_args()

    best_checkpoint = train(args.config, args.experiment)
    print(best_checkpoint)
