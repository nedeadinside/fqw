from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import yaml
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    apply_project_chat_template,
    load_splits,
)
from src.training.lora_config import get_bnb_config, get_lora_config

EXPERIMENT_ID = "E2"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def setup_tokenizer(model_name: str, custom_tokens: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = apply_project_chat_template(tokenizer)
    tokenizer.padding_side = "right"
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": custom_tokens}
    )
    print(f"[tokenizer] added={num_added}")

    for token in custom_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1

    return tokenizer


def setup_model(model_name: str, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def train(config_path: str) -> str:
    config = load_config(config_path)
    model_name = config["model_name"]
    custom_tokens = config.get("custom_special_tokens", CUSTOM_SPECIAL_TOKENS)

    output_dir = Path(config["output_dir"]) / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(model_name, custom_tokens)

    splits = load_splits(
        processed_data_dir=Path(config["processed_data_dir"]),
        tokenizer=tokenizer,
        seed=config.get("seed", 42),
    )
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    model = setup_model(model_name, tokenizer)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.get("gradient_checkpointing", True),
    )

    lora_config = get_lora_config(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha"),
        lora_dropout=config.get("lora_dropout", 0.05),
        use_custom_tokens=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    bf16_flag = config.get("bf16", True)

    trainer_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        optim=config.get("optim", "paged_adamw_8bit"),
        bf16=bf16_flag,
        fp16=not bf16_flag,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 200),
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.get("greater_is_better", False),
        logging_steps=config.get("logging_steps", 10),
        report_to=config.get("report_to", "wandb"),
        run_name=f"{config.get('run_name', 'nl2sql')}-{EXPERIMENT_ID}",
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        packing=config.get("packing", False),
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"[train] saved={best_dir}")

    return str(best_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main training run")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    checkpoint_path = train(args.config)
    print(checkpoint_path)
