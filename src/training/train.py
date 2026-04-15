"""
Обучающий скрипт: QLoRA + SFTTrainer для задачи NL-to-SQL.

Запуск:
    python -m src.training.train [--config configs/training_config.yaml] [--experiment E2]

Поддерживаемые эксперименты (из PLAN.md 4.1):
    E1  QLoRA r=8,  custom tokens
    E2  QLoRA r=16, custom tokens  (основной)
    E3  QLoRA r=32, custom tokens
    E4  QLoRA r=16, БЕЗ custom tokens (ablation)
    E5  QLoRA r=16, attention only,  custom tokens (ablation)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# Добавляем корень проекта в sys.path, чтобы импортировать src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    CUSTOM_SPECIAL_TOKENS,
    load_splits,
)
from src.training.lora_config import (
    get_bnb_config,
    get_lora_config,
    get_lora_config_attention_only,
)


# ---------------------------------------------------------------------------
# Конфигурация экспериментов
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = {
    "E1": {"lora_r": 8,  "use_custom_tokens": True,  "attention_only": False},
    "E2": {"lora_r": 16, "use_custom_tokens": True,  "attention_only": False},
    "E3": {"lora_r": 32, "use_custom_tokens": True,  "attention_only": False},
    "E4": {"lora_r": 16, "use_custom_tokens": False, "attention_only": False},
    "E5": {"lora_r": 16, "use_custom_tokens": True,  "attention_only": True},
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_tokenizer(model_name: str, use_custom_tokens: bool, custom_tokens: list[str]):
    """Загружает токенизатор и добавляет кастомные специальные токены."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"  # Обязательно для SFT

    if use_custom_tokens:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": custom_tokens}
        )
        print(f"[tokenizer] Добавлено {num_added} кастомных токенов: {custom_tokens}")

        # Проверка: каждый тег = ровно 1 токен
        for tok in custom_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            assert len(ids) == 1, (
                f"Токен '{tok}' токенизируется в {len(ids)} подтокенов "
                f"вместо 1. Проверьте add_special_tokens."
            )
        print("[tokenizer] Все кастомные токены корректно добавлены (1 токен каждый).")

    return tokenizer


def setup_model(model_name: str, tokenizer, use_quantization: bool = True):
    """Загружает модель с 4-bit квантизацией и расширяет embedding matrix."""
    bnb_config = get_bnb_config() if use_quantization else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Расширяем embedding matrix для новых кастомных токенов
    model.resize_token_embeddings(len(tokenizer))
    print(f"[model] Embedding matrix расширена до {len(tokenizer)} токенов.")

    return model


# ---------------------------------------------------------------------------
# Основная функция обучения
# ---------------------------------------------------------------------------

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

    print(f"\n{'='*60}")
    print(f"Эксперимент: {experiment_id}")
    print(f"  Модель:          {model_name}")
    print(f"  LoRA rank:       {lora_r}")
    print(f"  Custom tokens:   {use_custom_tokens}")
    print(f"  Attention only:  {attention_only}")
    print(f"  Output dir:      {output_dir}")
    print(f"{'='*60}\n")

    # 1. Токенизатор
    tokenizer = setup_tokenizer(model_name, use_custom_tokens, custom_tokens)

    # 2. Датасеты
    splits = load_splits(
        processed_data_dir=cfg["processed_data_dir"],
        tokenizer=tokenizer,
        max_seq_length=cfg["max_seq_length"],
        use_custom_tokens=use_custom_tokens,
    )
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    # 3. Модель
    model = setup_model(model_name, tokenizer, use_quantization=True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    # 4. LoRA
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
    model.print_trainable_parameters()

    # 5. DataCollator — loss только на токенах ответа (SQL)
    # "<|im_start|>assistant\n" — начало assistant-turn в Qwen ChatML
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 6. SFTConfig (наследует от TrainingArguments)
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
        max_seq_length=cfg.get("max_seq_length", 2048),
        packing=cfg.get("packing", False),
    )

    # 7. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 8. Обучение
    print("[train] Начало обучения...")
    trainer.train()

    # 9. Сохранение лучшего checkpoint + токенизатора
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"[train] Модель сохранена в {best_dir}")

    return str(best_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning для NL-to-SQL")
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Путь к YAML-конфигу",
    )
    parser.add_argument(
        "--experiment",
        default="E2",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="ID эксперимента (E1/E2/E3/E4/E5)",
    )
    args = parser.parse_args()

    best_checkpoint = train(args.config, args.experiment)
    print(f"\nДообученная модель: {best_checkpoint}")
