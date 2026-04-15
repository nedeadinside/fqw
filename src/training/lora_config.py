"""
LoRA / QLoRA конфигурации для Qwen2.5-Coder-7B-Instruct.

Две функции:
    get_bnb_config()  — 4-bit NF4 квантизация (BitsAndBytesConfig)
    get_lora_config() — LoRA адаптер + опциональный modules_to_save
                        для обучения кастомных embedding
"""

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NormalFloat квантизация для QLoRA.

    NF4 — информационно-оптимальный тип данных для нормально
    распределённых весов нейросети (Dettmers et al., 2023).
    Double quantization экономит ~0.37 бит/параметр.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    use_custom_tokens: bool = True,
) -> LoraConfig:
    """LoRA конфигурация.

    Args:
        r: ранг матриц обновления (8 / 16 / 32 / 64)
        lora_alpha: масштаб (по умолчанию 2*r, как в оригинальной статье)
        lora_dropout: dropout на LoRA-адаптерах
        use_custom_tokens: если True — добавляет modules_to_save для
            embed_tokens и lm_head, чтобы обучать кастомные эмбеддинги
            (<schema>, </schema>, <question>, </question>)

    Целевые модули:
        Attention: q_proj, k_proj, v_proj, o_proj
        MLP:       gate_proj, up_proj, down_proj
        Расширенный охват MLP даёт +1-2% EX по сравнению с attention-only
        (Hu et al., 2022 + ablation E5 в данной работе).
    """
    if lora_alpha is None:
        lora_alpha = 2 * r

    # КРИТИЧЕСКИ ВАЖНО для кастомных токенов:
    # LoRA покрывает attention/MLP, но НЕ embedding layer и lm_head.
    # Без modules_to_save новые 4 строки embed_tokens инициализируются
    # случайно и не обновляются при обучении.
    # modules_to_save = обучать эти слои полностью в bf16.
    modules_to_save = ["embed_tokens", "lm_head"] if use_custom_tokens else None

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,
    )


def get_lora_config_attention_only(
    r: int = 16,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    use_custom_tokens: bool = True,
) -> LoraConfig:
    """LoRA только на attention-проекциях (ablation E5)."""
    if lora_alpha is None:
        lora_alpha = 2 * r

    modules_to_save = ["embed_tokens", "lm_head"] if use_custom_tokens else None

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,
    )
