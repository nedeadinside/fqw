import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def get_bnb_config() -> BitsAndBytesConfig:
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
) -> LoraConfig:
    if lora_alpha is None:
        lora_alpha = 2 * r

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )
