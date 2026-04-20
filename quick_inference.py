from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import load_jsonl, stratified_dev_split
from src.evaluation.generate import (
    generate_predictions,
    load_vllm_engine,
    setup_tokenizer,
)


def load_eval_samples(processed_data_dir: str, num_samples: int) -> list[dict]:
    data_dir = Path(processed_data_dir)
    spider_val, _ = stratified_dev_split(load_jsonl(data_dir / "spider_dev.jsonl"))
    return spider_val[:num_samples]


def print_inference_results(predictions: list[dict]) -> None:
    sep = "=" * 140
    print(f"\n{sep}\n{'INFERENCE RESULTS':^140}\n{sep}\n")
    for idx, pred in enumerate(predictions, 1):
        match = (
            "EXACT MATCH"
            if pred["predicted_sql"].strip().lower() == pred["gold_sql"].strip().lower()
            else "MISMATCH"
        )
        print(f"[Example {idx}/{len(predictions)}]")
        print(f"Database: {pred['db_id']}")
        print(f"\nQuestion:\n  {pred['question']}")
        print(f"\nGold SQL:\n  {pred['gold_sql']}")
        print(f"\nPredicted SQL:\n  {pred['predicted_sql']}")
        print(f"\nStatus: {match}\n{'-' * 140}\n")
    print(sep)


def main(
    model_path: str,
    num_samples: int,
    processed_data_dir: str,
    chat_template_path: str | None,
    max_new_tokens: int,
    max_input_length: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    vllm_dtype: str,
    max_lora_rank: int,
) -> None:
    tokenizer = setup_tokenizer(model_path, chat_template_path=chat_template_path)
    llm, lora_request = load_vllm_engine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=vllm_dtype,
        max_lora_rank=max_lora_rank,
    )

    records = load_eval_samples(processed_data_dir, num_samples)
    predictions = generate_predictions(
        records=records,
        llm=llm,
        tokenizer=tokenizer,
        lora_request=lora_request,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
    )
    print_inference_results(predictions)


if __name__ == "__main__":
    MODEL_PATH = "./results/checkpoints/E2/best"
    NUM_SAMPLES = 2
    PROCESSED_DATA_DIR = "./processed_data"
    CHAT_TEMPLATE_PATH: str | None = "templates/qwen_chat_template.jinja"
    MAX_NEW_TOKENS = 512
    MAX_INPUT_LENGTH = 4096
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.9
    MAX_MODEL_LEN: int | None = 4096
    VLLM_DTYPE = "bfloat16"
    MAX_LORA_RANK = 64

    main(
        model_path=MODEL_PATH,
        num_samples=NUM_SAMPLES,
        processed_data_dir=PROCESSED_DATA_DIR,
        chat_template_path=CHAT_TEMPLATE_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=MAX_INPUT_LENGTH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        vllm_dtype=VLLM_DTYPE,
        max_lora_rank=MAX_LORA_RANK,
    )
