from __future__ import annotations

from pathlib import Path

from src.data.dataset import load_jsonl
from src.evaluation.generate import (
    generate_predictions,
    load_model,
    setup_tokenizer,
)


def load_eval_samples(
    processed_data_dir: str, split: str, num_samples: int
) -> list[dict]:
    data_dir = Path(processed_data_dir)
    records = load_jsonl(data_dir / f"{split}.jsonl")
    return records[:num_samples]


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
    split: str,
    load_in_4bit: bool,
) -> None:
    tokenizer = setup_tokenizer(model_path, chat_template_path=chat_template_path)
    model = load_model(model_path, tokenizer, load_in_4bit=load_in_4bit)

    records = load_eval_samples(processed_data_dir, split, num_samples)
    predictions = generate_predictions(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
    )
    print_inference_results(predictions)


if __name__ == "__main__":
    MODEL_PATH = "./artifacts/experiments/E2/checkpoints/best"
    NUM_SAMPLES = 2
    PROCESSED_DATA_DIR = "./processed_data"
    CHAT_TEMPLATE_PATH: str | None = "templates/qwen_chat_template.jinja"
    MAX_NEW_TOKENS = 512
    MAX_INPUT_LENGTH = 4096
    SPLIT = "val"
    LOAD_IN_4BIT = False

    main(
        model_path=MODEL_PATH,
        num_samples=NUM_SAMPLES,
        processed_data_dir=PROCESSED_DATA_DIR,
        chat_template_path=CHAT_TEMPLATE_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=MAX_INPUT_LENGTH,
        split=SPLIT,
        load_in_4bit=LOAD_IN_4BIT,
    )
