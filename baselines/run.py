from __future__ import annotations

import json

from src.data.dataset import load_jsonl
from baselines.core import (
    LOAD_IN_4BIT,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    load_base_model,
    run_baseline,
    setup_tokenizer_base,
)


def _tag(records: list[dict], source: str = "spider") -> list[dict]:
    for r in records:
        r.setdefault("source", source)
    return records


def _summary(metrics: dict) -> dict:
    return {
        "ex_strict": metrics.get("ex_strict"),
        "ex_permuted": metrics.get("ex_permuted"),
        "em": metrics.get("em"),
        "vsr": metrics.get("vsr"),
        "n_examples": metrics.get("n_examples"),
    }


def main() -> None:
    test_records = _tag(load_jsonl(f"{PROCESSED_DATA_DIR}/test.jsonl"))
    val_records = _tag(load_jsonl(f"{PROCESSED_DATA_DIR}/val.jsonl"))

    tokenizer = setup_tokenizer_base(MODEL_NAME)
    model = load_base_model(MODEL_NAME, load_in_4bit=LOAD_IN_4BIT)

    print("=== E0_ZS ===")
    m_zs = run_baseline("zero-shot", "E0_ZS", test_records, val_records, model, tokenizer)
    print(json.dumps(_summary(m_zs), ensure_ascii=False, indent=2))

    print("=== E0_FS ===")
    m_fs = run_baseline("few-shot", "E0_FS", test_records, val_records, model, tokenizer)
    print(json.dumps(_summary(m_fs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
