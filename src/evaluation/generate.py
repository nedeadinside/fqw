from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (
    SYSTEM_PROMPT,
    load_jsonl,
    stratified_dev_split,
)
from src.evaluation._config import (
    load_config,
    merge_train_config,
    resolve_config_path,
    resolve_optional_path,
)

ALLOWED_SPLITS = {"val", "test", "test_spider_held_out"}
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


def _first_stop_idx(text: str, stops: tuple[str, ...]) -> int | None:
    idx: int | None = None
    for stop in stops:
        stop_idx = text.find(stop)
        if stop_idx != -1 and (idx is None or stop_idx < idx):
            idx = stop_idx
    return idx


def extract_sql(generated_text: str) -> str:
    sql = generated_text
    if "<|im_start|>assistant" in sql:
        sql = sql.split("<|im_start|>assistant")[-1]

    stop_idx = _first_stop_idx(sql, SQL_STOP_TOKENS)
    if stop_idx is not None:
        sql = sql[:stop_idx]

    sql = sql.strip()

    if "<evidence>" in sql and "</evidence>" in sql:
        sql = sql[sql.index("</evidence>") + len("</evidence>") :]

    sql = sql.strip()
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
    custom_special_tokens: list[str] | None = None,
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

    if custom_special_tokens:
        for tok in custom_special_tokens:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    "Tokenizer in model checkpoint is incompatible with custom token "
                    f"{tok!r}. Ensure inference uses the exact tokenizer saved with training."
                )

    return tokenizer


def load_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    dtype: str = "bfloat16",
    max_lora_rank: int = 64,
):
    from vllm import LLM

    model_source = model_path
    tokenizer_source = model_path
    lora_request = None

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from vllm.lora.request import LoRARequest

        adapter_meta = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_name = adapter_meta.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config}")
        model_source = base_model_name
        lora_request = LoRARequest("sql_adapter", 1, model_path)

    llm_kwargs: dict[str, Any] = {
        "model": model_source,
        "tokenizer": tokenizer_source,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if lora_request is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank

    model = LLM(**llm_kwargs)

    return model, lora_request


def generate_predictions(
    records: list[dict],
    llm,
    tokenizer,
    lora_request,
    max_new_tokens: int = 512,
    max_input_length: int = 3072,
    do_sample: bool = False,
    num_beams: int = 1,
    seed: int = 42,
) -> list[dict]:
    from vllm import SamplingParams

    if do_sample:
        temperature = 0.7
        top_p = 0.95
    else:
        temperature = 0.0
        top_p = 1.0

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        stop=list(SQL_STOP_TOKENS),
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        use_beam_search=num_beams > 1,
        seed=seed,
    )

    prompts: list[str] = []
    for ex in records:
        prompt = make_inference_prompt(ex, tokenizer)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count > max_input_length:
            raise ValueError(
                f"Prompt exceeds max_input_length={max_input_length} for example_id={ex.get('example_id')} "
                f"(tokens={token_count}). Increase max_input_length or reduce schema context."
            )
        prompts.append(prompt)

    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    predictions = []
    for i, (ex, out) in enumerate(zip(records, outputs)):
        if out.outputs:
            predicted_response = out.outputs[0].text.strip()
        else:
            predicted_response = ""

        predictions.append(
            {
                "example_id": ex.get("example_id", f"{i}"),
                "source": ex.get("source", "unknown"),
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["sql"],
                "predicted_response": predicted_response,
                "predicted_sql": extract_sql(predicted_response),
            }
        )

    return predictions


def _tag(records: list[dict], source: str) -> list[dict]:
    for r in records:
        r.setdefault("source", source)
    return records


def select_records(processed_data_dir: str | Path, split: str) -> list[dict]:
    data_dir = Path(processed_data_dir)

    if split == "test_spider_held_out":
        return _tag(load_jsonl(data_dir / "spider_test.jsonl"), "spider")

    spider_dev = _tag(load_jsonl(data_dir / "spider_dev.jsonl"), "spider")
    spider_val, spider_test = stratified_dev_split(spider_dev)

    if split == "val":
        return spider_val
    return spider_test


def _resolve_model_path(
    cfg: dict[str, Any],
    run_id: str,
    model_path_override: str | None,
) -> str:
    if model_path_override:
        return model_path_override
    if "output_dir" not in cfg:
        raise ValueError("output_dir is required in config when model_path is not set")
    return str(Path(cfg["output_dir"]) / run_id / "best")


def _save_predictions(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def generate(
    config_path: str,
    chat_template_path: str | None = None,
    run_id: str = "E2",
    model_path_override: str | None = None,
) -> Path:
    cfg = merge_train_config(load_config(resolve_config_path(config_path)))

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

    tokenizer = setup_tokenizer(
        model_path,
        chat_template_path=chat_template_path,
        custom_special_tokens=cfg.get("custom_special_tokens"),
    )
    llm, lora_request = load_vllm_engine(
        model_path=model_path,
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=cfg.get("max_model_len", cfg.get("max_seq_length")),
        dtype=cfg.get("vllm_dtype", "bfloat16"),
        max_lora_rank=cfg.get("max_lora_rank", 64),
    )

    predictions = generate_predictions(
        records=records,
        llm=llm,
        tokenizer=tokenizer,
        lora_request=lora_request,
        max_new_tokens=cfg.get("max_new_tokens", 512),
        max_input_length=cfg.get("max_input_length", 3072),
        do_sample=cfg.get("do_sample", False),
        num_beams=cfg.get("num_beams", 1),
        seed=cfg.get("seed", 42),
    )

    results_dir = Path(cfg.get("results_dir", "./results"))
    out_path = results_dir / "predictions" / f"{run_id}_{split}_predictions.jsonl"
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
