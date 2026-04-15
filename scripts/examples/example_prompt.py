import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def load_examples(path: Path, limit: int = 3):
    examples = []
    with open(path, encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj):
            if index >= limit:
                break
            examples.append(json.loads(line))
    return examples


def main():
    project_root = Path(__file__).resolve().parents[2]
    template_dir = project_root / "templates"
    template_name = "qwen_chat_template.jinja"
    data_path = project_root / "processed_data" / "spider_dev.jsonl"

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_name)
    examples = load_examples(data_path)

    for index, example in enumerate(examples, 1):
        messages = [
            {
                "role": "system",
                "content": "You are a SQL expert. Convert natural language questions to SQL queries.",
            },
            {
                "role": "user",
                "content": {
                    "schema": example["schema"],
                    "question": example["question"],
                },
            },
            {"role": "assistant", "content": example["sql"]},
        ]

        formatted_prompt = template.render(
            messages=messages,
            add_generation_prompt=True,
        )
        print(f"Example {index}:\n")
        print(formatted_prompt)


if __name__ == "__main__":
    main()
