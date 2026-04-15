import json
import os

from jinja2 import Environment, FileSystemLoader

template_dir = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("chat_template.jinja")

examples = []
spider_train_path = os.path.join(template_dir, "processed_data", "spider_dev.jsonl")
with open(spider_train_path, "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        examples.append(json.loads(line))

for i, example in enumerate(examples, 1):
    messages = [
        {
            "role": "system",
            "content": "You are a SQL expert. Convert natural language questions to SQL queries.",
        },
        {
            "role": "user",
            "content": {"schema": example["schema"], "question": example["question"]},
        },
        {"role": "assistant", "content": example["sql"]},
    ]

    formatted_prompt = template.render(
        bos_token="<|begin_of_text|>", messages=messages, add_generation_prompt=True
    )
    print(f"Example {i}:\n")
    print(formatted_prompt)
