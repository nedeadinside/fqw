import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _db_id(sql_context: str) -> str:
    return "gretel_" + hashlib.md5(sql_context.encode()).hexdigest()[:8]


def _extract_ddl(sql_context: str) -> str:
    statements = [s.strip() for s in sql_context.split(";") if s.strip()]
    return ";\n".join(s for s in statements if s.upper().lstrip().startswith("CREATE")) + ";"


def _create_database(db_path: Path, sql_context: str) -> bool:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.executescript(sql_context)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  Warning: failed to create {db_path.name}: {e}", file=sys.stderr)
        if db_path.exists():
            db_path.unlink()
        return False


def download(output_dir: str) -> None:
    out = Path(output_dir)
    (out / "databases").mkdir(parents=True, exist_ok=True)

    print("Downloading gretelai/synthetic_text_to_sql...")
    ds = load_dataset("gretelai/synthetic_text_to_sql")

    train_all = list(ds["train"])
    val_size = max(1, int(len(train_all) * 0.1))
    splits = {
        "val": train_all[:val_size],
        "train": train_all[val_size:],
        "test": list(ds["test"]),
    }

    schemas: dict = {}

    for split_name, rows in splits.items():
        out_file = out / f"{split_name}.jsonl"
        print(f"Processing {split_name} ({len(rows)} rows) → {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            for row in tqdm(rows, desc=split_name):
                ctx = row["sql_context"]
                did = _db_id(ctx)

                if did not in schemas:
                    db_path = out / "databases" / did / f"{did}.sqlite"
                    ok = _create_database(db_path, ctx)
                    if ok:
                        schemas[did] = {"ddl": _extract_ddl(ctx)}

                if did not in schemas:
                    continue

                record = {
                    "db_id": did,
                    "sql_prompt": row["sql_prompt"],
                    "sql": row["sql"],
                    "sql_context": ctx,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    schemas_path = out / "schemas.json"
    with open(schemas_path, "w", encoding="utf-8") as f:
        json.dump(schemas, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(schemas)} unique databases, schemas → {schemas_path}")


if __name__ == "__main__":
    download("/home/matvey/projects/fqw/raw_data/Gretel")
