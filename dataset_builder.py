import io
import json
import sqlite3
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.evidence import generate_evidence


class BaseDatasetBuilder(ABC):
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        include_samples: bool = False,
        num_samples: int = 3,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables: Dict[str, Any] = {}
        self.include_samples = include_samples
        self.num_samples = num_samples
        self._schema_cache: Dict[str, str] = {}
        self._sample_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._db_path_cache: Dict[str, Optional[Path]] = {}

    @abstractmethod
    def load_tables(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_queries(self, split: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_db_path(self, db_id: str) -> Optional[Path]:
        raise NotImplementedError

    def get_sample_values(
        self, db_id: str, table_name: str, column_name: str
    ) -> List[str]:
        cache_key = (db_id, table_name, column_name)
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        db_path = self._get_cached_db_path(db_id)
        if db_path is None or not db_path.exists():
            return []

        try:
            with self._get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                safe_table = f'"{table_name}"'
                safe_column = f'"{column_name}"'
                query = f"SELECT {safe_column} FROM {safe_table} WHERE {safe_column} IS NOT NULL LIMIT {self.num_samples}"
                cursor.execute(query)
                rows = cursor.fetchall()

            samples = [
                (lambda s: s[:47] + "..." if len(s) > 50 else s)(str(row[0]))
                for row in rows
                if row[0] is not None
            ]
            self._sample_cache[cache_key] = samples
            return samples
        except Exception:
            return []

    @contextmanager
    def _get_db_connection(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _get_cached_db_path(self, db_id: str) -> Optional[Path]:
        if db_id not in self._db_path_cache:
            self._db_path_cache[db_id] = self.get_db_path(db_id)
        return self._db_path_cache[db_id]

    def get_schema_ddl(self, db_id: str) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        if db_id not in self.tables:
            return ""

        db_schema = self.tables[db_id]
        table_names_original = db_schema.get("table_names_original", [])
        column_names_original = db_schema.get("column_names_original", [])
        column_types = db_schema.get("column_types", [])
        primary_keys = db_schema.get("primary_keys", [])
        foreign_keys = db_schema.get("foreign_keys", [])

        pk_set = set()
        for pk in primary_keys:
            if isinstance(pk, list):
                pk_set.update(pk)
            else:
                pk_set.add(pk)

        fk_map = {
            from_col_idx: (
                table_names_original[column_names_original[to_col_idx][0]],
                column_names_original[to_col_idx][1],
            )
            for from_col_idx, to_col_idx in foreign_keys
        }

        table_columns: Dict[int, List[Tuple]] = defaultdict(list)
        for col_idx, (table_id, col_name) in enumerate(column_names_original):
            if table_id >= 0:
                col_type = (
                    column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                )
                is_pk = col_idx in pk_set
                fk_ref = fk_map.get(col_idx)
                table_columns[table_id].append((col_name, col_type, is_pk, fk_ref))

        ddl_statements = []
        for table_id, table_name in enumerate(table_names_original):
            columns = table_columns.get(table_id, [])
            if not columns:
                continue

            col_definitions = []
            fk_definitions = []
            sample_comments = []

            for col_name, col_type, is_pk, fk_ref in columns:
                col_def = f"{col_name} {col_type.upper()}"
                if is_pk:
                    col_def += " PRIMARY KEY"
                col_definitions.append(col_def)

                if fk_ref:
                    ref_table, ref_col = fk_ref
                    fk_definitions.append(
                        f"FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col})"
                    )

                if self.include_samples:
                    samples = self.get_sample_values(db_id, table_name, col_name)
                    if samples:
                        samples_str = ", ".join(f"{s}" for s in samples)
                        sample_comments.append(f"- {col_name}: {samples_str}")

            all_definitions = col_definitions + fk_definitions
            columns_str = ", ".join(all_definitions)
            ddl = f"CREATE TABLE {table_name} ({columns_str});"

            if sample_comments:
                ddl += "\n" + "\n".join(sample_comments)

            ddl_statements.append(ddl)

        result = "\n".join(ddl_statements)
        self._schema_cache[db_id] = result
        return result

    @abstractmethod
    def build_record(
        self, idx: int, example: Dict[str, Any], schema: str
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def build_dataset(self, split: str, output_file: Optional[str] = None) -> str:
        if not self.tables:
            self.load_tables()

        queries = self.load_queries(split)

        if output_file is None:
            output_file = self.output_dir / f"{split}.jsonl"

        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as out_f:
            for idx, example in enumerate(queries):
                db_id = example.get("db_id")
                schema = self.get_schema_ddl(db_id)
                record = self.build_record(idx, example, schema)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return str(output_path)

    def get_dataset_stats(self, split: str) -> Dict[str, Any]:
        if not self.tables:
            self.load_tables()

        queries = self.load_queries(split)

        if not queries:
            return {
                "dataset": self.get_dataset_name(),
                "split": split,
                "total_examples": 0,
                "unique_databases": 0,
                "avg_sql_length": 0,
                "avg_question_length": 0,
                "databases_distribution": {},
            }

        dbs = defaultdict(int)
        total_sql_length = 0
        total_question_length = 0

        for example in queries:
            dbs[example.get("db_id")] += 1
            total_sql_length += len(self._get_sql_field(example))
            total_question_length += len(example.get("question", ""))

        num_queries = len(queries)
        return {
            "dataset": self.get_dataset_name(),
            "split": split,
            "total_examples": num_queries,
            "unique_databases": len(dbs),
            "avg_sql_length": round(total_sql_length / num_queries, 2),
            "avg_question_length": round(total_question_length / num_queries, 2),
            "databases_distribution": dict(dbs),
        }

    @abstractmethod
    def _get_sql_field(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError


class SpiderDatasetBuilder(BaseDatasetBuilder):
    SPIDER_PATHS = [
        "database",
        "test_database",
    ]

    def get_dataset_name(self) -> str:
        return "spider"

    def get_db_path(self, db_id: str) -> Optional[Path]:
        for rel_path in self.SPIDER_PATHS:
            db_path = self.data_dir / rel_path / db_id / f"{db_id}.sqlite"
            if db_path.exists():
                return db_path
        return None

    def load_tables(self) -> None:
        tables_files = [
            self.data_dir / "tables.json",
            self.data_dir / "test_tables.json",
        ]

        for tables_file in tables_files:
            if tables_file.exists():
                with open(tables_file, "r", encoding="utf-8") as f:
                    tables_data = json.load(f)
                for db_info in tables_data:
                    if db_info["db_id"] not in self.tables:
                        self.tables[db_info["db_id"]] = db_info

    def load_queries(self, split: str = "train") -> List[Dict[str, Any]]:
        split_files = {
            "train": [
                self.data_dir / "train_spider.json",
                self.data_dir / "train_others.json",
            ],
            "val": [self.data_dir / "dev.json"],
            "test": [self.data_dir / "test.json"],
        }

        if split not in split_files:
            raise ValueError(f"Unknown split: {split}")

        queries = []
        for file in split_files[split]:
            if file.exists():
                with open(file, "r", encoding="utf-8") as f:
                    queries.extend(json.load(f))
        return queries

    def _get_sql_field(self, example: Dict[str, Any]) -> str:
        return example.get("query", "")

    def build_record(
        self, idx: int, example: Dict[str, Any], schema: str
    ) -> Dict[str, Any]:
        sql = example.get("query", "")
        question = example.get("question", "") or ""
        return {
            "example_id": idx,
            "db_id": example.get("db_id"),
            "question": question,
            "sql": sql,
            "schema": schema,
            "evidence": generate_evidence(sql, question),
            "source": "spider",
            "complexity": example.get("hardness", "unknown"),
        }


class GretelDatasetBuilder(BaseDatasetBuilder):
    def get_dataset_name(self) -> str:
        return "gretel"

    def get_db_path(self, db_id: str) -> Optional[Path]:
        path = self.data_dir / "databases" / db_id / f"{db_id}.sqlite"
        return path if path.exists() else None

    def load_tables(self) -> None:
        schemas_file = self.data_dir / "schemas.json"
        if not schemas_file.exists():
            return
        with open(schemas_file, "r", encoding="utf-8") as f:
            self.tables = json.load(f)

    def load_queries(self, split: str) -> List[Dict[str, Any]]:
        split_file = self.data_dir / f"{split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        queries = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        return queries

    def _get_sql_field(self, example: Dict[str, Any]) -> str:
        return example.get("sql", "")

    def get_schema_ddl(self, db_id: str) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        entry = self.tables.get(db_id, {})
        ddl = entry.get("ddl", "")

        if self.include_samples and ddl:
            ddl = self._enrich_ddl_with_samples(db_id, ddl)

        self._schema_cache[db_id] = ddl
        return ddl

    def _enrich_ddl_with_samples(self, db_id: str, ddl: str) -> str:
        db_path = self._get_cached_db_path(db_id)
        if db_path is None:
            return ddl

        try:
            with self._get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
        except Exception:
            return ddl

        sample_blocks: Dict[str, List[str]] = {}
        for table_name in tables:
            try:
                with self._get_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    columns = [row[1] for row in cursor.fetchall()]
                lines = []
                for col in columns:
                    samples = self.get_sample_values(db_id, table_name, col)
                    if samples:
                        lines.append(f"- {col}: {', '.join(samples)}")
                if lines:
                    sample_blocks[table_name] = lines
            except Exception:
                continue

        if not sample_blocks:
            return ddl

        enriched_statements = []
        for stmt in ddl.split(";\n"):
            stmt = stmt.strip().rstrip(";")
            if not stmt:
                continue
            upper = stmt.upper().lstrip()
            table_name = None
            if upper.startswith("CREATE TABLE"):
                for tname in sample_blocks:
                    if tname.upper() in upper:
                        table_name = tname
                        break
            result = stmt + ";"
            if table_name and table_name in sample_blocks:
                result += "\n" + "\n".join(sample_blocks[table_name])
            enriched_statements.append(result)

        return "\n".join(enriched_statements)

    def _validate_sql(self, db_id: str, sql: str) -> bool:
        db_path = self._get_cached_db_path(db_id)
        if db_path is None:
            return False
        try:
            with self._get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
            return True
        except Exception:
            return False

    def build_record(
        self, idx: int, example: Dict[str, Any], schema: str
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("GretelDatasetBuilder uses build_dataset directly")

    def build_dataset(self, split: str, output_file: Optional[str] = None) -> str:
        if not self.tables:
            self.load_tables()

        queries = self.load_queries(split)

        if output_file is None:
            output_file = self.output_dir / f"{split}.jsonl"

        output_path = Path(output_file)
        skip_invalid_sql = 0
        skip_bad_evidence = 0

        with open(output_path, "w", encoding="utf-8") as out_f:
            for idx, example in enumerate(queries):
                db_id = example.get("db_id")

                if not self._validate_sql(db_id, example.get("sql", "")):
                    skip_invalid_sql += 1
                    continue

                schema = self.get_schema_ddl(db_id)

                buf = io.StringIO()
                with redirect_stderr(buf):
                    evidence = generate_evidence(
                        example.get("sql", ""), example.get("sql_prompt", "")
                    )
                if buf.getvalue() or not evidence:
                    skip_bad_evidence += 1
                    continue

                record = {
                    "example_id": idx,
                    "db_id": db_id,
                    "question": example.get("sql_prompt", ""),
                    "sql": example.get("sql", ""),
                    "schema": schema,
                    "evidence": evidence,
                    "source": "gretel",
                    "complexity": "unknown",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total = len(queries)
        print(
            f"[gretel/{split}] total={total} | invalid_sql={skip_invalid_sql} | bad_evidence={skip_bad_evidence} | kept={total - skip_invalid_sql - skip_bad_evidence}",
            file=sys.stderr,
        )

        return str(output_path)


def build_all_datasets(
    data_root: str,
    output_dir: str,
    include_samples: bool = False,
    num_samples: int = 3,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    builders_config = [
        (
            data_root / "Spider",
            SpiderDatasetBuilder,
            {"train": "train", "val": "val", "test": "test"},
        ),
        (
            data_root / "Gretel",
            GretelDatasetBuilder,
            {"train": "train", "val": "val"},
        ),
    ]

    dataset_files: Dict[str, List[Path]] = defaultdict(list)

    for dataset_path, builder_class, splits in builders_config:
        if not dataset_path.exists():
            continue

        builder = builder_class(
            str(dataset_path), str(output_dir), include_samples, num_samples
        )
        dataset_name = builder.get_dataset_name()

        for source_split, target_split in splits.items():
            try:
                temp_file = output_dir / f"{dataset_name}_{source_split}.jsonl"
                builder.build_dataset(split=source_split, output_file=str(temp_file))
                dataset_files[target_split].append(temp_file)
            except (FileNotFoundError, ValueError):
                continue

    for split, files in dataset_files.items():
        merged_path = output_dir / f"{split}.jsonl"
        with open(merged_path, "w", encoding="utf-8") as out_f:
            for file in files:
                with open(file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
        for file in files:
            file.unlink(missing_ok=True)


def main():
    data_root = "/home/matvey/projects/fqw/raw_data"
    output_dir = "/home/matvey/projects/fqw/processed_data"

    build_all_datasets(data_root, output_dir, include_samples=True, num_samples=3)


if __name__ == "__main__":
    main()
