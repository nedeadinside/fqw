import json
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
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
        """Return the path to the SQLite database file for the given db_id."""
        raise NotImplementedError

    def get_sample_values(
        self, db_id: str, table_name: str, column_name: str
    ) -> List[str]:
        """Extract sample values from a column in the database."""
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
        """Context manager for database connections."""
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _get_cached_db_path(self, db_id: str) -> Optional[Path]:
        """Get cached database path."""
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
        """Return the path to the SQLite database file for Spider."""
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


def build_all_datasets(
    data_root: str,
    output_dir: str,
    include_samples: bool = False,
    num_samples: int = 3,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    builders_config = [
        (data_root / "Spider", SpiderDatasetBuilder, ["train", "val", "test"]),
    ]

    for dataset_path, builder_class, splits in builders_config:
        if not dataset_path.exists():
            continue

        builder = builder_class(
            str(dataset_path), str(output_dir), include_samples, num_samples
        )
        for split in splits:
            try:
                builder.build_dataset(split=split)
            except (FileNotFoundError, ValueError):
                continue


def main():
    data_root = "/home/matvey/projects/fqw/raw_data"
    output_dir = "/home/matvey/projects/fqw/processed_data"

    build_all_datasets(data_root, output_dir, include_samples=True, num_samples=3)


if __name__ == "__main__":
    main()
