import io
import json
import os
import sqlite3
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.data.evidence import generate_evidence


def _iter_json_array(
    path: Path, chunk_size: int = 1 << 20
) -> Generator[Dict[str, Any], None, None]:
    """Stream objects from a JSON array without loading the whole file in memory."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        idx = 0
        started = False
        eof = False

        while True:
            if idx >= len(buffer) and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1

            if not started:
                if idx >= len(buffer):
                    if eof:
                        break
                    continue
                if buffer[idx] != "[":
                    raise ValueError(f"Expected JSON array in {path}")
                started = True
                idx += 1
                continue

            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1

            if idx >= len(buffer):
                if eof:
                    break
                continue

            if buffer[idx] == "]":
                break

            if buffer[idx] == ",":
                idx += 1
                continue

            try:
                value, next_idx = decoder.raw_decode(buffer, idx)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            if isinstance(value, dict):
                yield value

            idx = next_idx
            if idx > (1 << 20):
                buffer = buffer[idx:]
                idx = 0


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
        self._sample_cache: Dict[Tuple[str, str, str], List[Any]] = {}
        self._column_types_cache: Dict[str, Dict[Tuple[str, str], str]] = {}
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
    ) -> List[Any]:
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

            samples: List[Any] = []
            for row in rows:
                v = row[0]
                if v is None:
                    continue
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8", errors="replace")
                    except Exception:
                        v = str(v)
                if isinstance(v, str) and len(v) > 50:
                    v = v[:47] + "..."
                samples.append(v)
            self._sample_cache[cache_key] = samples
            return samples
        except Exception:
            return []

    def _get_column_sql_types(self, db_id: str) -> Dict[Tuple[str, str], str]:
        if db_id in self._column_types_cache:
            return self._column_types_cache[db_id]

        result: Dict[Tuple[str, str], str] = {}
        db_path = self._get_cached_db_path(db_id)
        if db_path is None or not db_path.exists():
            self._column_types_cache[db_id] = result
            return result

        try:
            with self._get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                for table_name in tables:
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    for row in cursor.fetchall():
                        col_name = row[1]
                        col_type = (row[2] or "").strip() or "TEXT"
                        result[(table_name, col_name)] = col_type
        except Exception:
            pass

        self._column_types_cache[db_id] = result
        return result

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

        real_types = self._get_column_sql_types(db_id)

        ddl_statements = []
        for table_id, table_name in enumerate(table_names_original):
            columns = table_columns.get(table_id, [])
            if not columns:
                continue

            col_lines: List[str] = []
            col_comments: List[str] = []
            pk_cols: List[str] = []
            fk_clauses: List[str] = []

            for col_name, spider_type, is_pk, fk_ref in columns:
                sql_type = real_types.get(
                    (table_name, col_name), (spider_type or "TEXT").upper()
                )
                col_lines.append(f"    `{col_name}` {sql_type}")
                if self.include_samples:
                    samples = self.get_sample_values(db_id, table_name, col_name)
                    col_comments.append(f" -- example: {samples!r}" if samples else "")
                else:
                    col_comments.append("")

                if is_pk:
                    pk_cols.append(f"`{col_name}`")
                if fk_ref:
                    ref_table, ref_col = fk_ref
                    fk_clauses.append(
                        f"    FOREIGN KEY (`{col_name}`) REFERENCES `{ref_table}` (`{ref_col}`)"
                    )

            trailing: List[str] = []
            if pk_cols:
                trailing.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
            trailing.extend(fk_clauses)

            n_cols = len(col_lines)
            total = n_cols + len(trailing)
            rendered: List[str] = []
            for i, (line, comment) in enumerate(zip(col_lines, col_comments)):
                is_last = i == total - 1
                sep = "" if is_last else ","
                rendered.append(f"{line}{sep}{comment}")
            for j, line in enumerate(trailing):
                is_last = n_cols + j == total - 1
                sep = "" if is_last else ","
                rendered.append(f"{line}{sep}")

            body = "\n".join(rendered)
            ddl_statements.append(f"CREATE TABLE {table_name} (\n{body}\n);")

        result = "\n\n".join(ddl_statements)
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


class SynSQLDatasetBuilder(BaseDatasetBuilder):
    _COMPLEXITY_LIMIT_MAP = {
        "Simple": 10_000,
        "Moderate": 10_000,
        "Complex": 10_000,
        "Highly Complex": 10_000,
    }

    def get_dataset_name(self) -> str:
        return "synsql"

    def load_tables(self) -> None:
        self.tables = {}

    def load_queries(self, split: str) -> List[Dict[str, Any]]:
        return []

    def _get_sql_field(self, example: Dict[str, Any]) -> str:
        return example.get("sql", "")

    def get_db_path(self, db_id: str) -> Optional[Path]:
        candidates = [
            self.data_dir / "databases" / db_id / f"{db_id}.sqlite",
            self.data_dir / "databases" / f"{db_id}.sqlite",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def get_schema_ddl(self, db_id: str) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        db_path = self._get_cached_db_path(db_id)
        if db_path is None or not db_path.exists():
            self._schema_cache[db_id] = ""
            return ""

        try:
            with self._get_db_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                ddl_statements: List[str] = []
                for table_name in tables:
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    table_info = cursor.fetchall()
                    if not table_info:
                        continue

                    cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
                    fk_rows = cursor.fetchall()

                    col_lines: List[str] = []
                    col_comments: List[str] = []
                    pk_cols: List[str] = []
                    for row in table_info:
                        col_name = row[1]
                        col_type = (row[2] or "").strip() or "TEXT"
                        col_lines.append(f"    `{col_name}` {col_type}")

                        if self.include_samples:
                            samples = self.get_sample_values(
                                db_id, table_name, col_name
                            )
                            col_comments.append(
                                f" -- example: {samples!r}" if samples else ""
                            )
                        else:
                            col_comments.append("")

                        if row[5]:
                            pk_cols.append(f"`{col_name}`")

                    fk_clauses: List[str] = [
                        f"    FOREIGN KEY (`{fk[3]}`) REFERENCES `{fk[2]}` (`{fk[4]}`)"
                        for fk in fk_rows
                    ]

                    trailing: List[str] = []
                    if pk_cols:
                        trailing.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
                    trailing.extend(fk_clauses)

                    total = len(col_lines) + len(trailing)
                    rendered: List[str] = []
                    for i, (line, comment) in enumerate(zip(col_lines, col_comments)):
                        sep = "" if i == total - 1 else ","
                        rendered.append(f"{line}{sep}{comment}")
                    for j, line in enumerate(trailing):
                        sep = "" if len(col_lines) + j == total - 1 else ","
                        rendered.append(f"{line}{sep}")

                    body = "\n".join(rendered)
                    ddl_statements.append(f"CREATE TABLE {table_name} (\n{body}\n);")

                schema = "\n\n".join(ddl_statements)
                self._schema_cache[db_id] = schema
                return schema
        except Exception:
            self._schema_cache[db_id] = ""
            return ""

    @staticmethod
    def _merge_question(example: Dict[str, Any]) -> str:
        question = (example.get("question") or "").strip()
        external_knowledge = (example.get("external_knowledge") or "").strip()
        if question and external_knowledge:
            return f"{question}\n{external_knowledge}"
        return question or external_knowledge

    def build_record(
        self, idx: int, example: Dict[str, Any], schema: str
    ) -> Dict[str, Any]:
        return {
            "example_id": idx,
            "db_id": example.get("db_id"),
            "question": self._merge_question(example),
            "sql": example.get("sql", ""),
            "schema": schema,
            "evidence": example.get("evidence", ""),
            "source": "synsql",
            "complexity": example.get("sql_complexity", "unknown"),
        }

    def _complexity_limit_reached(
        self, counts: Dict[str, int], complexity: str
    ) -> bool:
        limit = self._COMPLEXITY_LIMIT_MAP.get(complexity)
        if limit is None:
            return False
        return counts.get(complexity, 0) >= limit

    def _all_limits_reached(self, counts: Dict[str, int]) -> bool:
        return all(
            counts.get(complexity, 0) >= limit
            for complexity, limit in self._COMPLEXITY_LIMIT_MAP.items()
        )

    def build_dataset(self, split: str, output_file: Optional[str] = None) -> str:
        if split != "train":
            raise ValueError("SynSQL supports train-only ingestion in this pipeline")

        data_file = self.data_dir / "data.json"
        if not data_file.exists():
            raise FileNotFoundError(f"SynSQL data file not found: {data_file}")

        if output_file is None:
            output_file = self.output_dir / f"{split}.jsonl"
        output_path = Path(output_file)

        max_rows_env = os.getenv("SYNSQL_MAX_ROWS", "").strip()
        max_rows = int(max_rows_env) if max_rows_env.isdigit() else None

        complexity_counts: Dict[str, int] = {}
        skipped_missing_fields = 0
        skipped_bad_evidence = 0
        skipped_complexity_limit = 0
        total = 0

        with open(output_path, "w", encoding="utf-8") as out_f:
            for idx, example in enumerate(_iter_json_array(data_file)):
                if max_rows is not None and total >= max_rows:
                    break
                if self._all_limits_reached(complexity_counts):
                    break

                total += 1

                db_id = example.get("db_id")
                sql = example.get("sql")
                if not db_id or not sql:
                    skipped_missing_fields += 1
                    continue

                complexity = example.get("sql_complexity", "unknown")
                if self._complexity_limit_reached(complexity_counts, complexity):
                    skipped_complexity_limit += 1
                    continue

                merged_question = self._merge_question(example)

                buf = io.StringIO()
                with redirect_stderr(buf):
                    try:
                        evidence = generate_evidence(sql, merged_question)
                    except Exception:
                        evidence = ""
                if buf.getvalue() or not evidence:
                    skipped_bad_evidence += 1
                    continue

                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

                schema = self.get_schema_ddl(db_id)
                tmp = {
                    **example,
                    "question": merged_question,
                    "external_knowledge": "",
                    "evidence": evidence,
                }
                record = self.build_record(idx, tmp, schema)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        kept = (
            total
            - skipped_missing_fields
            - skipped_bad_evidence
            - skipped_complexity_limit
        )
        print(
            f"[synsql/train] total={total} | skipped_missing_fields={skipped_missing_fields} | "
            f"bad_evidence={skipped_bad_evidence} | skipped_complexity_limit={skipped_complexity_limit} | "
            f"kept={kept} | complexity_counts={complexity_counts}",
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
        (data_root / "Spider", SpiderDatasetBuilder, ["train", "val", "test"]),
        (data_root / "Gretel", GretelDatasetBuilder, ["train", "val", "test"]),
        (data_root / "SynSQL", SynSQLDatasetBuilder, ["train"]),
    ]

    dataset_files: Dict[str, List[Path]] = defaultdict(list)

    for dataset_path, builder_class, splits in builders_config:
        if not dataset_path.exists():
            continue

        builder = builder_class(
            str(dataset_path), str(output_dir), include_samples, num_samples
        )
        dataset_name = builder.get_dataset_name()

        for split in splits:
            try:
                temp_file = output_dir / f"{dataset_name}_{split}.jsonl"
                builder.build_dataset(split=split, output_file=str(temp_file))
                dataset_files[split].append(temp_file)
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
