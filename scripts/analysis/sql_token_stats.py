import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


SQL_KEYWORDS_BY_STANDARD = {
    "SQL-86/87 (Core)": [
        "SELECT",
        "FROM",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "TABLE",
        "INDEX",
        "VIEW",
        "GRANT",
        "REVOKE",
        "ORDER BY",
        "GROUP BY",
        "HAVING",
        "DISTINCT",
        "ALL",
        "AS",
        "NULL",
        "IS NULL",
        "IS NOT NULL",
        "IN",
        "BETWEEN",
        "LIKE",
        "JOIN",
        "INNER JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "CROSS JOIN",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "EXISTS",
        "ANY",
        "SOME",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "ASC",
        "DESC",
    ],
    "SQL-89 (Integrity)": [
        "PRIMARY KEY",
        "FOREIGN KEY",
        "REFERENCES",
        "UNIQUE",
        "CHECK",
        "DEFAULT",
        "NOT NULL",
    ],
    "SQL-92 (Major)": [
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "CAST",
        "COALESCE",
        "NULLIF",
        "OUTER JOIN",
        "FULL JOIN",
        "FULL OUTER JOIN",
        "LEFT OUTER JOIN",
        "RIGHT OUTER JOIN",
        "NATURAL JOIN",
        "USING",
        "SUBSTRING",
        "UPPER",
        "LOWER",
        "TRIM",
        "CONCAT",
        "||",
        "SET",
    ],
    "SQL:1999 (Recursive)": [
        "WITH",
        "WITH RECURSIVE",
        "RECURSIVE",
        "SIMILAR TO",
        "REGEXP",
        "RLIKE",
        "OVER",
        "PARTITION BY",  # Оконные функции (введены здесь, расширены в SQL:2003)
    ],
    "SQL:2003 (Window/XML)": [
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "NTILE",
        "LEAD",
        "LAG",
        "FIRST_VALUE",
        "LAST_VALUE",
        "ROWS BETWEEN",
        "RANGE BETWEEN",
        "UNBOUNDED PRECEDING",
        "UNBOUNDED FOLLOWING",
        "CURRENT ROW",
        "XMLPARSE",
        "XMLSERIALIZE",
        "XMLELEMENT",
        "AUTOINCREMENT",
        "AUTO_INCREMENT",
        "IDENTITY",
        "SEQUENCE",
        "NEXTVAL",
    ],
    "SQL:2006 (XML)": [
        "XMLQUERY",
        "XMLTABLE",
        "XMLEXISTS",
    ],
    "SQL:2008 (FETCH/TRUNCATE)": [
        "TRUNCATE",
        "FETCH",
        "FETCH FIRST",
        "FETCH NEXT",
        "OFFSET",
        "INSTEAD OF",
    ],
    "SQL:2011 (Temporal)": [
        "PERIOD FOR",
        "SYSTEM_TIME",
        "FOR PORTION OF",
        "CURRENT_TIMESTAMP",
        "CURRENT_DATE",
        "CURRENT_TIME",
    ],
    "SQL:2016 (JSON)": [
        "JSON",
        "JSON_VALUE",
        "JSON_QUERY",
        "JSON_TABLE",
        "JSON_OBJECT",
        "JSON_ARRAY",
        "JSON_EXISTS",
        "JSON_EXTRACT",
        "->",
        "->>",  # Операторы JSON
        "MATCH_RECOGNIZE",
    ],
    "SQL:2023 (Graph/JSON)": [
        "GRAPH",
        "PROPERTY GRAPH",
        "MATCH",
        "PATH",
    ],
    "Common Extensions": [
        "LIMIT",
        "TOP",  # Ограничение результатов (не стандарт, но широко используется)
        "ILIKE",  # Регистронезависимый LIKE (PostgreSQL)
        "GLOB",  # SQLite pattern matching
        "HAVING",
        "VALUES",
        "ON",
        "COLLATE",
        "ESCAPE",
        "IIF",  # Условное выражение
        "IFNULL",
        "NVL",  # NULL-обработка
        "REPLACE",
        "SUBSTR",
        "LENGTH",
        "INSTR",
        "ABS",
        "ROUND",
        "CEIL",
        "FLOOR",
        "DATE",
        "TIME",
        "DATETIME",
        "STRFTIME",
        "JULIANDAY",
        "PRINTF",
        "GROUP_CONCAT",
        "STRING_AGG",
        "RANDOM",
        "RANDOMBLOB",
        "TYPEOF",
        "ZEROBLOB",
        "HEX",
        "QUOTE",
        "UNICODE",
    ],
}

QUERY_TYPES = {
    "SELECT": r"\bSELECT\b",
    "INSERT": r"\bINSERT\b",
    "UPDATE": r"\bUPDATE\b",
    "DELETE": r"\bDELETE\b",
    "CREATE": r"\bCREATE\b",
    "DROP": r"\bDROP\b",
    "ALTER": r"\bALTER\b",
}

COMPOSITE_PATTERNS = {
    "INNER JOIN": r"\bINNER\s+JOIN\b",
    "LEFT JOIN": r"\bLEFT\s+(OUTER\s+)?JOIN\b",
    "RIGHT JOIN": r"\bRIGHT\s+(OUTER\s+)?JOIN\b",
    "FULL JOIN": r"\bFULL\s+(OUTER\s+)?JOIN\b",
    "CROSS JOIN": r"\bCROSS\s+JOIN\b",
    "NATURAL JOIN": r"\bNATURAL\s+JOIN\b",
    "JOIN (simple)": r"(?<!\w)\bJOIN\b(?!\s+(INNER|LEFT|RIGHT|FULL|CROSS|NATURAL))",
    "WITH (CTE)": r"\bWITH\b(?!\s+RECURSIVE)",
    "WITH RECURSIVE": r"\bWITH\s+RECURSIVE\b",
    "Subquery (nested SELECT)": r"\(\s*SELECT\b",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "LIMIT": r"\bLIMIT\b",
    "OFFSET": r"\bOFFSET\b",
    "FETCH": r"\bFETCH\b",
    "TOP": r"\bTOP\b",
    "UNION": r"\bUNION\b(?!\s+ALL)",
    "UNION ALL": r"\bUNION\s+ALL\b",
    "INTERSECT": r"\bINTERSECT\b",
    "EXCEPT": r"\bEXCEPT\b",
    "WHERE": r"\bWHERE\b",
    "CASE WHEN": r"\bCASE\s+WHEN\b",
    "CASE": r"\bCASE\b",
    "COALESCE": r"\bCOALESCE\b",
    "NULLIF": r"\bNULLIF\b",
    "IIF": r"\bIIF\b",
    "IFNULL": r"\bIFNULL\b",
    "IN (list)": r"\bIN\s*\(",
    "NOT IN": r"\bNOT\s+IN\b",
    "BETWEEN": r"\bBETWEEN\b",
    "LIKE": r"\bLIKE\b",
    "ILIKE": r"\bILIKE\b",
    "GLOB": r"\bGLOB\b",
    "IS NULL": r"\bIS\s+NULL\b",
    "IS NOT NULL": r"\bIS\s+NOT\s+NULL\b",
    "EXISTS": r"\bEXISTS\b",
    "NOT EXISTS": r"\bNOT\s+EXISTS\b",
    "DISTINCT": r"\bDISTINCT\b",
    "COUNT(*)": r"\bCOUNT\s*\(\s*\*\s*\)",
    "COUNT": r"\bCOUNT\s*\(",
    "SUM": r"\bSUM\s*\(",
    "AVG": r"\bAVG\s*\(",
    "MIN": r"\bMIN\s*\(",
    "MAX": r"\bMAX\s*\(",
    "GROUP_CONCAT": r"\bGROUP_CONCAT\s*\(",
    "OVER (window)": r"\bOVER\s*\(",
    "PARTITION BY": r"\bPARTITION\s+BY\b",
    "ROW_NUMBER": r"\bROW_NUMBER\s*\(",
    "RANK": r"\bRANK\s*\(",
    "DENSE_RANK": r"\bDENSE_RANK\s*\(",
    "LEAD": r"\bLEAD\s*\(",
    "LAG": r"\bLAG\s*\(",
    "CAST": r"\bCAST\s*\(",
    "CONVERT": r"\bCONVERT\s*\(",
    "CONCAT": r"\bCONCAT\s*\(",
    "|| (concat)": r"\|\|",
    "SUBSTRING/SUBSTR": r"\b(SUBSTRING|SUBSTR)\s*\(",
    "UPPER": r"\bUPPER\s*\(",
    "LOWER": r"\bLOWER\s*\(",
    "TRIM": r"\bTRIM\s*\(",
    "REPLACE": r"\bREPLACE\s*\(",
    "LENGTH": r"\bLENGTH\s*\(",
    "INSTR": r"\bINSTR\s*\(",
    "ABS": r"\bABS\s*\(",
    "ROUND": r"\bROUND\s*\(",
    "DATE": r"\bDATE\s*\(",
    "TIME": r"\bTIME\s*\(",
    "DATETIME": r"\bDATETIME\s*\(",
    "STRFTIME": r"\bSTRFTIME\s*\(",
    "JULIANDAY": r"\bJULIANDAY\s*\(",
    "CURRENT_DATE": r"\bCURRENT_DATE\b",
    "CURRENT_TIME": r"\bCURRENT_TIME\b",
    "CURRENT_TIMESTAMP": r"\bCURRENT_TIMESTAMP\b",
    "ASC": r"\bASC\b",
    "DESC": r"\bDESC\b",
    "AS (alias)": r"\bAS\b",
    "Division (/)": r"(?<![<>])/(?![/*])",
    "Multiplication (*)": r"(?<!\()\*(?!\))",
    "Addition (+)": r"(?<![|])\+",
    "Subtraction (-)": r"(?<!-)--?(?!-)",
}


def strip_sql_literals_and_comments(sql: str) -> str:
    

    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)

    sql = re.sub(r"'(?:''|[^'])*'", " ", sql)
    sql = re.sub(r'"(?:""|[^"])*"', " ", sql)
    sql = re.sub(r"`(?:``|[^`])*`", " ", sql)
    sql = re.sub(r"\[(?:\]\]|[^\]])*\]", " ", sql)

    return sql


def load_jsonl(file_path: Path) -> List[dict]:
    
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_sql(record: dict) -> str:
    
    return record.get("sql", "")


def count_pattern(sql: str, pattern: str) -> int:
    
    return len(re.findall(pattern, sql, re.IGNORECASE))


def analyze_sql(sql: str) -> Dict[str, int]:
    
    stats = defaultdict(int)

    sql = strip_sql_literals_and_comments(sql)

    for token_name, pattern in COMPOSITE_PATTERNS.items():
        count = count_pattern(sql, pattern)
        if count > 0:
            stats[token_name] = count

    return dict(stats)


def get_query_type(sql: str) -> str:
    
    sql = strip_sql_literals_and_comments(sql)
    for qtype, pattern in QUERY_TYPES.items():
        if re.search(pattern, sql, re.IGNORECASE):
            return qtype
    return "UNKNOWN"


def analyze_dataset(file_path: Path) -> Tuple[Dict[str, int], Dict[str, int], int]:
    
    records = load_jsonl(file_path)

    total_stats = defaultdict(int)
    query_type_stats = defaultdict(int)

    for record in records:
        sql = extract_sql(record)
        if not sql:
            continue

        qtype = get_query_type(sql)
        query_type_stats[qtype] += 1

        token_stats = analyze_sql(sql)
        for token, count in token_stats.items():
            total_stats[token] += count

    return dict(total_stats), dict(query_type_stats), len(records)


def get_sql_standard_for_token(token: str) -> str:
    
    token_upper = token.upper()

    token_mapping = {
        "INNER JOIN": "INNER JOIN",
        "LEFT JOIN": "LEFT JOIN",
        "RIGHT JOIN": "RIGHT JOIN",
        "FULL JOIN": "FULL JOIN",
        "CROSS JOIN": "CROSS JOIN",
        "NATURAL JOIN": "NATURAL JOIN",
        "JOIN (simple)": "JOIN",
        "WITH (CTE)": "WITH",
        "WITH RECURSIVE": "WITH RECURSIVE",
        "Subquery (nested SELECT)": "SELECT",
        "GROUP BY": "GROUP BY",
        "ORDER BY": "ORDER BY",
        "UNION ALL": "UNION",
        "NOT IN": "IN",
        "IS NULL": "IS NULL",
        "IS NOT NULL": "IS NOT NULL",
        "NOT EXISTS": "EXISTS",
        "IN (list)": "IN",
        "COUNT(*)": "COUNT",
        "CASE WHEN": "CASE",
        "OVER (window)": "OVER",
        "PARTITION BY": "PARTITION BY",
        "|| (concat)": "||",
        "SUBSTRING/SUBSTR": "SUBSTRING",
        "AS (alias)": "AS",
        "Division (/)": None,
        "Multiplication (*)": None,
        "Addition (+)": None,
        "Subtraction (-)": None,
    }

    mapped_token = token_mapping.get(token, token_upper)
    if mapped_token is None:
        return "Operators"

    for standard, keywords in SQL_KEYWORDS_BY_STANDARD.items():
        for kw in keywords:
            if kw.upper() == mapped_token.upper():
                return standard

    return "Unknown/Extension"


def print_report(
    dataset_name: str,
    token_stats: Dict[str, int],
    query_type_stats: Dict[str, int],
    total_queries: int,
):
    
    print(f"\n{'='*80}")
    print(f" Датасет: {dataset_name}")
    print(f"{'='*80}")
    print(f"\nВсего запросов: {total_queries}")

    print(f"\n--- Типы запросов ---")
    for qtype, count in sorted(query_type_stats.items(), key=lambda x: -x[1]):
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  {qtype:15s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n--- Статистика SQL токенов (топ-50) ---")
    sorted_tokens = sorted(token_stats.items(), key=lambda x: -x[1])

    print(f"{'Токен':<30s} {'Кол-во':>8s} {'%':>8s} {'Стандарт SQL':<25s}")
    print("-" * 75)

    total_tokens = sum(token_stats.values())
    for token, count in sorted_tokens[:50]:
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        standard = get_sql_standard_for_token(token)
        print(f"  {token:<28s} {count:8d} {pct:7.1f}% {standard:<25s}")

    print(f"\n--- Покрытие стандартов SQL ---")
    standard_stats = defaultdict(lambda: {"count": 0, "tokens": set()})

    for token, count in token_stats.items():
        standard = get_sql_standard_for_token(token)
        standard_stats[standard]["count"] += count
        standard_stats[standard]["tokens"].add(token)

    print(f"{'Стандарт':<30s} {'Токенов':>10s} {'Вхождений':>12s}")
    print("-" * 55)

    for standard in SQL_KEYWORDS_BY_STANDARD.keys():
        stats = standard_stats.get(standard, {"count": 0, "tokens": set()})
        print(f"  {standard:<28s} {len(stats['tokens']):10d} {stats['count']:12d}")

    for standard in ["Unknown/Extension", "Operators"]:
        if standard in standard_stats:
            stats = standard_stats[standard]
            print(f"  {standard:<28s} {len(stats['tokens']):10d} {stats['count']:12d}")


def print_comparison(
    all_datasets: Dict[str, Tuple[Dict[str, int], Dict[str, int], int]],
):
    
    print(f"\n{'='*100}")
    print(" СРАВНИТЕЛЬНАЯ ТАБЛИЦА ДАТАСЕТОВ")
    print(f"{'='*100}")

    all_tokens = set()
    for _, (token_stats, _, _) in all_datasets.items():
        all_tokens.update(token_stats.keys())

    datasets = list(all_datasets.keys())
    header = f"{'Токен':<30s}"
    for ds in datasets:
        header += f" {ds[:12]:>12s}"
    print(header)
    print("-" * (30 + 13 * len(datasets)))

    token_sums = {}
    for token in all_tokens:
        total = sum(all_datasets[ds][0].get(token, 0) for ds in datasets)
        token_sums[token] = total

    sorted_tokens = sorted(token_sums.items(), key=lambda x: -x[1])

    for token, _ in sorted_tokens[:40]:
        row = f"  {token:<28s}"
        for ds in datasets:
            count = all_datasets[ds][0].get(token, 0)
            row += f" {count:12d}"
        print(row)

    print(f"\n--- Общие метрики ---")
    row = f"  {'Всего запросов':<28s}"
    for ds in datasets:
        row += f" {all_datasets[ds][2]:12d}"
    print(row)


SQL_STANDARD_CATEGORIES = {
    "Базовые конструкции\n(SQL-86/89)": [
        "WHERE",
        "DISTINCT",
        "ORDER BY",
        "ASC",
        "DESC",
        "LIKE",
        "GLOB",
        "IN (list)",
        "NOT IN",
        "BETWEEN",
        "IS NULL",
        "IS NOT NULL",
        "LIMIT",
        "OFFSET",
        "COUNT",
        "COUNT(*)",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
    ],
    "Реляционные операции\n(SQL-92)": [
        "INNER JOIN",
        "LEFT JOIN",
        "CROSS JOIN",
        "NATURAL JOIN",
        "JOIN (simple)",
        "GROUP BY",
        "HAVING",
        "Subquery (nested SELECT)",
        "UNION",
        "UNION ALL",
        "INTERSECT",
        "EXCEPT",
        "EXISTS",
        "NOT EXISTS",
        "CASE WHEN",
        "CASE",
        "COALESCE",
        "NULLIF",
        "CAST",
        "|| (concat)",
    ],
    "Аналитические расширения\n(SQL:1999-2003)": [
        "WITH (CTE)",
        "WITH RECURSIVE",
        "OVER (window)",
        "PARTITION BY",
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "LEAD",
        "LAG",
    ],
}


def check_sql_has_category_token(sql: str, category_tokens: List[str]) -> bool:
    
    sql = strip_sql_literals_and_comments(sql)
    for token in category_tokens:
        pattern = COMPOSITE_PATTERNS.get(token)
        if pattern and re.search(pattern, sql, re.IGNORECASE):
            return True
    return False


def analyze_dataset_for_coverage(
    file_paths: List[Path],
) -> Tuple[Dict[str, float], int]:
    
    category_counts = {cat: 0 for cat in SQL_STANDARD_CATEGORIES}
    total_queries = 0

    for file_path in file_paths:
        records = load_jsonl(file_path)
        for record in records:
            sql = extract_sql(record)
            if not sql:
                continue
            total_queries += 1
            for category, tokens in SQL_STANDARD_CATEGORIES.items():
                if check_sql_has_category_token(sql, tokens):
                    category_counts[category] += 1

    coverage = {}
    for category in SQL_STANDARD_CATEGORIES:
        coverage[category] = (
            (category_counts[category] / total_queries * 100)
            if total_queries > 0
            else 0
        )

    return coverage, total_queries


def generate_coverage_chart(
    data_dir: Path,
    output_path: str = "sql_standards_coverage.png",
):
    
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    bird_files = list(data_dir.glob("bird_*.jsonl"))
    spider_files = list(data_dir.glob("spider_*.jsonl"))

    bird_coverage, bird_total = analyze_dataset_for_coverage(bird_files)
    spider_coverage, spider_total = analyze_dataset_for_coverage(spider_files)

    print(f"\nBIRD: {bird_total} запросов (объединено {len(bird_files)} файлов)")
    print(f"Spider: {spider_total} запросов (объединено {len(spider_files)} файлов)")

    coverage_data = {
        "BIRD": bird_coverage,
        "Spider": spider_coverage,
    }

    categories = list(SQL_STANDARD_CATEGORIES.keys())
    datasets = list(coverage_data.keys())

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(categories))
    width = 0.35  # Ширина столбца

    colors = ["#4C72B0", "#55A868"]  # Синий и зелёный

    dataset_totals = {
        "BIRD": bird_total,
        "Spider": spider_total,
    }

    for i, (ds_name, coverage) in enumerate(coverage_data.items()):
        offset = width * i - width / 2
        values = [coverage[cat] for cat in categories]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"{ds_name} (n={dataset_totals[ds_name]})",
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel("Доля запросов (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Покрытие групп SQL-конструкций в наборах данных (SQLite)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)  # Ограничиваем до 100% с небольшим запасом для подписей

    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.xaxis.grid(False)

    fig.text(
        0.01,
        0.01,
        "Метрика: % запросов, содержащих ≥1 конструкцию из группы. Анализ: regex по очищенному SQL (без строк/комментариев).",
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = str(output_path)
    base = output_path
    if base.lower().endswith(".png"):
        base = base[:-4]
    elif "." in Path(base).name:
        base = str(Path(base).with_suffix(""))

    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"

    plt.savefig(
        png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nГрафики сохранены: {png_path}, {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Анализ SQL токенов в JSONL датасетах")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "processed_data"),
        help="Путь к директории с JSONL файлами",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь для сохранения результатов в JSON",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Ошибка: директория {data_dir} не существует")
        return 1

    jsonl_files = list(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Ошибка: JSONL файлы не найдены в {data_dir}")
        return 1

    print(f"Найдено {len(jsonl_files)} JSONL файлов")

    all_datasets = {}
    all_results = {}

    for file_path in sorted(jsonl_files):
        dataset_name = file_path.stem
        print(f"\nАнализирую: {dataset_name}...")

        token_stats, query_type_stats, total_queries = analyze_dataset(file_path)
        all_datasets[dataset_name] = (token_stats, query_type_stats, total_queries)

        all_results[dataset_name] = {
            "total_queries": total_queries,
            "query_types": query_type_stats,
            "token_stats": token_stats,
        }

        print_report(dataset_name, token_stats, query_type_stats, total_queries)

    print_comparison(all_datasets)

    chart_path = data_dir / "sqlite_sql_feature_coverage.png"
    generate_coverage_chart(data_dir, str(chart_path))

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены в: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
