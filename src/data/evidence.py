from __future__ import annotations

import re
from typing import Callable

import sqlglot
import sqlglot.expressions as exp
from sqlglot import ErrorLevel


def _get_arg(node: exp.Expression, *keys: str):
    for k in keys:
        v = node.args.get(k)
        if v is not None:
            return v
    return None


def _collect_aliases(stmt: exp.Select) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    from_clause = _get_arg(stmt, "from_", "from")
    if from_clause and isinstance(from_clause.this, exp.Table):
        t = from_clause.this
        if t.alias and t.name and t.alias.lower() != t.name.lower():
            alias_map[t.alias.upper()] = t.name
    for join in stmt.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            t = join.this
            if t.alias and t.name and t.alias.lower() != t.name.lower():
                alias_map[t.alias.upper()] = t.name
    return alias_map


def _resolve(text: str, alias_map: dict[str, str]) -> str:
    for alias, real in sorted(alias_map.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(alias)}\b(?=\.)", real, text, flags=re.IGNORECASE)
    return text


def _polish(text: str) -> str:
    text = re.sub(
        r"\bNOT\s+([\w.]+)\s+IN\b", r"\1 NOT IN", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\bNOT\s+EXISTS\s*\(?\s*(S\d+)\s*\)?",
        r"\1 is empty",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bEXISTS\s*\(?\s*(S\d+)\s*\)?",
        r"\1 is non-empty",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bNOT\s+IN\s*\(?\s*(S\d+)\s*\)?",
        r"is absent from \1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bIN\s*\(?\s*(S\d+)\s*\)?",
        r"matches a value in \1",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _build_renderer(sub_sql_to_label: dict[str, str]) -> Callable[[str], str]:
    if not sub_sql_to_label:
        return lambda s: s
    pairs = sorted(sub_sql_to_label.items(), key=lambda kv: -len(kv[0]))

    def render(s: str) -> str:
        for sub_sql, label in pairs:
            s = s.replace(sub_sql, label)
        return s

    return render


def _fmt(
    node: exp.Expression, render: Callable[[str], str], alias_map: dict[str, str]
) -> str:
    raw = node.sql(dialect="sqlite") if hasattr(node, "sql") else str(node)
    return _resolve(render(raw), alias_map)


def _fmt_p(
    node: exp.Expression, render: Callable[[str], str], alias_map: dict[str, str]
) -> str:
    return _polish(_fmt(node, render, alias_map))


def _describe_select(
    stmt: exp.Select,
    alias_map: dict[str, str],
    render: Callable[[str], str],
) -> list[str]:
    steps: list[str] = []

    from_clause = _get_arg(stmt, "from_", "from")
    if from_clause is not None:
        src_node = from_clause.this
        if isinstance(src_node, exp.Table) and src_node.name:
            steps.append(f"Retrieve data from table {src_node.name}.")
        else:
            steps.append(f"Retrieve data from {_fmt(src_node, render, alias_map)}.")

    for join in stmt.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            join_name = join.this.name
        else:
            join_name = _fmt(join.this, render, alias_map)
        on_node = join.args.get("on")
        using_node = join.args.get("using")
        if on_node is not None:
            steps.append(
                f"Join {join_name} on {_fmt_p(on_node, render, alias_map)}."
            )
        elif using_node is not None:
            steps.append(
                f"Join {join_name} using {_fmt(using_node, render, alias_map)}."
            )
        else:
            steps.append(f"Cross join with {join_name}.")

    where = stmt.args.get("where")
    if where is not None:
        steps.append(
            f"Filter rows where {_fmt_p(where.this, render, alias_map)}."
        )

    group = stmt.args.get("group")
    if group is not None:
        cols = ", ".join(_fmt(e, render, alias_map) for e in group.expressions)
        steps.append(f"Group results by {cols}.")

    having = stmt.args.get("having")
    if having is not None:
        steps.append(
            f"Keep groups satisfying {_fmt_p(having.this, render, alias_map)}."
        )

    for w in stmt.find_all(exp.Window):
        fn_sql = _fmt(w.this, render, alias_map)
        parts: list[str] = []
        pb = w.args.get("partition_by")
        if pb:
            parts.append(
                "partition by "
                + ", ".join(_fmt(p, render, alias_map) for p in pb)
            )
        od = w.args.get("order")
        if od is not None and od.expressions:
            order_parts = []
            for ordered in od.expressions:
                col = _fmt(ordered.this, render, alias_map)
                direction = (
                    "descending" if ordered.args.get("desc") else "ascending"
                )
                order_parts.append(f"{col} ({direction})")
            parts.append("ordered by " + ", ".join(order_parts))
        spec = "; ".join(parts) if parts else "full frame"
        steps.append(f"Compute window function {fn_sql} ({spec}).")

    order = stmt.args.get("order")
    if order is not None:
        parts = []
        for ordered in order.expressions:
            col = _fmt(ordered.this, render, alias_map)
            direction = "descending" if ordered.args.get("desc") else "ascending"
            parts.append(f"{col} ({direction})")
        steps.append(f"Sort results by {', '.join(parts)}.")

    limit = stmt.args.get("limit")
    offset = stmt.args.get("offset") or (
        limit.args.get("offset") if limit is not None else None
    )
    if offset is not None:
        off_expr = _get_arg(offset, "expression", "this") or offset
        off_val = (
            off_expr.sql(dialect="sqlite")
            if hasattr(off_expr, "sql")
            else str(off_expr)
        )
        steps.append(f"Skip first {off_val} rows.")
    if limit is not None:
        limit_val = _get_arg(limit, "expression", "this")
        if limit_val is not None:
            n = limit_val.sql(dialect="sqlite")
            word = "row" if n.strip() == "1" else "rows"
            steps.append(f"Return at most {n} {word}.")

    if stmt.expressions:
        distinct = stmt.args.get("distinct")
        cols = ", ".join(_fmt(e, render, alias_map) for e in stmt.expressions)
        if distinct:
            steps.append(f"Output distinct values of {cols}.")
        else:
            steps.append(f"Output: {cols}.")

    return steps


def _format_literal(lit: exp.Literal) -> str:
    val = lit.this
    if lit.is_string:
        return f"'{val}'"
    return str(val)


def _collect_entities(
    root: exp.Expression,
    alias_map: dict[str, str],
    question: str,
) -> tuple[list[str], list[str], list[str]]:
    tables: list[str] = []
    seen_t: set[str] = set()
    for t in root.find_all(exp.Table):
        if t.name and t.name not in seen_t:
            seen_t.add(t.name)
            tables.append(t.name)

    columns: list[str] = []
    seen_c: set[str] = set()
    for c in root.find_all(exp.Column):
        tbl = c.table
        col = c.name
        if not col:
            continue
        if tbl:
            real = alias_map.get(tbl.upper(), tbl)
            key = f"{real}.{col}"
        else:
            key = col
        if key not in seen_c:
            seen_c.add(key)
            columns.append(key)

    q_lower = question.lower() if question else ""
    values: list[str] = []
    seen_v: set[str] = set()
    for lit in root.find_all(exp.Literal):
        raw = lit.this
        if raw is None:
            continue
        fmt = _format_literal(lit)
        if fmt in seen_v:
            continue
        seen_v.add(fmt)
        if q_lower and str(raw).lower() in q_lower:
            values.append(f"{fmt} (from question)")
        else:
            values.append(fmt)

    return tables, columns, values


def _depth(n: exp.Expression) -> int:
    d = 0
    p = n.parent
    while p is not None:
        d += 1
        p = p.parent
    return d


def _collect_nested_selects(stmt: exp.Expression) -> list[exp.Select]:
    out: list[exp.Select] = []
    for sel in stmt.find_all(exp.Select):
        if sel is stmt:
            continue
        if isinstance(sel.parent, exp.CTE):
            continue
        out.append(sel)
    out.sort(key=_depth, reverse=True)
    return out


def _replacement_source(sel: exp.Select) -> exp.Expression:
    return sel.parent if isinstance(sel.parent, exp.Subquery) else sel


def _describe_any_select(
    inner: exp.Select, render: Callable[[str], str]
) -> list[str]:
    return _describe_select(inner, _collect_aliases(inner), render)


def _describe_setop(
    stmt: exp.Expression, render: Callable[[str], str]
) -> list[str]:
    op_name = type(stmt).__name__.upper()
    steps = [f"Combine two result sets with {op_name}."]
    for label, sub in [("First part", stmt.this), ("Second part", stmt.expression)]:
        if isinstance(sub, exp.Select):
            for s in _describe_any_select(sub, render):
                steps.append(f"[{label}] {s}")
        elif isinstance(sub, (exp.Union, exp.Intersect, exp.Except)):
            for s in _describe_setop(sub, render):
                steps.append(f"[{label}] {s}")

    outer_order = stmt.args.get("order")
    if outer_order is not None:
        parts = []
        for ordered in outer_order.expressions:
            col = render(ordered.this.sql(dialect="sqlite"))
            direction = "descending" if ordered.args.get("desc") else "ascending"
            parts.append(f"{col} ({direction})")
        steps.append(f"Sort combined results by {', '.join(parts)}.")

    outer_limit = stmt.args.get("limit")
    if outer_limit is not None:
        lv = _get_arg(outer_limit, "expression", "this")
        if lv is not None:
            n = lv.sql(dialect="sqlite")
            word = "row" if n.strip() == "1" else "rows"
            steps.append(f"Return at most {n} {word}.")

    return steps


def generate_evidence(sql: str, question: str = "") -> str:
    if not sql or not sql.strip():
        return ""
    try:
        stmt = sqlglot.parse_one(sql, read="sqlite", error_level=ErrorLevel.IGNORE)
    except Exception:
        return ""
    if stmt is None:
        return ""

    nested_selects = _collect_nested_selects(stmt)
    sub_labels = {id(sel): f"S{i + 1}" for i, sel in enumerate(nested_selects)}
    sub_sql_to_label: dict[str, str] = {}
    for sel in nested_selects:
        src = _replacement_source(sel)
        sub_sql_to_label.setdefault(src.sql(dialect="sqlite"), sub_labels[id(sel)])
    render = _build_renderer(sub_sql_to_label)

    cte_blocks: list[tuple[str, list[str]]] = []
    with_clause = _get_arg(stmt, "with_", "with")
    if with_clause is not None:
        for cte in with_clause.expressions or []:
            if isinstance(cte.this, exp.Select) and cte.alias:
                cte_blocks.append((cte.alias, _describe_any_select(cte.this, render)))

    sub_blocks: list[tuple[str, list[str]]] = []
    for sel in nested_selects:
        sub_blocks.append((sub_labels[id(sel)], _describe_any_select(sel, render)))

    merged_alias: dict[str, str] = {}
    if isinstance(stmt, exp.Select):
        merged_alias.update(_collect_aliases(stmt))
    for sel in nested_selects:
        merged_alias.update(_collect_aliases(sel))
    if with_clause is not None:
        for cte in with_clause.expressions or []:
            if isinstance(cte.this, exp.Select):
                merged_alias.update(_collect_aliases(cte.this))

    tables, columns, values = _collect_entities(stmt, merged_alias, question)

    if isinstance(stmt, (exp.Union, exp.Intersect, exp.Except)):
        main_steps = _describe_setop(stmt, render)
    elif isinstance(stmt, exp.Select):
        main_steps = _describe_select(stmt, _collect_aliases(stmt), render)
    else:
        return ""

    if not main_steps:
        return ""

    parts: list[str] = []
    header_lines: list[str] = []
    if tables:
        header_lines.append(f"  Tables: {', '.join(tables)}")
    if columns:
        header_lines.append(f"  Columns: {', '.join(columns)}")
    if values:
        header_lines.append(f"  Values: {', '.join(values)}")
    if header_lines:
        parts.append("Entities:")
        parts.extend(header_lines)

    for name, steps in cte_blocks:
        parts.append(f"CTE {name}:")
        for i, s in enumerate(steps, 1):
            parts.append(f"  {i}. {s}")

    for label, steps in sub_blocks:
        parts.append(f"Compute {label}:")
        for i, s in enumerate(steps, 1):
            parts.append(f"  {i}. {s}")

    parts.append("Steps:")
    for i, s in enumerate(main_steps, 1):
        parts.append(f"{i}. {s}")

    return "\n".join(parts)
