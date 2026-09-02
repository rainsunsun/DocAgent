"""数据分析数据引擎：DuckDB 只读查询 + SQL 安全校验。

只暴露两类能力给 Agent：
- list_tables()：列出表结构与行数，让 LLM 先看 schema 再写 SQL，减少幻觉；
- query(sql)：执行只读 SELECT，返回格式化表格（限制行数）。

安全边界（面试可讲的点）：只允许 SELECT/WITH、单语句、拒绝危险关键词、
表名白名单、结果行数上限，杜绝 DROP/删库/注入。
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

from ..config import settings

_conn: duckdb.DuckDBPyConnection | None = None

# 已注册表白名单（第一版只有 sales；加新表时在此登记）
_ALLOWED_TABLES = {"sales"}

# 危险关键词（DuckDB 语法，整词匹配）
_FORBIDDEN = (
    "DROP", "INSERT", "UPDATE", "DELETE", "ALTER", "CREATE",
    "ATTACH", "DETACH", "COPY", "INSTALL", "LOAD", "PRAGMA",
    "EXPORT", "IMPORT", "SET", "RESET", "CHECKPOINT", "VACUUM",
)

# FROM/JOIN 后引用的表名（CTE 名由下面的正则单独提取）
_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_CTE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)


def _get_conn() -> duckdb.DuckDBPyConnection:
    """惰性创建内存 DuckDB，注册 sales 视图（read_csv_auto 自动推断列类型）。"""
    global _conn
    if _conn is None:
        csv_path = Path(settings.sales_csv).resolve()
        _conn = duckdb.connect(database=":memory:")
        _conn.execute(
            f"CREATE OR REPLACE VIEW sales AS SELECT * FROM read_csv_auto('{csv_path}')"
        )
    return _conn


def _validate_sql(sql: str) -> str:
    """只允许只读 SELECT/WITH 单语句，返回去尾分号后的规范化 SQL。

    抛 ValueError 说明拒绝原因（由上层工具转成错误提示返回给 LLM）。
    """
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("SQL 不能为空")

    upper = s.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("只允许只读 SELECT 查询")

    if ";" in s:
        raise ValueError("不支持多语句")

    for kw in _FORBIDDEN:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"禁止关键词：{kw}")

    # 表名白名单：FROM/JOIN 后引用的表名须为已注册表（sales）或本语句的 CTE 别名
    ctes = set(_CTE_RE.findall(upper))
    allowed = {t.upper() for t in _ALLOWED_TABLES} | {t.upper() for t in ctes}
    for t in _TABLE_RE.findall(s):
        if t.upper() not in allowed:
            raise ValueError(f"未知表：{t}")
    return s


def _format_table(cols: list[str], rows: list[tuple], truncated: bool, limit: int) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = ["NULL" if v is None else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    if truncated:
        lines.append(f"（结果超过 {limit} 行，已截断）")
    return "\n".join(lines)


def list_tables() -> str:
    """列出已注册表的结构（表名、行数、列名与类型），供 LLM 写 SQL 前参考。"""
    conn = _get_conn()
    n_rows = conn.sql("SELECT COUNT(*) FROM sales").fetchall()[0][0]
    cols = conn.sql(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'sales' ORDER BY ordinal_position"
    ).fetchall()
    lines = [f"表 sales（{n_rows} 行）"]
    lines += [f"- {name}: {dtype}" for name, dtype in cols]
    return "\n".join(lines)


def query(sql: str, limit: int = 50) -> str:
    """执行只读 SELECT，返回格式化表格（最多 limit 行）。"""
    sql = _validate_sql(sql)
    conn = _get_conn()
    rel = conn.sql(sql)
    cols = list(rel.columns)
    rows = rel.limit(limit + 1).fetchall()  # 多取一行判断是否截断
    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]
    return _format_table(cols, rows, truncated, limit)
