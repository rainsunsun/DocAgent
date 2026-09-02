"""数据分析数据引擎测试：SQL 安全校验 + 查询正确性 + 工具分发。"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

import app.agent.data_engine as de
from app.agent import tools

SALES_CSV = Path(__file__).parent.parent / "data" / "sales.csv"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(de.settings, "sales_csv", str(SALES_CSV))
    monkeypatch.setattr(de, "_conn", None)


# ---- SQL 安全校验 ----

def test_validate_allows_select():
    assert de._validate_sql("SELECT * FROM sales") == "SELECT * FROM sales"


def test_validate_allows_with_cte():
    de._validate_sql("WITH t AS (SELECT * FROM sales) SELECT * FROM t")


def test_validate_rejects_dml():
    for sql in (
        "DROP TABLE sales",
        "INSERT INTO sales VALUES (1)",
        "UPDATE sales SET amount=0",
        "DELETE FROM sales",
    ):
        with pytest.raises(ValueError):
            de._validate_sql(sql)


def test_validate_rejects_non_select():
    with pytest.raises(ValueError):
        de._validate_sql("SHOW TABLES")


def test_validate_rejects_multi_statement():
    with pytest.raises(ValueError):
        de._validate_sql("SELECT * FROM sales; DROP TABLE sales")


def test_validate_rejects_unknown_table():
    with pytest.raises(ValueError):
        de._validate_sql("SELECT * FROM secret_table")


# ---- 查询正确性 ----

def test_list_tables_has_schema():
    out = de.list_tables()
    assert "sales" in out
    assert "amount" in out and "order_date" in out


def test_query_count_matches_csv():
    out = de.query("SELECT COUNT(*) AS cnt FROM sales")
    assert "500" in out


def test_query_group_by_returns_all_regions():
    out = de.query("SELECT region, SUM(amount) AS total FROM sales GROUP BY region")
    for r in ("华东", "华北", "华南", "西南", "西北"):
        assert r in out


def test_query_matches_direct_duckdb():
    # 与直接用 DuckDB 读同一 CSV 的 ground truth 一致（CAST 成 DOUBLE 避免精度歧义）
    gt = duckdb.connect().execute(
        f"SELECT region, CAST(SUM(amount) AS DOUBLE) AS total FROM read_csv_auto('{SALES_CSV}') "
        "GROUP BY region ORDER BY region"
    ).fetchall()
    conn = de._get_conn()
    got = conn.sql(
        "SELECT region, CAST(SUM(amount) AS DOUBLE) AS total FROM sales GROUP BY region ORDER BY region"
    ).fetchall()
    assert got == gt


# ---- 工具分发 ----

def test_execute_tool_sql_query_ok():
    out = tools.execute_tool("sql_query", {"sql": "SELECT COUNT(*) FROM sales"})
    assert "500" in out


def test_execute_tool_sql_query_rejected():
    out = tools.execute_tool("sql_query", {"sql": "DROP TABLE sales"})
    assert "查询被拒绝" in out


def test_execute_tool_list_tables():
    out = tools.execute_tool("list_tables", {})
    assert "sales" in out
