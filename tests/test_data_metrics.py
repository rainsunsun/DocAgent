"""数据分析评估的纯函数测试：数字提取 + 数值比对 + 三类 kind 判定（免 LLM）。"""
from __future__ import annotations

from app.eval import data_metrics as dm


# ---- 数字提取 ----

def test_extract_numbers_plain():
    assert dm._extract_numbers("722514.03 元") == [722514.03]


def test_extract_numbers_thousand_sep():
    assert dm._extract_numbers("722,514.03 元") == [722514.03]


def test_extract_numbers_negative():
    assert dm._extract_numbers("同比下降 -0.1396") == [-0.1396]


def test_extract_numbers_multiple():
    assert dm._extract_numbers("从 100 涨到 150") == [100.0, 150.0]


def test_extract_numbers_percent():
    # 只提取数值部分，不含 % 符号
    assert dm._extract_numbers("下降 13.96%") == [13.96]


# ---- 数值接近 ----

def test_is_close_exact():
    assert dm._is_close(722514.03, 722514.03)


def test_is_close_rounded():
    assert dm._is_close(722514, 722514.03)  # 绝对误差 0.03 < 0.5


def test_is_close_rejects_far():
    assert not dm._is_close(72, 722514.03)


def test_is_close_zero_guard():
    assert dm._is_close(0.1, 0.0)
    assert not dm._is_close(1.0, 0.0)


# ---- 三类 kind 判定 ----

def test_check_number():
    case = {"kind": "number", "expected": 722514.03}
    assert dm._check("2025 年销售额为 722,514.03 元", case)
    assert not dm._check("销售额约 72 万", case)


def test_check_pct_percent_form():
    case = {"kind": "pct", "expected": -0.1396}
    assert dm._check("同比下滑 13.96%", case)


def test_check_pct_decimal_form():
    case = {"kind": "pct", "expected": -0.1396}
    assert dm._check("同比下降了 0.1396", case)


def test_check_pct_rejects_wrong():
    case = {"kind": "pct", "expected": -0.1396}
    assert not dm._check("同比增长了 20%", case)


def test_check_text():
    case = {"kind": "text", "expected": "华东"}
    assert dm._check("销售额最高的地区是华东", case)
    assert not dm._check("销售额最高的地区是华北", case)
