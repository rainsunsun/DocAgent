"""grade / rewrite 节点的单元测试（mock LLM，覆盖脏输出分支）。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent import nodes


def _docs(n: int = 1) -> list:
    return [SimpleNamespace(text=f"检索片段 {i}") for i in range(n)]


# ---- _is_yes 纯函数 ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("YES", True),
        ("yes", True),
        ("YES。", True),
        ("是的", True),
        ("No", False),
        ("NO", False),
        ("否", False),
        ("不足以回答", False),
        ("随便一句", False),
    ],
)
def test_is_yes(text: str, expected: bool):
    assert nodes._is_yes(text) is expected


# ---- grade_node ----
def test_grade_empty_docs_returns_rewrite():
    assert nodes.grade_node({"docs": []}) == {"grade": "rewrite"}


def test_grade_yes_marks_ok(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(return_value="YES"))
    assert nodes.grade_node({"question": "q", "docs": _docs()})["grade"] == "ok"


def test_grade_no_marks_rewrite(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(return_value="No"))
    assert nodes.grade_node({"question": "q", "docs": _docs()})["grade"] == "rewrite"


def test_grade_llm_unavailable_skips_check(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr(nodes, "chat", boom)
    assert nodes.grade_node({"question": "q", "docs": _docs()})["grade"] == "ok"


def test_grade_uses_temperature_zero(monkeypatch):
    m = Mock(return_value="YES")
    monkeypatch.setattr(nodes, "chat", m)
    nodes.grade_node({"question": "q", "docs": _docs()})
    assert m.call_args.kwargs["temperature"] == 0.0


# ---- rewrite_node ----
def test_rewrite_kept_when_similar(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(return_value="改写的查询"))
    monkeypatch.setattr(nodes, "similarity", Mock(return_value=0.8))
    r = nodes.rewrite_node({"question": "原问题"})
    assert r["query"] == "改写的查询"
    assert r["rewrite_count"] == 1


def test_rewrite_reverted_on_drift(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(return_value="跑偏的查询"))
    monkeypatch.setattr(nodes, "similarity", Mock(return_value=0.1))
    r = nodes.rewrite_node({"question": "原问题"})
    assert r["query"] == "原问题"


def test_rewrite_llm_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(side_effect=RuntimeError("no key")))
    assert nodes.rewrite_node({"question": "原问题"})["query"] == "原问题"


def test_rewrite_empty_result_falls_back(monkeypatch):
    monkeypatch.setattr(nodes, "chat", Mock(return_value=""))
    assert nodes.rewrite_node({"question": "原问题"})["query"] == "原问题"
