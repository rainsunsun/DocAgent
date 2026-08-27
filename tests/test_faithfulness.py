"""忠实度审校输出解析的单元测试。"""
from __future__ import annotations

from app.agent.nodes import _parse_faithfulness


def test_supported():
    assert _parse_faithfulness("SUPPORTED")[0] == "supported"


def test_partially():
    assert _parse_faithfulness("PARTIALLY: 结论正确，但数字无出处")[0] == "partial"


def test_unsupported():
    assert _parse_faithfulness("UNSUPPORTED")[0] == "unsupported"


def test_unknown_fallback():
    assert _parse_faithfulness("随便说点什么")[0] == "unknown"
