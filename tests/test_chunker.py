"""chunker 单元测试。"""
from __future__ import annotations

from app.rag.chunker import chunk_text


def test_empty_text():
    assert chunk_text("", "s") == []


def test_single_short_paragraph():
    chunks = chunk_text("一句话。", "s", chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "一句话。"
    assert chunks[0].chunk_index == 0


def test_overlap_preserved():
    text = "\n".join(f"第{i}段内容" for i in range(50))
    chunks = chunk_text(text, "s", chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # 相邻块之间有重叠：前一块结尾出现在后一块开头
    assert chunks[1].text.startswith(chunks[0].text[-10:])


def test_chunk_index_sequential():
    text = "\n".join(f"第{i}段内容" for i in range(100))
    chunks = chunk_text(text, "s", chunk_size=50, overlap=5)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
