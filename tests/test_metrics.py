"""评估工具纯函数测试。"""
from __future__ import annotations

from app.eval.metrics import _citation_validity, _map_to_chunk_ids, _strip_citations, _token_f1
from app.rag.chunker import Chunk


def test_citation_validity_all_valid():
    assert _citation_validity("答案是 RAG [1][2]", 3) == (2, 2)


def test_citation_validity_out_of_range():
    assert _citation_validity("答案是 [1][9]", 3) == (1, 2)


def test_citation_validity_no_cites():
    assert _citation_validity("没有引用", 3) == (0, 0)


def test_strip_citations():
    assert _strip_citations("RAG 是[1][2]一种技术") == "RAG 是一种技术"


def test_token_f1_identical():
    assert _token_f1("rag 检索", "rag 检索") == 1.0


def test_token_f1_no_overlap():
    assert _token_f1("hello world", "foo bar") == 0.0


def test_map_to_chunk_ids_non_sequential():
    # doc id 非 0..n-1 顺序时，映射必须用 chunk_index 而非数组下标
    chunks = [
        Chunk(text="a", source="0", chunk_index=10),
        Chunk(text="b", source="1", chunk_index=20),
    ]
    assert _map_to_chunk_ids({0: 0.9, 1: 0.8}, chunks) == {10: 0.9, 20: 0.8}
