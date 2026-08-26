"""tokenizer 单元测试。"""
from __future__ import annotations

from app.rag.tokenizer import tokenize


def test_chinese_split():
    toks = tokenize("混合检索的优缺点")
    assert len(toks) > 1  # 中文被切开，而不是一整坨


def test_english_kept_whole():
    toks = tokenize("使用 RAGFlow 和 base_url")
    assert "RAGFlow" in toks
    assert "base_url" in toks


def test_punctuation_filtered():
    toks = tokenize("什么，是。RAG？")
    assert "，" not in toks and "。" not in toks and "？" not in toks
