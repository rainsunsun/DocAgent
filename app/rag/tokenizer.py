"""中文/英文混合分词：中文用 jieba，英文/数字/下划线保持整体。

为什么需要它：BM25 依赖词频统计，而中文没有空格分隔，直接用 .split() 会把
一整段当成一个词，BM25 完全失效。这里先摘出英文/数字连续串，再对中文用 jieba。
"""
from __future__ import annotations

import re

import jieba

_ALNUM = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[一-鿿]")


def _is_meaningful(tok: str) -> bool:
    """只保留含中文字符或字母数字的词元，过滤纯标点/空白。"""
    return bool(_ALNUM.search(tok) or _CJK.search(tok))


def tokenize(text: str) -> list[str]:
    """返回有意义的词元列表。

    先摘出英文/数字/下划线连续串（保住 RAGFlow、base_url 这类词），
    再对其余的中文片段用 jieba 切词，最后过滤纯标点。
    """
    tokens: list[str] = []
    pos = 0
    for m in _ALNUM.finditer(text):
        if m.start() > pos:
            tokens.extend(t for t in jieba.lcut(text[pos:m.start()]) if _is_meaningful(t))
        tokens.append(m.group())
        pos = m.end()
    if pos < len(text):
        tokens.extend(t for t in jieba.lcut(text[pos:]) if _is_meaningful(t))
    return tokens
