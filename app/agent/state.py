"""Agent 共享状态定义。"""
from __future__ import annotations

from typing import TypedDict

from ..rag.retriever import RetrievedDoc


class AgentState(TypedDict, total=False):
    question: str
    query: str
    docs: list[RetrievedDoc]
    grade: str
    rewrite_count: int
    answer: str
