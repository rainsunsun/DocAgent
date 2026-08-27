"""Agent 工具：把 RAG 能力包装成 LLM 可调用的工具（供后续 ReAct 型 Agent 使用）。"""
from __future__ import annotations

from langchain_core.tools import tool

from ..rag import pipeline


@tool
def search(query: str, user_id: str = "default") -> str:
    """在已入库文档中检索与 query 最相关的片段，返回带来源的文本。"""
    docs = pipeline.retrieve(user_id, query, top_k=6)
    if not docs:
        return "未检索到相关片段。"
    return "\n\n".join(f"[来源 {d.source}#{d.chunk_index}]\n{d.text}" for d in docs)


@tool
def query_rewrite(question: str) -> str:
    """把口语化问题改写为更适合检索的关键词查询。"""
    from ..llm import chat

    prompt = f"把下面的问题改写成一个更具体、更适合检索的查询，只输出查询本身：\n{question}"
    return (chat([{"role": "user", "content": prompt}]) or "").strip()
