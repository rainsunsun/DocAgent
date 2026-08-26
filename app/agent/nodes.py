"""Agent 节点：检索 -> 相关性判断 -> query 改写 -> 带引用生成。"""
from __future__ import annotations

from ..config import settings
from ..llm import chat
from ..rag import pipeline
from .state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """用当前 query（或原始问题）检索，结果写回 docs。"""
    query = state.get("query") or state["question"]
    docs = pipeline.retrieve(query, settings.top_k)
    return {"docs": docs, "query": query}


def grade_node(state: AgentState) -> dict:
    """自省：判断当前检索结果是否足以回答问题。"""
    docs = state.get("docs", [])
    if not docs:
        return {"grade": "rewrite"}
    question = state["question"]
    context = "\n".join(f"[{i + 1}] {d.text[:200]}" for i, d in enumerate(docs))
    prompt = (
        "你负责判断检索片段是否足以回答用户问题。\n"
        f"问题：{question}\n\n检索片段：\n{context}\n\n"
        "如果这些片段足以回答，只回复 YES；否则只回复 NO。"
    )
    try:
        ans = (chat([{"role": "user", "content": prompt}]) or "").strip().upper()
    except RuntimeError:
        return {"grade": "ok"}  # 无 LLM 时跳过判断，直接生成
    return {"grade": "ok" if ans.startswith("YES") else "rewrite"}


def rewrite_node(state: AgentState) -> dict:
    """把原始问题改写为更适合检索的查询。"""
    count = state.get("rewrite_count", 0)
    prompt = (
        f"把下面的问题改写成一个更具体、更适合检索的查询，只输出查询本身：\n{state['question']}"
    )
    try:
        new_query = (chat([{"role": "user", "content": prompt}]) or "").strip()
    except RuntimeError:
        new_query = state["question"]
    return {"query": new_query, "rewrite_count": count + 1}


def generate_node(state: AgentState) -> dict:
    """基于检索结果生成带来源编号的回答。"""
    question = state["question"]
    docs = state.get("docs", [])
    if not docs:
        return {"answer": "未在知识库中检索到相关内容，无法回答该问题。"}
    context = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
    prompt = (
        "你是严谨的知识库问答助手。请仅基于下面提供的资料回答用户问题，"
        "在引用处标注来源编号（如 [1][2]）。资料不足时明确说明，不要编造。\n\n"
        f"资料：\n{context}\n\n问题：{question}\n\n回答："
    )
    try:
        answer = chat([{"role": "user", "content": prompt}], temperature=0.1) or ""
    except RuntimeError as e:
        answer = f"（{e}）检索到 {len(docs)} 个相关片段，请配置 LLM_API_KEY 后重试。"
    return {"answer": answer}
