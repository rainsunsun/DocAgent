"""Agent 节点：检索 -> 相关性判断 -> query 改写 -> 带引用生成。"""
from __future__ import annotations

from ..config import settings
from ..llm import chat
from ..rag import pipeline
from ..rag.embedder import similarity
from .state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """用当前 query（或原始问题）检索，结果写回 docs。"""
    query = state.get("query") or state["question"]
    user_id = state.get("user_id", "default")
    docs = pipeline.retrieve(user_id, query, settings.top_k)
    return {"docs": docs, "query": query}


def _is_yes(ans: str) -> bool:
    """健壮解析 grade 回复：兼容 YES/No/是的/否 及带标点、前后缀。"""
    a = (ans or "").strip().upper()
    if a.startswith("YES") or a.startswith("是"):
        return True
    if a.startswith("NO") or a.startswith("否") or a.startswith("不"):
        return False
    # 兜底：包含 YES 且不含 NO 才算 ok，其余按不足处理（宁可改写，不误判为够）
    return "YES" in a and "NO" not in a


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
        ans = chat([{"role": "user", "content": prompt}], temperature=0.0) or ""
    except RuntimeError:
        return {"grade": "ok"}  # 无 LLM 时跳过判断，直接生成
    return {"grade": "ok" if _is_yes(ans) else "rewrite"}


def rewrite_node(state: AgentState) -> dict:
    """把原始问题改写为更适合检索的查询，并做语义漂移门控。"""
    count = state.get("rewrite_count", 0)
    question = state["question"]
    prompt = (
        f"把下面的问题改写成一个更具体、更适合检索的查询，只输出查询本身：\n{question}"
    )
    try:
        new_query = (chat([{"role": "user", "content": prompt}]) or "").strip()
    except RuntimeError:
        new_query = question

    # 语义漂移门控：改写结果与原文相似度过低说明越改越偏，弃用、沿用原文。
    if new_query and new_query != question:
        try:
            if similarity(question, new_query, settings.embedding_model) < settings.rewrite_min_similarity:
                new_query = question
        except Exception:
            new_query = question  # embedding 失败时保守起见，不阻塞主流程
    return {"query": new_query or question, "rewrite_count": count + 1}


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


def _parse_faithfulness(ans: str) -> tuple[str, str]:
    """解析审校输出：SUPPORTED / PARTIALLY / UNSUPPORTED（可后跟冒号理由）。"""
    a = (ans or "").strip()
    u = a.upper()
    if u.startswith("SUPPORTED"):
        return "supported", a
    if u.startswith("PARTIAL"):
        return "partial", a
    if u.startswith("UNSUPPORTED"):
        return "unsupported", a
    return "unknown", a


def verify_node(state: AgentState) -> dict:
    """忠实度校验：判断回答中的事实性陈述是否都被检索片段支持。"""
    answer = state.get("answer", "")
    docs = state.get("docs", [])
    if not answer:
        return {"faithfulness": "n/a", "faithfulness_reason": "无回答"}
    if not docs:
        # 无检索片段时 generate 已明确拒答，属于「诚实未编造」
        return {"faithfulness": "supported", "faithfulness_reason": "无检索片段，回答已拒答，未编造"}

    context = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
    prompt = (
        "你是严格的审校员。请判断下面「回答」中的事实性陈述是否都能被「资料」支持，"
        "尤其注意数字、名称、结论是否在资料中出现。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{state['question']}\n\n"
        f"回答：\n{answer}\n\n"
        "只输出一行：SUPPORTED / PARTIALLY / UNSUPPORTED，可选地后跟冒号和一句简短理由（指出哪句不被支持）。"
    )
    try:
        out = chat([{"role": "user", "content": prompt}], temperature=0.0) or ""
        level, reason = _parse_faithfulness(out)
    except RuntimeError:
        return {"faithfulness": "unknown", "faithfulness_reason": "无 LLM，跳过忠实度校验"}
    return {"faithfulness": level, "faithfulness_reason": reason}
