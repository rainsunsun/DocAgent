"""LangGraph 状态图：retrieve -> grade -> (generate | rewrite -> retrieve)。

自省回路：检索结果不足以回答时，改写 query 重查（最多 1 次），否则直接生成。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import nodes
from .state import AgentState

MAX_REWRITES = 1


def route_after_grade(state: AgentState) -> str:
    if state.get("grade") == "ok":
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate"
    return "rewrite"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("grade", nodes.grade_node)
    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("generate", nodes.generate_node)
    g.add_node("verify", nodes.verify_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


_graph = None


def run(question: str, user_id: str = "default") -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"question": question, "user_id": user_id, "rewrite_count": 0})
