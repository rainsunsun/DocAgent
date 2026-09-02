"""工具调用型 Agent（ReAct 回路）：LLM 决策 -> 调工具 -> 观察 -> 再决策。

与 graph.py 的「自省式 RAG 管线」并存，二者区别：
- 自省管线：LLM 只输出文本（YES/NO、改写 query），代码决定控制流；
- ReAct 回路：LLM 通过 tool_calls 主动要求调用工具，代码只负责执行。

死循环防护：MAX_STEPS 限制工具调用轮数，到达上限后强制直接作答。
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..llm import chat, chat_with_tools
from . import memory
from .tools import TOOL_SCHEMAS, execute_tool


class ReactState(TypedDict, total=False):
    question: str
    user_id: str
    messages: list[dict]
    tool_calls: list[dict]
    step: int
    answer: str


MAX_STEPS = 3

SYSTEM_PROMPT = (
    "你是一个能使用工具的严谨助手。规则：\n"
    "1. 问题需要文档/知识库信息时调用 search；\n"
    "2. 需要精确数学计算时调用 calculator；\n"
    "3. 涉及「今天/现在/当前日期时间」时调用 current_datetime；\n"
    "4. 计算、日期这类确定性任务，用工具，不要靠语言模型猜；\n"
    "5. 能直接回答的简单问题直接回答，不要滥用工具。\n"
    "6. 涉及销售额、销量、同比、环比、占比、排名等数据分析问题时：先调用 list_tables 看表结构，再用 sql_query 查数，必要时用 calculator 精确计算；\n"
    "7. 指标口径不明确（如「销售额」是否含税、季度怎么划分）时，先用 search 查口径文档再算；\n"
    "8. 数据分析的回答必须给出具体数字和计算过程，不要凭印象猜。"
)


def agent_node(state: ReactState) -> dict:
    """LLM 决策：要么要求调工具（返回 tool_calls），要么给出最终回答。"""
    messages = list(state.get("messages", []))
    step = state.get("step", 0)

    if step >= MAX_STEPS:
        # 到达步数上限：强制终止，不再提供工具，直接要求给出最终回答。
        messages = messages + [{
            "role": "user",
            "content": "已达到工具调用上限，请不要再调用工具，直接根据已有信息给出最终回答。",
        }]
        content = chat(messages, temperature=0.0)
        messages = messages + [{"role": "assistant", "content": content}]
        return {"messages": messages, "answer": content, "tool_calls": [], "step": step}

    content, tool_calls = chat_with_tools(messages, TOOL_SCHEMAS, temperature=0.0)
    assistant_msg: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        # 仅在有工具调用时才带 tool_calls；arguments 已由 chat_with_tools 归一化为 JSON 字符串
        assistant_msg["tool_calls"] = [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in tool_calls
        ]
    messages = messages + [assistant_msg]

    out: dict = {"messages": messages, "tool_calls": tool_calls, "step": step}
    if not tool_calls:
        out["answer"] = content
    return out


def tool_node(state: ReactState) -> dict:
    """执行上一步 agent_node 请求的所有工具，把结果作为 tool 消息追加进历史。"""
    messages = list(state.get("messages", []))
    for c in state.get("tool_calls", []):
        result = execute_tool(c["name"], c["arguments"], state.get("user_id", "default"))
        messages = messages + [{"role": "tool", "tool_call_id": c["id"], "content": result}]
    return {"messages": messages, "tool_calls": [], "step": state.get("step", 0) + 1}


def route_after_agent(state: ReactState) -> str:
    return "tools" if state.get("tool_calls") else END


def build_graph():
    g = StateGraph(ReactState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


_graph = None


def run(question: str, user_id: str = "default", session_id: str = "") -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    history = memory.load(session_id) if session_id else []
    initial: ReactState = {
        "question": question,
        "user_id": user_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": question},
        ],
        "step": 0,
    }
    result = _graph.invoke(initial)
    if session_id:
        memory.append(session_id, question, result.get("answer", ""))
    return result
