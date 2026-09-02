"""Agent 工具：RAG 检索 + 确定性工具（计算、日期）。

工具分两类，对应「哪些环节用大模型、哪些用确定性程序」：
- search: 走 RAG 混合检索（语义能力，靠模型）；
- calculator / current_datetime: 纯确定性程序，不碰 LLM，结果精确可复现。
"""
from __future__ import annotations

import ast
import json
import operator
from datetime import datetime

from ..rag import pipeline

from . import data_engine


# ---- 工具实现（纯函数，便于单测）----

def search(query: str, user_id: str = "default") -> str:
    """在个人知识库中检索与 query 最相关的片段，返回带来源的文本。"""
    docs = pipeline.retrieve(user_id, query, top_k=6)
    if not docs:
        return "未检索到相关片段。"
    return "\n\n".join(f"[来源 {d.source}#{d.chunk_index}]\n{d.text}" for d in docs)


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST):
    """只允许常量与四则运算/幂/取模，用 AST 白名单杜绝 eval 任意代码执行。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def calculator(expression: str) -> str:
    """安全地求算术表达式，如 "(3+5)*2"；非法表达式返回错误提示而非抛异常。"""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"计算失败：{exc}"


def current_datetime() -> str:
    """返回当前日期时间（本地时区）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_tables() -> str:
    """列出数据分析可用的表结构（表名、行数、列名与类型）。"""
    return data_engine.list_tables()


def sql_query(sql: str) -> str:
    """执行只读 SQL 查询（仅 SELECT），返回格式化结果；被拒绝或出错时返回错误提示。"""
    try:
        return data_engine.query(sql)
    except ValueError as exc:
        return f"查询被拒绝：{exc}"
    except Exception as exc:  # DuckDB 语法错误 / 列不存在等，转成提示让 LLM 修正重试
        return f"查询失败：{exc}"


# ---- OpenAI tool schema + 分发 ----

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "在个人知识库中检索与 query 相关的片段并返回带来源文本。"
                "当问题需要文档、知识库、参考资料中的信息时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词或完整问题"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "计算算术表达式并返回精确结果。"
                "仅当问题涉及需要精确计算的数学运算（四则运算、幂、取模）时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "算术表达式，如 (3+5)*2 或 2**10"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "返回当前日期和时间。当问题涉及「今天/现在/当前日期时间」时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "列出数据分析可用的数据表结构（表名、行数、列名与类型）。"
                "写 SQL 查询前应先调用它确认表名和字段，避免写错列名。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "在零售销售数据表 sales 上执行只读 SQL（仅 SELECT）查询，返回格式化结果。"
                "支持聚合、过滤、分组、排序。"
                "涉及销售额、销量、同比、环比、占比、排名等数据分析问题时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "要执行的 SELECT 语句，表名为 sales"}},
                "required": ["sql"],
            },
        },
    },
]


def execute_tool(name: str, arguments, user_id: str = "default") -> str:
    """按 name 分发执行工具。arguments 可为 JSON 字符串或 dict，返回结果文本。"""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return f"工具参数不是合法 JSON：{arguments}"
    else:
        args = arguments or {}
    if name == "search":
        return search(args.get("query", ""), user_id=user_id)
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "current_datetime":
        return current_datetime()
    if name == "list_tables":
        return list_tables()
    if name == "sql_query":
        return sql_query(args.get("sql", ""))
    return f"未知工具：{name}"
