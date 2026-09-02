"""MCP Server：把 DocAgent 的能力通过 MCP 协议暴露给外部 client（Claude Desktop 等）。

运行：python -m app.mcp.server

面试点：MCP 与 Function Calling 的区别——
- Function Calling：应用内部，LLM 通过 tool_calls 调用本进程的工具（见 agent/react.py）；
- MCP：标准协议（JSON-RPC over stdio），把工具暴露给任意 MCP client，模型与工具解耦。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..rag import pipeline

mcp = FastMCP("docagent")


@mcp.tool()
def search(query: str, user_id: str = "default") -> str:
    """在个人知识库中检索与 query 最相关的片段，返回带来源的文本。"""
    docs = pipeline.retrieve(user_id, query, top_k=6)
    if not docs:
        return "未检索到相关片段。"
    return "\n\n".join(f"[来源 {d.source}#{d.chunk_index}]\n{d.text}" for d in docs)


@mcp.tool()
def ingest(path: str, user_id: str = "default") -> str:
    """把指定路径的文档入库（分块 -> embedding -> 建索引），返回入库块数。"""
    n = pipeline.ingest_document(user_id, path)
    return f"已入库 {n} 个分块"


if __name__ == "__main__":
    mcp.run()
