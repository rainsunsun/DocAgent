"""MCP Server（P3，待实现）。

用 mcp 库暴露两个 tool：
    search(query) -> 检索结果
    ingest(path)  -> 入库
供 Claude Desktop / 其他 MCP client 调用。

面试点：MCP 解决了什么问题（模型-工具解耦、标准协议）？对比 Function Calling？
"""
from __future__ import annotations
