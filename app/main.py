"""FastAPI 入口：/ingest 入库，/query 用 LangGraph Agent 问答。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import react
from .agent.graph import run as run_agent
from .rag import pipeline

app = FastAPI(title="DocAgent", version="0.2.0")


class IngestRequest(BaseModel):
    path: str
    user_id: str = "default"


class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"
    session_id: str = ""


class DeleteRequest(BaseModel):
    source: str
    user_id: str = "default"


class Source(BaseModel):
    text: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    faithfulness: str = ""
    faithfulness_reason: str = ""


class AgentResponse(BaseModel):
    answer: str
    steps: int


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    """加载文档 -> 分块 -> embedding -> 入库。"""
    try:
        n = pipeline.ingest_document(req.user_id, req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "chunks": n}


@app.post("/query")
def query(req: QueryRequest) -> QueryResponse:
    """检索 + 判断 + 改写 + 带引用生成 + 忠实度校验（LangGraph Agent）。"""
    result = run_agent(req.question, req.user_id)
    sources = [
        Source(text=d.text, source=d.source, chunk_index=d.chunk_index)
        for d in result.get("docs", [])
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        faithfulness=result.get("faithfulness", ""),
        faithfulness_reason=result.get("faithfulness_reason", ""),
    )


@app.post("/agent")
def agent(req: QueryRequest) -> AgentResponse:
    """工具调用型 Agent（ReAct）：LLM 决策 -> 调工具 -> 观察 -> 再决策。

    传 session_id 时保留该会话的短期记忆（最近 8 轮），不传则无状态。
    """
    result = react.run(req.question, req.user_id, session_id=req.session_id)
    return AgentResponse(answer=result.get("answer", ""), steps=result.get("step", 0))


@app.get("/documents")
def documents(user_id: str = "default") -> dict:
    """列出该用户已入库文档的 source。"""
    return {"sources": pipeline.list_documents(user_id)}


@app.post("/delete")
def delete_doc(req: DeleteRequest) -> dict:
    """按 source 删除一篇文档并重建索引。"""
    n = pipeline.delete_document(req.user_id, req.source)
    return {"status": "ok" if n else "not_found", "removed_chunks": n}
