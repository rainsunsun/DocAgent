"""FastAPI 入口：/ingest 入库，/query 用 LangGraph Agent 问答。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent.graph import run as run_agent
from .rag import pipeline

app = FastAPI(title="DocAgent", version="0.2.0")


class IngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    faithfulness: str = ""
    faithfulness_reason: str = ""


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    """加载文档 -> 分块 -> embedding -> 入库。"""
    n = pipeline.ingest_document(req.path)
    return {"status": "ok", "chunks": n}


@app.post("/query")
def query(req: QueryRequest) -> QueryResponse:
    """检索 + 判断 + 改写 + 带引用生成 + 忠实度校验（LangGraph Agent）。"""
    result = run_agent(req.question)
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
