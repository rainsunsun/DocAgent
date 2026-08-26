"""共享 RAG 管线：FastAPI 端点与 Agent 节点复用同一套检索器。"""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from .chunker import Chunk, chunk_text, load_document
from .embedder import embed_texts
from .reranker import rerank
from .retriever import HybridRetriever, RetrievedDoc

retriever = HybridRetriever(settings.milvus_uri, settings.collection_name)
_chunks: list[Chunk] = []


def ingest_document(path: str) -> int:
    """加载文档 -> 分块 -> embedding -> 入库，返回块数。"""
    global _chunks
    text = load_document(Path(path))
    chunks = chunk_text(text, path, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
    retriever.index(chunks, vectors)
    _chunks = chunks
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """混合检索（BM25+向量，RRF 融合）+ 精排，返回 top_k 个文档。"""
    k = top_k or settings.top_k
    query_vec = embed_texts([query], settings.embedding_model)[0]
    docs = retriever.retrieve(query, query_vec, k * 2)
    idxs = rerank(query, [d.text for d in docs], settings.reranker_model, k)
    return [docs[i] for i in idxs]
