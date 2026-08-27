"""共享 RAG 管线：FastAPI 端点与 Agent 节点复用同一套检索器。"""
from __future__ import annotations

from pathlib import Path

import threading
from pathlib import Path

from ..config import settings
from .chunker import Chunk, chunk_text, load_document
from .embedder import embed_texts
from .reranker import rerank
from .retriever import HybridRetriever, RetrievedDoc

retriever = HybridRetriever(settings.milvus_uri, settings.collection_name)
# 多文档语料：source -> (chunks, vectors)，ingest/delete 时整体重建索引
_documents: dict[str, tuple[list[Chunk], list[list[float]]]] = {}
_lock = threading.Lock()


def _rebuild_index() -> None:
    """把所有已入库文档的 chunk/向量展平，整体重建索引。"""
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for cs, vs in _documents.values():
        chunks.extend(cs)
        vectors.extend(vs)
    retriever.reset()
    if chunks:
        retriever.index(chunks, vectors)


def ingest_document(path: str) -> int:
    """加载文档 -> 分块 -> embedding -> 追加到多文档语料，返回块数。"""
    text = load_document(Path(path))
    chunks = chunk_text(text, path, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
    with _lock:
        _documents[path] = (chunks, vectors)
        _rebuild_index()
    return len(chunks)


def delete_document(source: str) -> int:
    """按 source 删除一篇文档并重建索引，返回删除的块数（0 表示不存在）。"""
    with _lock:
        entry = _documents.pop(source, None)
        if entry is None:
            return 0
        _rebuild_index()
        return len(entry[0])


def list_documents() -> list[str]:
    """返回已入库文档的 source 列表。"""
    with _lock:
        return list(_documents.keys())


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """混合检索（BM25+向量，RRF 融合）+ 精排，返回 top_k 个文档。"""
    k = top_k or settings.top_k
    query_vec = embed_texts([query], settings.embedding_model)[0]
    docs = retriever.retrieve(query, query_vec, k * 2)
    idxs = rerank(query, [d.text for d in docs], settings.reranker_model, k)
    return [docs[i] for i in idxs]
