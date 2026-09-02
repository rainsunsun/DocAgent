"""共享 RAG 管线：FastAPI 端点与 Agent 节点复用同一套检索器。

多用户隔离：每个 user_id 拥有独立的 retriever 实例（独立 collection / BM25 / 向量），
A 用户的文档不会被 B 检索到。
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..config import settings
from .chunker import Chunk, chunk_text, load_document
from .embedder import embed_texts
from .reranker import rerank
from .retriever import HybridRetriever, RetrievedDoc

_lock = threading.RLock()
# 多用户隔离：user_id -> { source -> (chunks, vectors) }；user_id -> retriever
_documents: dict[str, dict[str, tuple[list[Chunk], list[list[float]]]]] = {}
_retrievers: dict[str, HybridRetriever] = {}


def _collection_for(user_id: str) -> str:
    """每个用户一个独立集合名（Milvus collection / numpy 内存实例都按此隔离）。"""
    return f"{settings.collection_name}_{user_id}"


def _get_retriever(user_id: str) -> HybridRetriever:
    """按 user_id 获取（或惰性创建）专属检索器。"""
    with _lock:
        ret = _retrievers.get(user_id)
        if ret is None:
            ret = HybridRetriever(settings.milvus_uri, _collection_for(user_id))
            _retrievers[user_id] = ret
        return ret


def _rebuild_index(user_id: str) -> None:
    """把某用户所有已入库文档展平，整体重建其索引。"""
    retriever = _get_retriever(user_id)
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for cs, vs in _documents.get(user_id, {}).values():
        chunks.extend(cs)
        vectors.extend(vs)
    retriever.reset()
    if chunks:
        retriever.index(chunks, vectors)


def _append_to_index(user_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """只把新 chunk 增量追加进该用户的索引（不重建既有数据）。"""
    _get_retriever(user_id).add(chunks, vectors)


def _resolve_ingest_path(path: str) -> Path:
    """把入库路径限制在 docs_dir 内，拒绝绝对越界与 .. 穿越（防任意文件读取）。"""
    root = Path(settings.docs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"非法路径（仅允许 {root} 目录内）：{path}")
    if not candidate.is_file():
        raise ValueError(f"文件不存在或不可读：{path}")
    return candidate


def ingest_document(user_id: str, path: str) -> int:
    """加载文档 -> 分块 -> embedding -> 追加到该用户语料，返回块数。"""
    resolved = _resolve_ingest_path(path)
    text = load_document(resolved)
    chunks = chunk_text(text, path, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
    with _lock:
        docs = _documents.setdefault(user_id, {})
        is_new = path not in docs
        docs[path] = (chunks, vectors)
        # 新文档走增量追加；重复入库同 source 走全量重建（需移除旧块）
        if is_new:
            _append_to_index(user_id, chunks, vectors)
        else:
            _rebuild_index(user_id)
    return len(chunks)


def ingest_chunks(
    user_id: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    source: str = "eval",
) -> int:
    """直接入库已分块/向量化的内容（供评估/测试用），返回块数。"""
    with _lock:
        docs = _documents.setdefault(user_id, {})
        is_new = source not in docs
        docs[source] = (chunks, vectors)
        if is_new:
            _append_to_index(user_id, chunks, vectors)
        else:
            _rebuild_index(user_id)
    return len(chunks)


def delete_document(user_id: str, source: str) -> int:
    """按 source 删除该用户一篇文档并重建索引，返回删除块数（0 表示不存在）。"""
    with _lock:
        entry = _documents.get(user_id, {}).pop(source, None)
        if entry is None:
            return 0
        _rebuild_index(user_id)
        return len(entry[0])


def list_documents(user_id: str) -> list[str]:
    """返回该用户已入库文档的 source 列表。"""
    with _lock:
        return list(_documents.get(user_id, {}))


def retrieve(user_id: str, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """该用户视角下的混合检索（BM25+向量，RRF 融合）+ 精排。"""
    k = top_k or settings.top_k
    with _lock:
        has_docs = bool(_documents.get(user_id))
    if not has_docs:
        return []
    retriever = _get_retriever(user_id)
    query_vec = embed_texts([query], settings.embedding_model)[0]
    docs = retriever.retrieve(query, query_vec, k * 2)
    idxs = rerank(query, [d.text for d in docs], settings.reranker_model, k)
    return [docs[i] for i in idxs]
