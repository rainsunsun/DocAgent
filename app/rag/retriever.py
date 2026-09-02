"""混合检索：BM25（稀疏）+ 向量（稠密），RRF 融合。

面试点：为什么混合检索比纯向量好？RRF 如何无参数融合两路排序？

向量后端：优先 Milvus（生产级、分布式）；未安装 milvus-lite 时回退到内存 numpy
余弦相似度（Windows 本地无 milvus-lite，用此 fallback 快速跑通）。两者都是 COSINE
度量、结果等价，接口一致。
"""
from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .tokenizer import tokenize


def _milvus_lite_available() -> bool:
    return importlib.util.find_spec("milvus_lite") is not None


@dataclass
class RetrievedDoc:
    text: str
    source: str
    chunk_index: int
    score: float


class HybridRetriever:
    def __init__(self, milvus_uri: str, collection_name: str):
        self.collection = collection_name
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._vectors = None  # numpy 后端用的归一化向量矩阵
        self._use_milvus = _milvus_lite_available()
        self._lock = threading.Lock()  # 保护 ingest 与 query 并发时的共享可变状态
        if self._use_milvus:
            from pymilvus import MilvusClient

            self.client = MilvusClient(milvus_uri)
        else:
            self.client = None

    def _ensure_collection(self, dim: int) -> None:
        if self.client.has_collection(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            dimension=dim,
            metric_type="COSINE",
        )

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """写入 chunk 向量，并构建 BM25 稀疏索引。"""
        if not chunks:
            return
        with self._lock:
            self._chunks = chunks
            # 中文用 jieba 分词、英文/数字保持整体，否则 BM25 对中文完全失效
            self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
            if self._use_milvus:
                self._ensure_collection(len(vectors[0]))
                rows = [
                    {
                        "id": i,
                        "vector": vectors[i],
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                    }
                    for i, c in enumerate(chunks)
                ]
                self.client.insert(collection_name=self.collection, data=rows)
            else:
                import numpy as np

                self._vectors = np.asarray(vectors, dtype="float32")

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """增量追加 chunk：不重建既有数据（向量只插新增行，BM25 需整体重拟合）。

        仅支持 append-only（调用方保证不删除中间块）；删除走 reset + index 全量重建。
        """
        if not chunks:
            return
        with self._lock:
            start = len(self._chunks)
            self._chunks.extend(chunks)
            # BM25Okapi 无增量 API，需用完整语料重拟合（纯词频统计，代价远低于 embedding/建索引）
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])
            if self._use_milvus:
                self._ensure_collection(len(vectors[0]))
                rows = [
                    {
                        "id": start + i,
                        "vector": vectors[i],
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                    }
                    for i, c in enumerate(chunks)
                ]
                self.client.insert(collection_name=self.collection, data=rows)
            else:
                import numpy as np

                new = np.asarray(vectors, dtype="float32")
                self._vectors = (
                    np.concatenate([self._vectors, new], axis=0)
                    if self._vectors is not None
                    else new
                )

    def reset(self) -> None:
        """清空向量库，便于重复评测时保持 doc id 对齐。"""
        with self._lock:
            if self._use_milvus and self.client.has_collection(self.collection):
                self.client.drop_collection(self.collection)
            self._chunks = []
            self._bm25 = None
            self._vectors = None

    def _dense(self, query_vec: list[float], n: int) -> dict[int, float]:
        if self._use_milvus:
            hits = self.client.search(
                collection_name=self.collection,
                data=[query_vec],
                limit=n,
                output_fields=["text", "source", "chunk_index"],
            )[0]
            return {int(h["id"]): float(h["distance"]) for h in hits}

        import numpy as np

        q = np.asarray(query_vec, dtype="float32")
        sims = self._vectors @ q  # embedding 已归一化，点积即余弦相似度
        top = np.argsort(-sims)[:n]
        return {int(i): float(sims[i]) for i in top}

    def _sparse(self, query: str, n: int) -> dict[int, float]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return {i: float(scores[i]) for i in ranked}

    @staticmethod
    def _rrf(
        dense: dict[int, float],
        sparse: dict[int, float],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """倒数排名融合：对两路各自按分数排名，累加 1/(k+rank)，无需调权重。"""
        fused: dict[int, float] = {}
        for rank, (idx, _) in enumerate(sorted(dense.items(), key=lambda x: -x[1])):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sorted(sparse.items(), key=lambda x: -x[1])):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def retrieve(self, query: str, query_vec: list[float], top_k: int = 6) -> list[RetrievedDoc]:
        with self._lock:
            if not self._chunks:  # 空索引直接返回，避免 _vectors/_bm25 为 None 时崩溃
                return []
            dense = self._dense(query_vec, top_k * 2)
            sparse = self._sparse(query, top_k * 2)
            ranked = self._rrf(dense, sparse)[:top_k]
            docs: list[RetrievedDoc] = []
            for idx, score in ranked:
                c = self._chunks[idx]
                docs.append(
                    RetrievedDoc(text=c.text, source=c.source, chunk_index=c.chunk_index, score=round(score, 4))
                )
            return docs
