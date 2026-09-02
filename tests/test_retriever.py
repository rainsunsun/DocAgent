"""RRF 融合的单元测试（纯函数，无模型加载）。"""
from __future__ import annotations

import app.rag.retriever as retriever_mod
from app.rag.chunker import Chunk
from app.rag.retriever import HybridRetriever


def test_rrf_both_hit_ranks_first():
    # doc1 两路都命中、doc0 只在 dense，融合后 doc1 应排第一
    dense = {0: 0.9, 1: 0.8, 2: 0.7}
    sparse = {1: 0.9, 3: 0.8}
    ranked = HybridRetriever._rrf(dense, sparse)
    ids = [i for i, _ in ranked]
    assert ids[0] == 1


def test_rrf_single_list_still_ranked():
    # 只在某一路命中的文档也应出现在融合结果里（缺席一路不额外惩罚）
    ranked = HybridRetriever._rrf({0: 0.9, 1: 0.5}, {0: 0.8})
    ids = [i for i, _ in ranked]
    assert 1 in ids
    assert ids[0] == 0  # 两路都命中者仍排前


def test_rrf_score_is_reciprocal_rank_sum():
    # 同一 doc 在两路各自第 1 名：融合分 = 2 / (k + 1)
    ranked = HybridRetriever._rrf({0: 0.9}, {0: 0.8}, k=60)
    assert abs(ranked[0][1] - 2 / 61) < 1e-9


def test_add_is_incremental_and_retrievable(monkeypatch):
    # 强制走 numpy fallback，避免依赖 milvus-lite 环境；验证 index + add 后两路都能检索到
    monkeypatch.setattr(retriever_mod, "_milvus_lite_available", lambda: False)
    c0 = Chunk(text="苹果 香蕉", source="s1", chunk_index=0)
    c1 = Chunk(text="电脑 手机", source="s2", chunk_index=1)
    ret = retriever_mod.HybridRetriever("unused", "test_add")
    ret.index([c0], [[1.0, 0.0]])
    ret.add([c1], [[0.0, 1.0]])
    docs = ret.retrieve("苹果", [1.0, 0.0], top_k=2)
    assert [d.chunk_index for d in docs] == [0, 1]
