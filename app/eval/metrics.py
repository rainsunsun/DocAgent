"""检索质量评估：命中率 / MRR，对比「纯向量 vs 纯 BM25 vs 混合」。

运行：python -m app.eval.metrics
说明：内置 data/eval_set.json 是 8 篇短文档的演示语料，主要验证评估流程；
     真实对比请把 evaluate(set_path) 换成自己的数据集跑。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from ..rag.chunker import Chunk
from ..rag.embedder import embed_texts
from ..rag.retriever import HybridRetriever


def _rank_of_relevant(ranked_ids: list[int], relevant: list[int]) -> int:
    """返回首个命中相关文档的排名（1-based），未命中返回 0。"""
    for rank, idx in enumerate(ranked_ids, start=1):
        if idx in relevant:
            return rank
    return 0


def hit_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r > 0) / len(ranks)


def evaluate(set_path: str | Path, top_k: int = 5) -> dict:
    data = json.loads(Path(set_path).read_text(encoding="utf-8"))
    docs, queries = data["docs"], data["queries"]

    # 每篇文档当作一个 chunk，doc_id == chunk_index，便于对齐 ground truth
    chunks = [Chunk(text=d["text"], source=str(d["id"]), chunk_index=d["id"]) for d in docs]
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)

    coll = settings.collection_name + "_eval"
    ret = HybridRetriever(settings.milvus_uri, coll)
    ret.reset()  # 清空旧集合，保证 doc id 对齐
    ret.index(chunks, vectors)

    results: dict[str, dict] = {}
    for mode in ("dense", "sparse", "hybrid"):
        ranks: list[int] = []
        for q in queries:
            qv = embed_texts([q["query"]], settings.embedding_model)[0]
            if mode == "dense":
                scored = ret._dense(qv, top_k)
            elif mode == "sparse":
                scored = ret._sparse(q["query"], top_k)
            else:
                scored = {d.chunk_index: d.score for d in ret.retrieve(q["query"], qv, top_k)}
            ranked_ids = [i for i, _ in sorted(scored.items(), key=lambda x: -x[1])]
            ranks.append(_rank_of_relevant(ranked_ids, q["relevant"]))
        results[mode] = {
            "hit@1": round(hit_at_k(ranks, 1), 3),
            "hit@3": round(hit_at_k(ranks, 3), 3),
            f"hit@{top_k}": round(hit_at_k(ranks, top_k), 3),
            "mrr": round(mrr(ranks), 3),
            "ranks": ranks,
        }
    return results


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"
    n_queries = len(json.loads(path.read_text(encoding="utf-8"))["queries"])
    res = evaluate(path)
    print(f"\n检索质量对比（{n_queries} 条查询，top_k=5）：\n")
    print(f"{'模式':<6}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for mode, m in res.items():
        print(f"{mode:<6}{m['hit@1']:>8}{m['hit@3']:>8}{m['hit@5']:>8}{m['mrr']:>8}")
    print("\nranks（1=第一名命中，0=未命中）：")
    for mode, m in res.items():
        print(f"{mode:<6}{m['ranks']}")


if __name__ == "__main__":
    main()
