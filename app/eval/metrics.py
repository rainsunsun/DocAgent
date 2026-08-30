"""检索质量评估：命中率 / MRR，对比「纯向量 vs 纯 BM25 vs 混合」。

运行：python -m app.eval.metrics
说明：内置 data/eval_set.json 是 8 篇短文档的演示语料，主要验证评估流程；
     真实对比请把 evaluate(set_path) 换成自己的数据集跑。
"""
from __future__ import annotations

import json
import re
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


def _citation_validity(answer: str, n_docs: int) -> tuple[int, int]:
    """解析 [n] 引用，返回 (指向合法文档的引用数, 总引用数)。"""
    cites = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    valid = sum(1 for c in cites if 1 <= c <= n_docs)
    return valid, len(cites)


def evaluate_answers(set_path: str | Path, user_id: str = "eval") -> dict:
    """端到端答案层评估：对每条查询跑完整 Agent，统计忠实度判定 + 引用有效性。

    与 evaluate() 不同，这里会调 LLM（generate/verify 都要），属手动运行的
    答案质量评估，需要已配置 LLM_API_KEY。忠实度由 verify 节点以 LLM-as-judge
    方式判定（reference-free），非对金标准答案打分。
    """
    from ..agent.graph import run
    from ..rag import pipeline

    data = json.loads(Path(set_path).read_text(encoding="utf-8"))
    docs, queries = data["docs"], data["queries"]

    chunks = [Chunk(text=d["text"], source=str(d["id"]), chunk_index=d["id"]) for d in docs]
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
    pipeline.ingest_chunks(user_id, chunks, vectors)

    verdicts: dict[str, int] = {}
    valid_cites = total_cites = answered = 0
    for q in queries:
        r = run(q["query"], user_id=user_id)
        verdicts[r.get("faithfulness", "unknown")] = verdicts.get(r.get("faithfulness", "unknown"), 0) + 1
        vc, tc = _citation_validity(r.get("answer", ""), len(r.get("docs", [])))
        valid_cites += vc
        total_cites += tc
        ans = r.get("answer", "")
        if ans and "无法回答" not in ans and "未在知识库" not in ans:
            answered += 1

    return {
        "queries": len(queries),
        "answered": answered,
        "faithfulness": verdicts,
        "citation_validity": f"{valid_cites}/{total_cites}",
    }


def main() -> None:
    import sys

    path = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"
    n_queries = len(json.loads(path.read_text(encoding="utf-8"))["queries"])

    if "--answers" in sys.argv:
        print(f"\n端到端答案质量（{n_queries} 条查询，需 LLM）：\n")
        res = evaluate_answers(path)
        print(f"回答数：{res['answered']}/{res['queries']}")
        print(f"忠实度判定分布：{res['faithfulness']}")
        print(f"引用有效性（合法/总）：{res['citation_validity']}")
        return

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
