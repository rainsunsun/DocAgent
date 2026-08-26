"""精排：用交叉编码器对「问题-文档」逐一打分，重排粗召回结果。"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from ..config import resolve_model


@lru_cache(maxsize=1)
def get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(resolve_model(model_name))


def rerank(query: str, docs: list[str], model_name: str, top_k: int) -> list[int]:
    """返回按相关度降序的文档下标（取 top_k）。"""
    if not docs:
        return []
    model = get_reranker(model_name)
    scores = model.predict([[query, d] for d in docs])
    ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
    return ranked[:top_k]
