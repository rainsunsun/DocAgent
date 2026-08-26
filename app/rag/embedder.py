"""Embedding 封装。默认 BGE（中文友好）。"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import resolve_model


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(resolve_model(model_name))


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    model = get_embedder(model_name)
    return model.encode(texts, normalize_embeddings=True).tolist()
