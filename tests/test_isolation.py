"""多用户隔离：每个用户独立的 retriever 与集合名。"""
from __future__ import annotations

import pytest

from app.rag import pipeline


def test_collection_name_per_user():
    assert pipeline._collection_for("alice") != pipeline._collection_for("bob")


def test_retriever_per_user():
    a1 = pipeline._get_retriever("alice")
    a2 = pipeline._get_retriever("alice")
    b = pipeline._get_retriever("bob")
    assert a1 is a2  # 同用户复用同一实例
    assert a1 is not b  # 不同用户隔离
    assert a1.collection != b.collection


def test_resolve_ingest_path_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "docs_dir", str(tmp_path))
    (tmp_path / "ok.md").write_text("hi", encoding="utf-8")
    assert pipeline._resolve_ingest_path("ok.md").name == "ok.md"


def test_resolve_ingest_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "docs_dir", str(tmp_path))
    with pytest.raises(ValueError):
        pipeline._resolve_ingest_path("../secret.md")


def test_resolve_ingest_path_rejects_absolute_outside(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "docs_dir", str(tmp_path))
    with pytest.raises(ValueError):
        pipeline._resolve_ingest_path("/etc/passwd")
