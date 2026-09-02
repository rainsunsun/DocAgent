"""配置解析测试：ModelScope 本地缓存探测。"""
from __future__ import annotations

from pathlib import Path

from app.config import resolve_model


def test_resolve_model_finds_versioned_snapshot(tmp_path, monkeypatch):
    # 快照目录名是版本哈希而非固定 "master" 时也要能探测到
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snap = (
        tmp_path / ".cache" / "modelscope" / "models" / "BAAI--bge-m3"
        / "snapshots" / "v1.0.0"
    )
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model("BAAI/bge-m3") == str(snap)


def test_resolve_model_falls_back_to_hf_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert resolve_model("BAAI/bge-m3") == "BAAI/bge-m3"
