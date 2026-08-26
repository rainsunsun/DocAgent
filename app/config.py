"""应用配置：从 .env / 环境变量读取。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---- 模型 ----
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    # ---- 向量库（Milvus Lite 本地文件，无需独立服务） ----
    milvus_uri: str = os.getenv("MILVUS_URI", "./data/milvus.db")
    collection_name: str = os.getenv("COLLECTION_NAME", "doc_agent")

    # ---- 检索 ----
    top_k: int = int(os.getenv("TOP_K", "6"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))


settings = Settings()


def resolve_model(name: str) -> str:
    """优先用 ModelScope 本地缓存加载模型（国内可离线），否则回退 HF 模型 ID。"""
    local = Path.home() / ".cache" / "modelscope" / "models" / name.replace("/", "--") / "snapshots" / "master"
    return str(local) if (local / "config.json").exists() else name
