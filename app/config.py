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

    # ---- 入库（/ingest 只允许读该目录内文件，防路径穿越）----
    docs_dir: str = os.getenv("DOCS_DIR", "./data")

    # ---- 数据分析（DuckDB 只读查询的 CSV 数据源）----
    sales_csv: str = os.getenv("SALES_CSV", "./data/sales.csv")

    # ---- 检索 ----
    top_k: int = int(os.getenv("TOP_K", "6"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ---- LLM 可靠性 ----
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    llm_backoff_base: float = float(os.getenv("LLM_BACKOFF_BASE", "1.0"))

    # ---- Agent 改写（语义漂移门控）----
    rewrite_min_similarity: float = float(os.getenv("REWRITE_MIN_SIMILARITY", "0.5"))

    # ---- 短期记忆 ----
    memory_backend: str = os.getenv("MEMORY_BACKEND", "memory")  # memory | redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    memory_ttl: int = int(os.getenv("MEMORY_TTL", "3600"))  # 秒，默认 1 小时


settings = Settings()


def resolve_model(name: str) -> str:
    """优先用 ModelScope 本地缓存加载模型（国内可离线），否则回退 HF 模型 ID。

    ModelScope 快照目录名通常是版本哈希而非固定 "master"，故遍历 snapshots/ 下
    任意子目录，取第一个含 config.json 的（字典序，结果确定）。
    """
    base = Path.home() / ".cache" / "modelscope" / "models" / name.replace("/", "--") / "snapshots"
    if base.is_dir():
        for snap in sorted(base.iterdir()):
            if (snap / "config.json").exists():
                return str(snap)
    return name
