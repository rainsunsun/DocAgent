"""LLM 调用封装（OpenAI 协议，默认 DeepSeek，可换 Qwen/GLM 等）。"""
from __future__ import annotations

from openai import OpenAI

from .config import settings


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """同步调用 LLM，返回 assistant 文本。未配置 key 时抛出清晰错误。"""
    if not settings.llm_api_key:
        raise RuntimeError("未设置 LLM_API_KEY，请在 .env 中配置后重启")
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
