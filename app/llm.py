"""LLM 调用封装（OpenAI 协议，默认 DeepSeek，可换 Qwen/GLM 等）。"""
from __future__ import annotations

import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from .config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """复用同一个 client（含连接池），避免每次调用都重建。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def _is_retryable(exc: Exception) -> bool:
    """只对瞬时错误重试：超时 / 网络 / 限流 / 5xx；4xx 参数错误重试无意义。"""
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and status >= 500


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """同步调用 LLM，返回 assistant 文本。

    带超时 + 指数退避重试，避免一次 API 抖动就让整个请求挂掉。
    未配置 key 时抛出清晰错误（调用方据此降级）。
    """
    if not settings.llm_api_key:
        raise RuntimeError("未设置 LLM_API_KEY，请在 .env 中配置后重启")

    last_exc: Exception | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            resp = _get_client().chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                timeout=settings.llm_timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
        if attempt < settings.llm_max_retries - 1:
            time.sleep(settings.llm_backoff_base * (2 ** attempt))

    raise RuntimeError(f"LLM 调用失败（已重试 {settings.llm_max_retries} 次）：{last_exc}")
