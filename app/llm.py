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


def _completion(messages: list[dict], temperature: float, tools: list[dict] | None = None):
    """带超时 + 指数退避重试的底层调用，返回 assistant message 对象。

    未配置 key 时抛出清晰错误（调用方据此降级）。tools 非空时启用函数调用。
    """
    if not settings.llm_api_key:
        raise RuntimeError("未设置 LLM_API_KEY，请在 .env 中配置后重启")

    kwargs: dict = dict(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        timeout=settings.llm_timeout,
    )
    if tools:
        kwargs["tools"] = tools

    last_exc: Exception | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            resp = _get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                # 非重试类错误（鉴权 401 / 参数 400 等）统一转 RuntimeError，
                # 让调用方（Agent 节点）用同一异常类型降级，而不是把原始异常冒泡成 500
                raise RuntimeError(f"LLM 调用失败：{exc}") from exc
        if attempt < settings.llm_max_retries - 1:
            time.sleep(settings.llm_backoff_base * (2 ** attempt))

    raise RuntimeError(f"LLM 调用失败（已重试 {settings.llm_max_retries} 次）：{last_exc}")


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    """同步调用 LLM，返回 assistant 文本。"""
    return _completion(messages, temperature).content or ""


def chat_with_tools(
    messages: list[dict], tools: list[dict], temperature: float = 0.0
) -> tuple[str, list[dict]]:
    """调用 LLM 并返回 (content, tool_calls)。

    tool_calls 归一化为 [{"id", "name", "arguments"}]，arguments 为 JSON 字符串。
    """
    msg = _completion(messages, temperature, tools=tools)
    calls = [
        # arguments 可能为 None（如无参工具 current_datetime），归一化为 "{}"，
        # 保证回填 assistant 消息时是合法 JSON 字符串，避免第二轮请求 400
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
        for tc in (msg.tool_calls or [])
    ]
    return msg.content or "", calls
