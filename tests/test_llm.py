"""LLM 调用封装的异常归一化测试（不触发真实网络）。"""
from __future__ import annotations

import pytest

import app.llm as llm


def test_completion_normalizes_non_retryable_error(monkeypatch):
    # 非重试类错误（401 鉴权 / 400 参数）应统一转 RuntimeError，
    # 让 Agent 节点用同一异常类型降级，而不是把原始异常冒泡成 500
    monkeypatch.setattr(llm.settings, "llm_api_key", "dummy")

    class FakeCompletions:
        def create(self, **kwargs):
            raise ValueError("bad request")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
    with pytest.raises(RuntimeError, match="LLM 调用失败"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_completion_raises_without_key(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_with_tools_normalizes_none_arguments(monkeypatch):
    # 无参工具 current_datetime 可能让模型返回 arguments=null，应归一化为 "{}"
    class FakeFunction:
        name = "current_datetime"
        arguments = None

    class FakeToolCall:
        id = "call_1"
        function = FakeFunction()

    class FakeMsg:
        content = None
        tool_calls = [FakeToolCall()]

    monkeypatch.setattr(llm, "_completion", lambda *a, **k: FakeMsg())
    content, calls = llm.chat_with_tools([{"role": "user", "content": "hi"}], [{}])
    assert calls == [{"id": "call_1", "name": "current_datetime", "arguments": "{}"}]
