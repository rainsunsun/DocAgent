"""工具调用型 Agent（ReAct）的单元测试：纯工具 + 节点分支（mock LLM）。"""
from __future__ import annotations

from unittest.mock import Mock

from langgraph.graph import END

from app.agent import memory, react, tools


# ---- 纯工具 ----

def test_calculator_basic():
    assert tools.calculator("(3+5)*2") == "16"


def test_calculator_power():
    assert tools.calculator("2**10") == "1024"


def test_calculator_blocks_code_execution():
    # AST 白名单必须挡住任意代码执行（eval 注入）
    assert tools.calculator("__import__('os').system('ls')").startswith("计算失败")


def test_current_datetime_format():
    out = tools.current_datetime()
    assert len(out) == 19 and out[4] == "-" and out[10] == " "


def test_execute_tool_dispatch(monkeypatch):
    monkeypatch.setattr(tools, "search", Mock(return_value="命中"))
    assert tools.execute_tool("search", '{"query": "RAG"}') == "命中"
    assert tools.execute_tool("calculator", '{"expression": "1+1"}') == "2"
    assert tools.execute_tool("unknown", "{}").startswith("未知工具")


def test_execute_tool_bad_json():
    assert tools.execute_tool("search", "not-json").startswith("工具参数")


# ---- ReAct 节点分支 ----

def _calls(name="search", args='{"query": "RAG"}'):
    return [{"id": "call_1", "name": name, "arguments": args}]


def test_agent_node_requests_tool(monkeypatch):
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("", _calls())))
    out = react.agent_node({"messages": [], "step": 0})
    assert out["tool_calls"] == _calls()
    assert out["messages"][-1]["role"] == "assistant"


def test_agent_node_returns_answer(monkeypatch):
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("最终答案", [])))
    out = react.agent_node({"messages": [], "step": 0})
    assert out["answer"] == "最终答案"
    assert out["tool_calls"] == []


def test_agent_node_message_format_no_tool_calls_key_on_answer(monkeypatch):
    # 最终答案的 assistant 消息不应带空 tool_calls（非标准格式，部分后端会拒）
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("最终答案", [])))
    out = react.agent_node({"messages": [], "step": 0})
    msg = out["messages"][-1]
    assert "tool_calls" not in msg
    assert msg == {"role": "assistant", "content": "最终答案"}


def test_agent_node_message_format_openai_tool_calls(monkeypatch):
    # 工具调用消息必须是 OpenAI 规范格式：id / type=function / function.name / function.arguments
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("", _calls())))
    out = react.agent_node({"messages": [], "step": 0})
    tc = out["messages"][-1]["tool_calls"][0]
    assert tc == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "RAG"}'},
    }


def test_agent_node_forced_stop_at_max_steps(monkeypatch):
    monkeypatch.setattr(react, "chat", Mock(return_value="强制回答"))
    out = react.agent_node({"messages": [], "step": react.MAX_STEPS})
    assert out["answer"] == "强制回答"
    assert out["tool_calls"] == []


def test_tool_node_executes_and_increments(monkeypatch):
    monkeypatch.setattr(react, "execute_tool", Mock(return_value="工具结果"))
    out = react.tool_node({"messages": [], "tool_calls": _calls(), "step": 0})
    assert out["step"] == 1
    assert out["messages"][-1]["role"] == "tool"
    assert out["messages"][-1]["content"] == "工具结果"


def test_route_after_agent():
    assert react.route_after_agent({"tool_calls": _calls()}) == "tools"
    assert react.route_after_agent({"tool_calls": []}) == END


def test_graph_completes_one_tool_round(monkeypatch):
    # 第一轮请求工具，第二轮给出最终答案，验证整条回路能走通
    calls = iter([("", _calls()), ("用了工具后的答案", [])])
    monkeypatch.setattr(
        react, "chat_with_tools", Mock(side_effect=lambda *a, **k: next(calls))
    )
    monkeypatch.setattr(react, "execute_tool", Mock(return_value="工具结果"))
    result = react.run("问题", "alice")
    assert result["answer"] == "用了工具后的答案"
    assert result["step"] == 1


# ---- 短期记忆 ----

def test_memory_accumulates_turns():
    memory.clear("s1")
    memory.append("s1", "Q1", "A1")
    memory.append("s1", "Q2", "A2")
    hist = memory.load("s1")
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    assert hist[0]["content"] == "Q1"


def test_memory_caps_turns():
    memory.clear("s2")
    for i in range(memory.MAX_TURNS + 3):
        memory.append("s2", f"Q{i}", f"A{i}")
    hist = memory.load("s2")
    assert len(hist) == memory.MAX_TURNS * 2
    assert hist[0]["content"] == f"Q{3}"  # 丢掉最老的 3 轮


def test_memory_skips_empty_answer():
    memory.clear("s3")
    memory.append("s3", "空答案", "")
    assert memory.load("s3") == []


def test_run_persists_history_with_session(monkeypatch):
    memory.clear("s4")
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("你好", [])))
    react.run("你好", "alice", session_id="s4")
    hist = memory.load("s4")
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[-1]["content"] == "你好"


def test_run_without_session_no_memory(monkeypatch):
    memory.clear("s5")
    monkeypatch.setattr(react, "chat_with_tools", Mock(return_value=("你好", [])))
    react.run("你好", "alice")
    assert memory.load("s5") == []


def test_redis_backend_roundtrip_and_trim():
    class FakeClient:
        def __init__(self):
            self.data: dict[str, str] = {}
            self.ttl = None

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None):
            self.data[key] = value
            self.ttl = ex

        def delete(self, key):
            self.data.pop(key, None)

    client = FakeClient()
    be = memory._RedisBackend(client, ttl=3600)
    be.append("s", "Q1", "A1")
    be.append("s", "Q2", "A2")
    hist = be.load("s")
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    assert hist[0]["content"] == "Q1"
    assert client.ttl == 3600
    be.clear("s")
    assert be.load("s") == []


def test_redis_backend_bad_json_returns_empty():
    class FakeClient:
        def get(self, key):
            return "not-json"

    be = memory._RedisBackend(FakeClient(), ttl=3600)
    assert be.load("s") == []
