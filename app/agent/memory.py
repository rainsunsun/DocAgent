"""短期会话记忆。

同一套 load/append/clear 接口，两个存储后端：
- memory（默认）：进程内 dict，零依赖，进程重启即清空；
- redis：单键存 JSON + TTL，多进程/多实例共享，进程重启不丢（短 TTL 兜底自动回收）。

面试点：为什么生产要用 Redis 而不是进程内 dict——
- 多 worker/多实例下，进程内 dict 各自为政，同一 session 落到不同实例时记忆「串台/丢失」；
- Redis 是独立进程，所有实例读写同一份，天然共享；
- TTL 让过期会话自动回收，避免键无限增长。

选型说明：单键 JSON（read-modify-write）而非 Redis LIST/Lua，是「最简单正确」的版本；
若并发写同一 session 需要严格原子性，再升级成 LIST + LTRIM 或 Lua 脚本。
"""
from __future__ import annotations

import json
import threading

MAX_TURNS = 8  # 每个 session 保留最近 8 轮（每轮 = 1 user + 1 assistant）


class _DictBackend:
    """进程内 dict 后端（默认，单机开发 / 测试）。"""

    def __init__(self) -> None:
        self._mem: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._mem.get(session_id, []))

    def append(self, session_id: str, question: str, answer: str) -> None:
        if not session_id or not answer:
            return
        with self._lock:
            hist = self._mem.setdefault(session_id, [])
            hist.append({"role": "user", "content": question})
            hist.append({"role": "assistant", "content": answer})
            del hist[: -(MAX_TURNS * 2)]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._mem.pop(session_id, None)


class _RedisBackend:
    """Redis 后端：键 docagent:mem:{session_id}，值 JSON 数组，写时刷新 TTL。"""

    def __init__(self, client, ttl: int) -> None:
        self._client = client
        self._ttl = ttl

    @staticmethod
    def _key(session_id: str) -> str:
        return f"docagent:mem:{session_id}"

    def load(self, session_id: str) -> list[dict]:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    def append(self, session_id: str, question: str, answer: str) -> None:
        if not session_id or not answer:
            return
        hist = self.load(session_id)
        hist.append({"role": "user", "content": question})
        hist.append({"role": "assistant", "content": answer})
        del hist[: -(MAX_TURNS * 2)]
        self._client.set(
            self._key(session_id), json.dumps(hist, ensure_ascii=False), ex=self._ttl
        )

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


_store = _DictBackend()
_configured = False


def _ensure_configured() -> None:
    """首次使用时按配置选择后端；redis 不可用时静默回退到 dict。"""
    global _store, _configured
    if _configured:
        return
    _configured = True
    from ..config import settings

    if settings.memory_backend != "redis":
        return
    try:
        import redis  # 懒加载：本地没装 redis-py 也不影响默认 dict 后端
    except ImportError:
        return
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _store = _RedisBackend(client, settings.memory_ttl)
    except Exception:
        # 连接失败（Redis 没起、地址错等）：不阻塞服务，回退 dict。
        _store = _DictBackend()


def load(session_id: str) -> list[dict]:
    """返回该 session 的历史消息副本。"""
    _ensure_configured()
    return _store.load(session_id)


def append(session_id: str, question: str, answer: str) -> None:
    """追加一轮对话，并裁剪到最近 MAX_TURNS 轮。"""
    _ensure_configured()
    _store.append(session_id, question, answer)


def clear(session_id: str) -> None:
    """清空某个 session 的记忆（测试 / 重置用）。"""
    _ensure_configured()
    _store.clear(session_id)
