"""数据分析 Agent 评估：逐题跑 ReAct，比对答案数值与 ground truth。

运行：python -m app.eval.data_metrics   （需 LLM_API_KEY，手动本地跑，不上 CI）
对比 RAG 答案层评估（语义相似度 / LLM-judge），这里用**数值精确比对**：数据分析的
正确答案是确定的数，可复现、无需 LLM 判分，是比文本相似度更严格的确定性指标。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..agent.react import run


def _extract_numbers(text: str) -> list[float]:
    """提取文本中所有数字（含千分位、小数、负号）。"""
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def _is_close(got: float, expected: float, rel: float = 0.01, abs_tol: float = 0.5) -> bool:
    """数值接近：相对误差 rel 内，或绝对误差 abs_tol 内（防除零 / 小值误判）。"""
    if expected == 0:
        return abs(got - expected) <= abs_tol
    return abs(got - expected) / abs(expected) <= rel or abs(got - expected) <= abs_tol


def _check(answer: str, case: dict) -> bool:
    """判断 Agent 答案是否覆盖 ground truth，按 kind 分三类比对。"""
    kind = case.get("kind", "number")
    expected = case["expected"]
    if kind == "text":
        return expected in answer
    nums = _extract_numbers(answer)
    if kind == "pct":
        # 百分比题：答案可能写「13.96%」也可能写小数「0.1396」，都算对。
        # 比绝对值，忽略「下降 / 增长」的符号表达差异。
        target = abs(expected)
        return any(_is_close(abs(n), target) or _is_close(abs(n), target * 100) for n in nums)
    return any(_is_close(n, expected) for n in nums)


def evaluate_data(set_path: str | Path, user_id: str = "eval") -> dict:
    """端到端数据分析评估：逐题跑 ReAct Agent，统计数值正确率。"""
    data = json.loads(Path(set_path).read_text(encoding="utf-8"))
    questions = data["questions"]

    passed = 0
    details: list[dict] = []
    for q in questions:
        r = run(q["question"], user_id=user_id)
        ans = r.get("answer", "")
        ok = _check(ans, q)
        passed += int(ok)
        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": ans,
                "steps": r.get("step", 0),
                "pass": ok,
            }
        )
    return {
        "queries": len(questions),
        "passed": passed,
        "accuracy": round(passed / len(questions), 3) if questions else 0.0,
        "details": details,
    }


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "data_eval_set.json"
    res = evaluate_data(path)
    print(f"\n数据分析 Agent 评估（{res['queries']} 题，需 LLM）：\n")
    for d in res["details"]:
        mark = "PASS" if d["pass"] else "FAIL"
        print(f"[{mark}] ({d['steps']} 步) {d['question']}")
        print(f"       答：{d['answer'][:120]}")
    print(f"\n数值正确率：{res['passed']}/{res['queries']} = {res['accuracy']}")


if __name__ == "__main__":
    main()
