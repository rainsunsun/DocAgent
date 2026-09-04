# DocAgent — Agent RAG 问答 + 自然语言数据分析系统

> 基于 LangGraph + Milvus + DuckDB 的 Agent 系统：既能做知识库问答（Agent 自主判断「检索够不够、要不要改写 query 重查」，回答带引用 + 忠实度校验），也能做自然语言数据分析（查表 → 写只读 SQL → 精确计算 → 下结论），并配套可量化的评估。

## 当前状态（全部完成 ✅）

- [x] **P1 基础 RAG**：文档加载 → 分块 → embedding → 混合检索（BM25 + 向量 + RRF）→ 精排
- [x] **P2 Agent 层**：LangGraph 编排（检索 → 相关性判断 → query 改写重查 → 带引用生成 → 忠实度校验）
- [x] **P3 记忆 + MCP**：短期记忆（进程内 / Redis 双后端）+ MCP server；长期记忆待完成
- [x] **数据分析 Agent**：DuckDB 只读查询工具（list_tables / sql_query）+ 零售销售数据 + 指标口径知识库
- [x] **P4 评估**：检索层 hit@k/MRR + 答案层忠实度/引用/语义相似度（`--answers`）
- [x] **P4 部署**：Docker 一键起（已验证：数据分析 + RAG 问答端到端跑通，镜像 ~550MB）

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 配置环境变量（P2 生成回答需要 LLM_API_KEY）
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（DeepSeek 等 OpenAI 协议均可）

# 3. 启动服务
uvicorn app.main:app --reload

# 4. 入库文档（可重复调用，追加到该 user 的多文档语料；user_id 隔离不同用户）
#    路径相对 DOCS_DIR（默认 ./data）解析，只允许该目录内文件，防路径穿越
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d "{\"path\": \"sample.md\", \"user_id\": \"alice\"}"

# 5. 提问（Agent 自动判断/改写，回答带 [1][2] 引用 + 忠实度校验）
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"什么是 RAG？\", \"user_id\": \"alice\"}"

# 6. 查看该用户已入库文档 / 按 source 删除一篇
curl "localhost:8000/documents?user_id=alice"
curl -X POST localhost:8000/delete -H "Content-Type: application/json" -d "{\"source\": \"sample.md\", \"user_id\": \"alice\"}"

# 7. 数据分析（Agent 查表 → 写只读 SQL → 精确算，用户无需懂 SQL）
curl -X POST localhost:8000/agent -H "Content-Type: application/json" -d "{\"question\": \"2025 年全年销售额是多少？\", \"user_id\": \"alice\"}"
```

> 所有端点都可带 `user_id`（默认 `default`）做**多用户隔离**：每个用户独立的 collection / BM25 / 向量，A 用户的文档不会被 B 检索到。

> 首次运行会联网下载 embedding / reranker 模型（BGE 系列）。向量库用 **Milvus Lite**（本地文件，无需独立服务）。

## Docker 部署（已验证）

```bash
# 1. 构建镜像（~550MB，依赖走清华源 + CPU 版 torch，国内可构建）
docker compose build

# 2. 启动（复用宿主机 ModelScope 模型缓存，避免重复下载 6.5G 模型）
docker compose up -d

# 3. 验证（数据分析 / 入库 / 问答）
curl -X POST localhost:8000/agent -H "Content-Type: application/json" -d '{"question":"2025 年全年销售额是多少元？"}'
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"path":"sample.md","user_id":"alice"}'
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question":"什么是 RAG？","user_id":"alice"}'
```

> - **生产离线部署**：`docker build --build-arg BAKE_MODELS=1 .` 把 BGE 模型烘焙进镜像（+6.5G，新机器 pull 即用、无需联网下模型）。
> - **国内构建**：Docker Hub 直连被墙时，先 `docker pull docker.m.daocloud.io/library/python:3.11-slim && docker tag docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim` 拉基础镜像。

## 目录结构

```
doc-agent/
├── app/
│   ├── main.py              # FastAPI 入口（/ingest /query）
│   ├── config.py            # 配置（.env）
│   ├── llm.py               # LLM 调用封装（OpenAI 协议）
│   ├── rag/                 # P1：基础 RAG
│   │   ├── chunker.py       #   分块
│   │   ├── embedder.py      #   embedding
│   │   ├── retriever.py     #   混合检索（BM25 + 向量 + RRF）
│   │   ├── reranker.py      #   精排
│   │   └── pipeline.py      #   共享检索管线（ingest / retrieve）
│   ├── agent/               # P2：LangGraph 编排 + 带引用生成
│   │   ├── graph.py         #   状态图
│   │   ├── nodes.py         #   retrieve / grade / rewrite / generate / verify
│   │   ├── react.py         #   ReAct 工具调用循环（含死循环保护）
│   │   ├── data_engine.py   #   数据分析引擎（DuckDB 只读查询 + SQL 安全校验）
│   │   ├── memory.py        #   P3：短期记忆（进程内 / Redis 双后端）
│   │   ├── tools.py         #   工具封装（search / sql_query / calculator 等）
│   │   └── state.py         #   状态定义
│   ├── mcp/                 # P3：MCP server（JSON-RPC over stdio）
│   └── eval/                # 评估：检索 hit@k/MRR + 答案忠实度/语义相似度
├── docs/
│   ├── architecture.md      # 架构设计 + 技术选型理由
│   └── evaluation.md        # 评估方法与指标
├── data/
│   ├── sample.md            # 示例文档（RAG 语料）
│   ├── metrics.md           # 指标口径（数据分析）
│   └── sales.csv            # 零售销售数据（数据分析，脚本生成）
├── scripts/generate_sales.py  # 确定性生成 sales.csv
├── tests/                   # 单元测试
├── Dockerfile
└── docker-compose.yml
```

## 技术选型理由

| 层 | 选型 | 理由 |
|----|------|------|
| Agent 框架 | LangGraph | 状态图编排、可控的自省回路；比 LangChain 链式更「真 Agent」 |
| 向量库 | Milvus（Lite） | 国产主流、生产级；Lite 本地文件免部署 |
| 检索 | BM25 + 向量 + RRF | 混合检索兼顾关键词精确与语义召回；RRF 融合无需调权重 |
| Embedding | BGE（bge-m3） | 中文效果好 |
| 精排 | bge-reranker-v2-m3 | 显著提升检索精度 |
| LLM | DeepSeek（OpenAI 协议） | 可平滑替换 Qwen / GLM / 本地 Ollama |
| 数据引擎 | DuckDB | 嵌入式无服务，read_csv_auto 直接查 CSV，与 Milvus Lite 同思路 |
| 后端 | FastAPI | 异步、现代，Flask 可平滑迁移 |

详见 [docs/architecture.md](docs/architecture.md)。

## 评估

分三层：检索层免 LLM，答案层与数据分析层需 LLM（均手动本地跑）：

```bash
# 检索层：纯向量 / 纯 BM25 / 混合 三种召回对比 hit@k / MRR
python -m app.eval.metrics

# 答案层：完整 Agent 跑一遍，统计忠实度、引用有效性、语义相似度、词重叠 F1
python -m app.eval.metrics --answers   # 需要 LLM_API_KEY

# 数据分析层：8 道自然语言数据题跑 ReAct，答案数值与 ground truth 精确比对
python -m app.eval.data_metrics   # 需要 LLM_API_KEY
```

- **检索层**：`data/eval_set.json`（8 篇文档 14 条查询）上对比 dense / sparse / hybrid 的 hit@k、MRR。
- **答案层**：忠实度由 verify 节点以 LLM-judge 判定（supported/partial/unsupported）；引用有效性抓「引用了不存在的文档」这类幻觉；语义相似度 + 词重叠 F1 是**对固定参考答案打分的确定性指标**，可复现、能进回归。
- **数据分析层**：`data/data_eval_set.json`（8 题：销售额/同比/环比/占比/排名/客单价/品类过滤）逐题跑 ReAct，答案数值与 DuckDB 算出的 ground truth 做**数值精确比对**（相对误差 1% 内），比文本相似度更严格、可复现。
- **单元测试**：`pytest tests -q`（98 个用例，含节点 LLM mock），push / PR 由 GitHub Actions 自动跑（CI 只跑单测，评估脚本因需下载模型 / LLM key 保持本地）。

## 路线图

- **P1**（已实现）基础 RAG 跑通
- **P2**（已实现）LangGraph Agent：检索 → 相关性判断 → query 改写重查 → 带引用生成
- **P3**（部分完成）短期记忆 + MCP server 已实现；长期记忆待做
- **P4** 评估 + 部署已落地（检索/答案/数据分析三层评估 + Docker 一键起），待补：纯 RAG vs RAG+Agent 对比实验


