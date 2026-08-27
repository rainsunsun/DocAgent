# DocAgent — Agent RAG 问答系统

> 基于 LangGraph + Milvus 的 Agent RAG 问答系统：Agent 自主判断「检索结果够不够、要不要改写 query 重查」，回答带来源引用，并配套可量化的检索质量评估。

## 当前状态（P1 ✅ / P2 ✅ / P3–P4 待完成）

- [x] **P1 基础 RAG**：文档加载 → 分块 → embedding → 混合检索（BM25 + 向量 + RRF）→ 精排
- [x] **P2 Agent 层**：LangGraph 编排（检索 → 相关性判断 → query 改写重查 → 带引用生成 → 忠实度校验）
- [ ] **P3 记忆 + MCP**：短期 / 长期记忆 + MCP server 暴露工具
- [ ] **P4 评估 + 部署**：检索命中率 / 回答忠实度 + Docker 一键起

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 配置环境变量（P2 生成回答需要 LLM_API_KEY）
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（DeepSeek 等 OpenAI 协议均可）

# 3. 启动服务
uvicorn app.main:app --reload

# 4. 入库文档（可重复调用，追加到多文档语料）
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d "{\"path\": \"./data/sample.md\"}"

# 5. 提问（P2：Agent 自动判断/改写，回答带 [1][2] 引用 + 忠实度校验）
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"什么是 RAG？\"}"

# 6. 查看已入库文档 / 按 source 删除一篇
curl localhost:8000/documents
curl -X POST localhost:8000/delete -H "Content-Type: application/json" -d "{\"source\": \"./data/sample.md\"}"
```

> 首次运行会联网下载 embedding / reranker 模型（BGE 系列）。向量库用 **Milvus Lite**（本地文件，无需独立服务）。

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
│   │   ├── nodes.py         #   retrieve / grade / rewrite / generate
│   │   ├── tools.py         #   工具封装
│   │   └── state.py         #   状态定义
│   ├── mcp/                 # P3：MCP server（待完成）
│   ├── memory/              # P3：记忆（待完成）
│   └── eval/                # P4：评估（待完成）
├── docs/
│   ├── architecture.md      # 架构设计 + 技术选型理由
│   └── evaluation.md        # 评估方法与指标
├── data/sample.md           # 示例文档
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
| 后端 | FastAPI | 异步、现代，Flask 可平滑迁移 |

详见 [docs/architecture.md](docs/architecture.md)。

## 路线图

- **P1**（已实现）基础 RAG 跑通
- **P2**（已实现）LangGraph Agent：检索 → 相关性判断 → query 改写重查 → 带引用生成
- **P3** 长期记忆 + MCP server
- **P4** 评估指标 + 纯 RAG vs RAG+Agent 对比实验 + Docker 一键部署

## 面试能讲的难点

1. 混合检索为什么比纯向量好？RRF 是怎么无参数融合两种排序的？
2. Agent 怎么决定「检索结果够不够，要不要改写 query 重查」？（grade 节点自省回路）
3. 引用如何对齐到原文 chunk，避免大模型幻觉？（chunk 级引用）
4. 怎么判断「回答是否忠实于上下文、是否编造」？（verify 节点忠实度校验，输出 supported/partial/unsupported）
