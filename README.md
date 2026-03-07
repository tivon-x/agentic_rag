# 🔍 Agentic RAG — 智能检索增强生成系统

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-blue.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-red.svg)

## 项目简介

这是一个基于 Agent 架构的 RAG（检索增强生成）系统，旨在解决传统 RAG 在处理复杂查询、语义偏差和召回率不足等问题。系统集成了混合检索技术与智能 Agent 编排，能够根据用户意图进行动态推理和知识聚合。

项目核心亮点包括：支持 BM25 与 FAISS 向量融合的混合检索管线，基于 LangGraph 的多步推理 Agent，以及支持实时流式输出的 Gradio 交互界面。系统采用工业级标准设计，具备完整的 Docker 支持、CI/CD 集成和集中化配置管理，非常适合作为企业级 AI 应用的参考架构。

## ✨ 功能特性

- 🔄 **混合检索**：结合 BM25 关键词匹配与 FAISS 向量语义检索，通过倒数秩融合（RRF）算法提升检索精准度。
- 🤖 **智能 Agent**：基于 LangGraph 构建多步推理图，支持查询分解、意图识别、长文本摘要和缺失信息澄清。
- 📊 **流式响应**：基于 Gradio 和 astream_events 实现低延迟流式输出，提供极致的用户交互体验。
- 💬 **会话管理**：内置多轮对话历史管理，支持独立会话线程，确保上下文连贯性。
- 📄 **PDF 处理**：采用 PDF → Markdown → 智能分块流程，支持递归分块、Token 分块及语义分块策略。
- ⚙️ **集中配置**：基于 frozen dataclass 实现的 AppSettings，统一管理系统参数，支持 .env 及环境变量无感覆盖。
- 🐳 **Docker 支持**：提供多阶段构建的 Dockerfile，采用非 root 用户运行，内置健康检查机制。
- 🧪 **测试覆盖**：包含完整的 pytest 单元测试套件，支持全流程离线模拟运行。
- 🔧 **CI/CD**：集成 GitHub Actions，自动执行代码风格检查（Ruff）和自动化测试。

## 🏗️ 系统架构

```mermaid
graph TD
    subgraph 索引阶段
        A[PDF 文档] --> B[数据处理<br/>PyPDF + 文本清洗]
        B --> C[智能分块<br/>Recursive / Token / Semantic]
        C --> D[FAISS 向量索引<br/>OpenAI Embeddings]
        C --> E[BM25 词频索引<br/>rank_bm25]
    end

    subgraph 检索阶段
        F[用户提问] --> G[查询分析<br/>LLM 结构化输出]
        G -->|清晰| H[查询分解<br/>多子问题拆分]
        G -->|模糊| I[请求澄清]
        H --> J[混合检索<br/>α·向量 + (1-α)·BM25]
        D --> J
        E --> J
    end

    subgraph 推理阶段
        J --> K[Agent 子图<br/>工具调用 + 中间件限制]
        K --> L[摘要聚合]
        L --> M[流式响应<br/>Gradio SSE]
    end
```

## 🚀 快速开始

**环境要求**: Python 3.12+, uv

### 方式一：本地运行

```bash
# 克隆项目
git clone <repo-url>
cd agentic_rag

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OpenAI 兼容的 API Key 等配置

# 索引文档
python main.py index path/to/pdfs

# 启动 UI
python main.py ui

# 命令行提问
python main.py ask "你的问题"
```

### 方式二：Docker 运行

```bash
# 构建镜像
docker compose build

# 启动服务
docker compose up -d

# 访问 http://localhost:7860
```

## 📁 项目结构

```
agentic_rag/
├── agent/                  # LangGraph Agent 模块
│   ├── graph.py           # 图组装：节点 + 边
│   ├── nodes.py           # 节点逻辑：摘要/重写/聚合
│   ├── edges.py           # 路由逻辑
│   ├── orchestrator_agent.py  # Agent 创建 + 中间件
│   ├── prompts.py         # 提示词模板（集中管理）
│   ├── tools.py           # 工具工厂（检索工具）
│   ├── schemas.py         # Pydantic 输出 Schema
│   └── graph_state.py     # 状态定义
├── indexing/              # 索引管线
│   ├── indexer.py         # 端到端索引流程
│   ├── data_processor.py  # PDF/文本加载 + 清洗
│   ├── chunker.py         # 分块策略（递归/Token/语义）
│   ├── embeddings.py      # OpenAI 兼容 Embeddings
│   ├── vectorstore.py     # FAISS 向量存储
│   ├── retriever.py       # BM25 + 向量融合检索
│   └── bm25_index.py      # BM25 索引构建
├── llm/                   # LLM 适配器
│   └── llm.py            # ChatOpenAI 封装
├── tests/                 # 单元测试
├── config.py              # Agent 运行时常量
├── factory.py             # 共享工厂函数
├── main.py                # CLI 入口
├── mappers.py             # 文件类型 & 分块器注册表
├── persistence.py         # BM25 持久化（pickle）
├── rag_answer.py          # 离线回答格式化
├── settings.py            # 集中配置 AppSettings
├── ui_gradio.py           # Gradio UI
├── Dockerfile             # 多阶段构建
├── docker-compose.yml     # 容器编排
└── pyproject.toml         # 项目元数据 & 依赖
```

## 🎯 设计决策

### 1. 混合检索 vs 纯向量检索
- **问题**：纯向量检索依赖语义相似度，在处理特定术语（如产品型号、人名）时召回率较低；而 BM25 对长文本的语义理解不足。
- **方案**：采用 BM25 + FAISS 向量融合检索，通过 `FUSION_ALPHA` 参数动态控制权重。
- **原因**：混合检索能同时兼顾精确匹配与语义理解。在工业界实践中，这种方案在真实场景下的鲁棒性远高于单一检索方式。

### 2. 查询分解 (Query Decomposition)
- **问题**：用户的提问往往包含多个隐含子问题，直接检索会导致信息覆盖不全。
- **方案**：在检索前引入 LLM 分析层，将复杂问题拆解为 2-3 个独立的子查询并行检索。
- **原因**：通过分解查询，可以显著提升每个子问题的检索精度（Precision），最终合并后的上下文包含更多有效信息。

### 3. Agent 中间件限制
- **问题**：LLM Agent 在推理过程中可能会由于循环依赖或幻觉进入无限工具调用。
- **方案**：在 LangGraph 编排层引入中间件机制，严格限制 `MAX_TOOL_CALLS=8` 和 `MAX_ITERATIONS=10`。
- **原因**：这些限制确保了系统的可预测性和 Token 成本的可控性，所有参数均可通过配置快速调整。

### 4. 集中配置管理
- **问题**：配置项分散在各模块硬编码会导致系统难以部署和维护。
- **方案**：使用 Python 的 `frozen dataclass` 定义 `AppSettings` 类，统一管理所有配置。
- **原因**：`frozen` 属性保证了运行时配置不可篡改，`dataclass` 提供了类型检查，支持从 `.env` 或系统环境变量无缝加载配置，适配现代云原生部署。

### 5. 离线模式 (Offline Mode)
- **问题**：在开发测试或无网络环境下，强依赖 API Key 会阻碍功能验证。
- **方案**：引入 `OFFLINE_MODE` 开关。开启时系统跳过 LLM 推理层，直接以 Markdown 格式输出检索结果的摘要。
- **原因**：这使得索引管线和检索逻辑可以在不产生 Token 费用的情况下进行 CI/CD 测试，提升了开发效率。

## ⚙️ 配置参考

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | LLM API 密钥 | (必填) |
| OPENAI_API_BASE | LLM API 地址 | (必填) |
| LLM_MODEL | 模型名称 | (必填) |
| LLM_TEMPERATURE | 生成温度 | 0.2 |
| EMBEDDING_MODEL | 嵌入模型 | text-embedding-3-small |
| EMBEDDING_API_KEY | 嵌入 API 密钥 | 同 OPENAI_API_KEY |
| EMBEDDING_API_BASE | 嵌入 API 地址 | 同 OPENAI_API_BASE |
| CHUNK_SIZE | 分块大小 | 512 |
| CHUNK_OVERLAP | 分块重叠 | 64 |
| CHUNKER_TYPE | 分块策略 | recursive |
| RETRIEVER_K | 检索结果数 | 10 |
| FUSION_ALPHA | 融合权重 | 0.5 |
| MAX_TOOL_CALLS | 最大工具调用 | 8 |
| MAX_ITERATIONS | 最大迭代次数 | 10 |
| MAX_CONTEXT_TOKENS | 最大上下文 Token | 5000 |
| LOG_LEVEL | 日志级别 | INFO |
| OFFLINE_MODE | 离线模式 | false |
| DATA_DIR | 数据目录 | data/ |
| FAISS_DIR | FAISS 索引目录 | data/index/faiss/ |
| BM25_PATH | BM25 索引路径 | data/index/bm25.pkl |

## 🔮 未来规划

- **多文档格式支持**：支持 Word、HTML 及 Markdown 等更多非结构化数据源。
- **中文分词优化**：集成 jieba 或 HanLP 分词器，提升 BM25 在中文语境下的匹配精度。
- **多用户隔离**：支持会话持久化与多用户权限管理。
- **检索可视化**：在 UI 界面增加来源高亮展示与检索相关度得分。
- **生产级数据库**：支持将本地 FAISS 迁移至 Milvus 或 Qdrant 等分布式向量数据库。

## 📝 开发指南

```bash
# 安装开发依赖
uv sync --dev

# 代码风格检查
uv run ruff check .

# 运行自动化测试
uv run pytest tests/ -v

# 重新构建 Docker 镜像
docker compose build --no-cache
```

## License

MIT License
