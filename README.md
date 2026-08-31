# Agentic RAG

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-blue.svg)
![BM25+FAISS](https://img.shields.io/badge/Retrieval-BM25%2BFAISS-2f855a.svg)
![Grounded Answer](https://img.shields.io/badge/Answer-Grounded%20with%20Citations-0f766e.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-red.svg)

基于 LangGraph 的 Hierarchical Agentic RAG 项目，支持本地知识库构建、多阶段检索、结构化证据答案、评测与可观测调试 UI。

## 项目背景

这个项目解决的不是“和单个 PDF 聊天”，而是如何把一组有边界的知识源升级成一个更接近生产形态的 Agentic RAG 系统。

核心改造方向是：从平面 chunk 检索升级到分层文档树检索，从简单召回升级到显式的 retrieval pipeline，从文本拼接回答升级到带结构化证据的 grounded answer。

## 核心能力

- 分层索引：支持 `flat` 与 `hierarchical` 两种模式；分层模式会构建 `document -> section -> paragraph` 文档树。
- 多阶段检索：查询规划、混合召回、去重、重排、context packing 全链路可观测。
- Grounded Answer：答案基于结构化证据生成，附带 citation、confidence 与 limitations。
- 知识库画像：`corpus_profile.json` 参与路由、改写、重排和回答风格控制。
- 评测体系：内置 routing / retrieval / answer 三类评测，可对比 `baseline_flat`、`flat_rerank`、`hierarchical`。
- Gradio 工作台：可视化索引模式、调试面板、证据引用和命中文档树位置。

## 架构亮点

- 为什么使用 LangGraph：路由、检索、聚合、越界处理都有明确节点职责，比简单链式调用更适合表达多分支决策和可观测状态流转。
- Hierarchical Index：文档被解析为 `Node` 树并保留 `doc_id / parent_id / order / metadata`；叶子节点做 embedding，父节点使用聚合策略生成向量。
- FusionRetriever：检索流程固定为 `plan -> retrieve -> dedupe -> rerank -> pack_context`，每个阶段都有中间产物，可直接进入 UI 调试面板。
- 存储抽象层：通过 `NodeStore`、`VectorStore`、`LexicalStore` Protocol 解耦业务与存储实现，当前默认实现是 JSON Node Store + FAISS + BM25。
- 结构化证据答案：最终输出不是“若干 chunk 拼接”，而是带 citation 的 grounded answer，便于回溯命中的文档位置和证据片段。

## 评测结果

首份离线基线报告见 [基线评测报告](docs/eval_baseline_report.md)。

- `flat_rerank` 相比 `baseline_flat`，`MRR` 从 `0.85` 提升到 `1.0`，`nDCG` 从 `0.8681` 提升到 `1.0`。
- `citation_precision` 从 `0.2833` 提升到 `0.4333`，说明重排后证据命中质量更稳定。
- 当前离线基线里 `hierarchical` 仍弱于 `flat_rerank`，因此它在这个仓库中不是“已经最优”，而是下一轮重点优化对象。

### M6 KITE AI Papers 非正式诊断

当前仅完成 M6A 数据与可复现准备。以下 B0 至 B3 结果来自固定 KITE snapshot（15 个问题、134 个 PDF）的本机诊断运行，复用了产品 fixed answer path；四份报告均为 `formal_run=false`，且工作区 patch 不一致，因此分数不是正式冻结基线、生产决策或已交付 M6 的证明，也不与 KITE README 的官方绝对分数直接比较：

| Pipeline | 平均分 | 有效题数 | p95 latency | 平均 context tokens |
|---|---:|---:|---:|---:|
| B0 | 3.8667 | 15/15 | 402.2 s | 16474.3 |
| B1（当前默认） | 6.1333 | 15/15 | 166.2 s | 16599.6 |
| B2 | 6.8667 | 15/15 | 170.0 s | 16935.5 |
| B3 | 6.5333 | 15/15 | 177.8 s | 17810.4 |

B2 仅作为非正式诊断候选保留，B3 因逐题 3 个 loss 且分数提升不足 0.5 未通过诊断门槛；M6B 至 M6D 尚未通过正式验收。生产默认仍为 `v1_flat_rerank`，不会由评测自动切换。逐题报告、失败案例和证据审计见 [KITE Benchmark 诊断报告](docs/kite_benchmark_report.md)、[生产 Pipeline 诊断记录](docs/production_pipeline_decision.md) 与 Web 的 [只读评测页](/evaluation)。

## 项目结构

```text
agentic_rag/
├── agent/
│   ├── graph.py    # LangGraph 主图装配
│   ├── nodes/      # 拆分后的节点包：summarize / route / plan / rewrite / aggregate ...
│   ├── prompts.py
│   └── states.py
├── core/           # settings、factory、corpus_profile、grounded answer 渲染
├── indexing/
│   ├── parsers/    # markdown / txt / pdf 分层解析
│   ├── builders/   # flat / hierarchical index builder
│   ├── stores/     # NodeStore / VectorStore / LexicalStore 抽象与实现
│   └── retriever.py
├── evals/          # 评测数据、指标与 runner
├── llms/           # 按任务类型路由模型
├── ui/             # Gradio UI
├── tests/          # pytest 测试
└── main.py         # CLI 入口
```

## 快速开始

建议使用 `uv`：

```bash
uv sync --extra dev
cp .env.example .env
```

构建索引：

```bash
uv run python main.py index path/to/knowledge --mode hierarchical
```

CLI 提问：

```bash
uv run python main.py ask "你的问题"
```

调试说明：

- 当前可观测性主要通过 Gradio UI 的调试面板提供，可查看 `route decision`、`query plan`、`retrieved candidates`、`reranked passages`、`packed context` 和 citations。
- `main.py ask` 当前版本尚未提供 `--debug` 参数，因此 README 不把它写成已支持能力。

启动 UI：

```bash
uv run python main.py ui
```

运行评测：

```bash
uv run python main.py eval --suite retrieval --offline
uv run python main.py eval --suite answer
```

## 优化后的检索与回答链路

```text
summarize_history
-> decide_retrieval
-> plan_query
-> rewrite_query
-> retrieve
-> dedupe
-> rerank
-> pack_context
-> aggregate_answers
```

输出不再是简单拼接 chunk，而是基于结构化 evidence 的 grounded answer。

## 工程化设计

- `agent/nodes/` 已替代单文件 `agent/nodes.py`，节点按职责拆分，便于维护和测试。
- `indexing/stores/` 提供可替换的存储接口，避免业务逻辑直接耦合到 FAISS / BM25。
- `tests/test_retrieval_pipeline.py` 覆盖 dedupe、rerank、pack_context 及端到端检索流程。

## 主要产物

索引默认写入 `data/index/`：

- `faiss/`
- `bm25.pkl`
- `nodes.jsonl`
- `doc_trees.json`
- `corpus_profile.json`

评测报告默认写入 `data/eval_reports/`。

## 常用命令

```bash
uv run ruff check .
uv run pytest -v
uv run pytest tests/test_retrieval_pipeline.py -v
```

## 相关文档

- [分层 RAG 设计任务](tasks.md)
- [评测指南](docs/eval_guide.md)
- [基线评测报告](docs/eval_baseline_report.md)

## License

MIT
