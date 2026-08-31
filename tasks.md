# Agentic RAG 项目优化任务文档

> **SUPERSEDED — 仅供历史参考，禁止据此判断当前里程碑。** 当前状态和执行边界以 `docs/research/v2_upgrade_plan.md` 为准。

## 目标：从平面 Chunk RAG 升级为分层化 Hierarchical Agentic RAG

## 1. 项目背景

当前项目是一个基于 LangGraph 的 Agentic RAG Demo，支持本地知识库构建、混合检索、任务路由和 UI 工作台。系统已具备以下能力：

* 索引阶段：文件加载、清洗、智能分块、FAISS 向量索引、BM25 索引、知识库画像；
* 问答阶段：`summarize_history -> decide_retrieval -> rewrite_query -> research_search -> aggregate_answers` 主链路；
* 路由策略：`retrieve / direct_answer / out_of_scope` 三态决策；
* 多模型分任务调用；
* 本地存储输出：`faiss/`、`bm25.pkl`、`corpus_profile.json`。

当前短板不在“有没有 RAG”，而在于：

1. **索引结构仍偏平面化**：主要围绕 chunk 建索引，没有显式保留 doc/section/paragraph/sentence 结构。
2. **检索链路偏短**：已有 query rewrite 和 research_search，但缺少显式的 retrieve → dedupe → rerank → truncate 分段治理。
3. **上下文构建较粗**：召回结果更像“若干 chunk 拼接”，而非“基于文档树重建上下文”。
4. **可解释性和评测体系不足**：缺少检索质量、答案质量、路由质量的系统化评估。
5. **存储层偏双轨割裂**：FAISS 与 BM25 分离，但没有一套统一的层级节点表示和节点元数据模型。

本次优化的核心目标是：
**将当前 Agentic RAG 升级为“分层索引 + 多阶段检索 + 结构化证据生成 + 可评测”的 production-oriented Hierarchical RAG。**

---

## 2. 总体改造方向

### 2.1 目标架构

将现有架构：

```text
Documents
-> loading/cleaning
-> chunking
-> FAISS + BM25
-> rewrite_query
-> research_search
-> aggregate_answers
```

升级为：

```text
Documents
-> parse into hierarchical payload
-> build document tree
-> embed leaf nodes
-> aggregate parent embeddings
-> store hierarchical nodes + lexical index + vector index
-> query planning
-> retrieve multi-granularity candidates
-> dedupe
-> rerank
-> context packing / truncate
-> grounded answer generation
-> citation-rich structured output
```

### 2.2 设计原则

1. **保留现有 LangGraph 主体框架**，不要推翻重写。
2. **先兼容、后替换**：先在现有 indexing / retrieval 流程上增量加入 hierarchical mode。
3. **优先提升可观测性**：每一步都应可输出调试信息和评测日志。
4. **优先增强索引表达能力，而不是盲目增加 agent 复杂度**。
5. **默认保持本地优先（local-first）**，但为后续切换 sqlite-vec / pgvector / Milvus 留接口。

---

## 3. 分阶段实施路线

建议拆成八个里程碑执行，而不是一次性大改。其中 Milestone 1–3 已完成。

### Milestone 1：建立层级化索引基础设施 ✅

目标：让系统从"chunk index"变成"document tree index"。

### Milestone 2：升级检索链路 ✅

目标：让 retrieval 具备 plan / retrieve / dedupe / rerank / truncate 的完整治理链。

### Milestone 3：升级答案生成 ✅

目标：让生成环节基于结构化证据，而不是简单拼接 chunk。

### Milestone 4：知识库画像升级 ✅

目标：将 corpus_profile 从静态描述升级为 retrieval prior 和 answer prior，在 routing、rewrite、rerank、answer style 中实际生效。

### Milestone 5：存储层改造 ✅

目标：抽象统一的检索存储接口，解耦业务逻辑与具体存储实现，为后续迁移 sqlite-vec / pgvector 做准备。

### Milestone 6：代码组织与检索管道可测试性提升 ✅

目标：将 `agent/nodes.py` 拆分为独立模块，补全 `FusionRetriever` 各检索阶段的单元测试，提升代码可维护性与可测试性。不涉及 LangGraph 图拓扑变更。

### Milestone 7：评测体系建设 ✅

目标：形成可持续迭代的实验闭环，能量化验证每阶段优化效果。

### Milestone 8：UI 与开发体验优化 ✅

目标：让 Gradio 工作台能可视化展示 hierarchical RAG 内部各阶段产物，而不只是最终答案。

---

# 4. 详细任务要求

## 4.1 索引层重构：从 chunking 升级到 hierarchical parsing

### 任务目标

在 `indexing/` 模块中新增一套层级化解析与索引流程，使文档被解析为树状结构，而不是只输出平面 chunk。

### 必做事项

#### 4.1.1 新增统一节点数据模型

新增核心数据结构，例如：

```python
@dataclass
class Node:
    node_id: str
    parent_id: str | None
    doc_id: str
    node_type: Literal["document", "section", "paragraph", "sentence", "chunk"]
    title: str | None
    text: str
    order: int
    level: int
    metadata: dict
    embedding: list[float] | None
    token_count: int | None
```

要求：

* 每个节点必须可追溯到 `doc_id`
* 必须保留 `parent_id`
* 必须保留顺序信息 `order`
* 必须支持元数据字段，如：

  * 文件名
  * 页码
  * 标题路径
  * section path
  * 来源类型（pdf/md/txt）
  * corpus_id

#### 4.1.2 实现 hierarchical parser

为不同文档类型实现统一解析接口：

```python
class HierarchicalParser(Protocol):
    def parse(self, file_path: str) -> ParsedDocumentTree: ...
```

最低要求：

* Markdown：依据标题层级、段落拆分
* TXT：依据空行、缩进、启发式标题识别
* PDF：先按页提取文本，再做标题/段落启发式切分

输出应至少包含：

```text
document -> section -> paragraph
```

如果 sentence 级拆分成本可控，可继续到 sentence；否则先以 paragraph 为叶子节点。

#### 4.1.3 叶子节点 embedding，父节点聚合

实现 hierarchical embedding 策略：

* 默认只对叶子节点（建议 paragraph）做 embedding
* section/document 节点 embedding 通过其子节点平均池化生成
* 允许未来切换为 weighted mean / attention pooling

建议提供配置：

```env
INDEX_MODE=flat|hierarchical
LEAF_NODE_TYPE=paragraph
PARENT_EMBED_POOLING=mean
```

#### 4.1.4 建立层级节点持久化

当前项目只明确输出 `faiss/`、`bm25.pkl` 和 `corpus_profile.json`。
需要新增统一节点持久化，例如：

```text
data/index/
  nodes.jsonl
  doc_trees.json
  faiss/
  bm25.pkl
  corpus_profile.json
```

其中：

* `nodes.jsonl`：所有节点扁平存储
* `doc_trees.json`：树结构索引
* 向量索引只索引可检索节点
* BM25 至少覆盖 paragraph 级文本

### 完成标准

* 能从输入文档构建出稳定的层级节点集合
* 节点可双向追踪：子 -> 父、父 -> 子
* 支持旧模式 flat 和新模式 hierarchical 共存
* CLI 构建索引时可通过参数切换

---

## 4.2 检索层升级：引入多阶段 retrieval pipeline

### 任务目标

将当前的 `rewrite_query -> research_search -> aggregate_answers` 检索链路，升级为显式可控的多阶段 retrieval pipeline。当前 README 已说明系统使用 `rewrite_query` 做自包含改写，并在 `research_search` 子图中完成检索。

### 必做事项

#### 4.2.1 新增 query planning 节点

在 `decide_retrieval` 之后、`rewrite_query` 之前或之后，新增：

```text
plan_query
```

职责：

* 判断问题类型：事实问答 / 总结 / 比较 / 多跳问题 / 定义型
* 生成 1~3 个子查询
* 指定偏好的检索粒度：

  * document-level
  * section-level
  * paragraph-level

输出示例：

```json
{
  "intent": "compare",
  "subqueries": [
    "LangGraph 中 decide_retrieval 的职责",
    "rewrite_query 在当前项目中的作用",
    "research_search 与 aggregate_answers 的协作方式"
  ],
  "preferred_node_types": ["section", "paragraph"]
}
```

#### 4.2.2 支持多粒度召回

检索器需要支持从不同粒度节点召回：

* 优先召回 paragraph
* 在总结型问题中允许先取 section，再向下展开 paragraph
* 在文档主题判断中允许使用 document embedding

建议设计统一接口：

```python
retrieve(query_plan, top_k, node_types=["paragraph", "section"])
```

#### 4.2.3 增加 dedupe 阶段

对检索结果做去重和冗余压缩：

* 相同文本去重
* 相邻重叠 paragraph 合并
* 同一 section 内高度相似结果合并
* 对同一父节点下的多个相邻叶子节点进行 window merge

输出中保留：

* 原始候选数
* 去重后候选数
* 合并日志

#### 4.2.4 增加 rerank 阶段

在混合召回后新增 rerank：

可选方案：

* Cross-encoder reranker
* LLM rerank（成本高，作为 fallback）
* 先规则 rerank，再模型 rerank

优先顺序建议：

1. lexical + vector fusion recall
2. metadata boost
3. reranker 重排

metadata boost 例子：

* 标题命中加权
* section title 命中加权
* corpus_profile boundary 命中加权

#### 4.2.5 增加 truncate / context packing 阶段

不是简单截断 top-k，而是做 **context packing**：

* 按 token budget 打包上下文
* 优先保留高分节点
* 若多个节点来自同一 section，优先保留结构连续片段
* 对 section 结果可自动补齐前后 paragraph 窗口

最终输出：

```python
PackedContext(
    passages=[...],
    total_tokens=...,
    dropped_candidates=...,
    packing_strategy="score_then_contiguity"
)
```

### 完成标准

* 检索图中出现显式阶段：
  `plan -> retrieve -> dedupe -> rerank -> truncate`
* 每一步都有中间产物可调试
* 相比原流程，召回结果冗余更少、上下文连续性更强

---

## 4.3 生成层升级：从“拼接答案”变成“基于证据树生成”

### 任务目标

让 `aggregate_answers` 不再只是聚合文本，而是基于结构化证据输出 grounded answer。

### 必做事项

#### 4.3.1 重写 aggregate_answers 的输入格式

从现在的“若干检索结果文本”升级为：

```json
{
  "question": "...",
  "query_plan": {...},
  "packed_context": [...],
  "evidence_groups": [
    {
      "doc_id": "...",
      "section_title": "...",
      "nodes": [...]
    }
  ]
}
```

推荐实现思路（结合当前项目现状）：

1. **不要再把 evidence 仅作为字符串留在 `agent_answers` 中**。当前 `search_relevant_chunks` 已使用 `response_format="content_and_artifact"`，应继续保留这一模式，但把 artifact 从“原始 Document 列表”升级为更完整的结构化证据对象。
2. **检索工具负责产出 evidence artifact，不负责偷偷写全局状态**。建议 `search_relevant_chunks` 返回：

```python
(
    "供 LLM 阅读的检索摘要文本",
    {
        "subquery": "...",
        "query_plan": {...},
        "packed_context": {...},
        "passages": [...],
        "debug": {...},
    },
)
```

3. **用 tool-call middleware 捕获 artifact，并写入 state**。推荐新增 `EvidenceCaptureMiddleware`，使用 LangChain 官方支持的 `@wrap_tool_call` / `AgentMiddleware.wrap_tool_call` 机制（https://docs.langchain.com/oss/python/langchain/middleware/custom），在 `handler(request)` 返回后拦截 `ToolMessage`，提取 `artifact`，再通过 `Command(update=...)` 把 evidence 写入 `ResearchSearchState`。
4. **状态层新增 evidence 字段并使用 reducer**。建议参考当前 `agent_answers` 的 reducer 模式，为主图或 research 子图新增：

   * `evidenceGroups`
   * `packedContexts`
   * `retrievalEvidence`

   这些字段应支持 append / reset，避免多次 tool call 或多子查询并发时相互覆盖。

建议补充 schema，例如：

```python
class EvidenceItem(BaseModel):
    doc_id: str
    node_id: str
    source: str
    section_path: list[str]
    page: int | None
    quote: str
    score: float | None
    relevance: str | None


class EvidenceGroup(BaseModel):
    subquery: str
    intent: str
    packed_context: dict
    evidence: list[EvidenceItem]
    debug: dict
```

目标不是让 `aggregate_answers` 继续“读若干 answer 文本做二次总结”，而是让它直接消费 retrieval pipeline 产出的结构化证据。

#### 4.3.2 输出结构化答案

要求模型输出至少包含：

```json
{
  "answer": "...",
  "reasoning_summary": "...",
  "evidence": [
    {
      "doc_id": "...",
      "node_id": "...",
      "quote": "...",
      "relevance": "..."
    }
  ],
  "confidence": 0.0,
  "limitations": "..."
}
```

如果不想强制 JSON，可先内部 JSON，外部再渲染 Markdown。

推荐实现要求：

1. 新增 `GroundedAnswer` / `GroundedAnswerPayload` Pydantic schema，统一约束生成结果。
2. `aggregate_answers` 不再直接 `llm.invoke()` 拼文本，而应优先使用 `with_structured_output(...)` 输出结构化答案。
3. 内部以 JSON / Pydantic 对象为标准结果，外部 UI / CLI 再将其渲染为 Markdown，避免把展示格式与推理数据结构绑死。
4. `confidence` 不要求一开始就做到严格校准，但至少应有可解释来源，例如：

   * evidence 命中数量
   * 证据之间是否一致
   * 是否存在明显空白或冲突

5. `limitations` 必须显式说明回答边界，不能用空泛套话代替。

建议示例：

```python
class GroundedAnswer(BaseModel):
    answer: str
    reasoning_summary: str
    evidence: list[EvidenceItem]
    confidence: float
    limitations: str
```

实现原则：

* **内部结构化，外部可渲染**；
* **先保证 grounded，再追求文风**；
* **不要把 citation 只做成末尾 Sources 文本列表**，而要保留可回溯的节点级信息。

#### 4.3.3 支持 citation rendering

UI 层输出时，允许展示：

* 文件名
* section path
* 页码/段落位置
* 可选展开原文片段

目标是让答案不是“像知道”，而是“能指回证据”。

推荐实现细节：

1. citation 渲染应直接基于 `GroundedAnswer.evidence`，而不是再次从 answer 文本中解析来源。
2. 每条 citation 至少应可追溯到：

   * `source` / 文件名
   * `doc_id`
   * `node_id`
   * `section_path`
   * `page` 或段落顺序

3. UI 上建议把“答案正文”和“证据列表”分开展示；证据列表支持展开查看 quote 原文。
4. 若多个证据来自同一 section，可在 UI 中按文档 / section 分组，避免 citation 面板过于碎片化。
5. debug 模式下可额外展示该 citation 对应的 retrieval 信息，如 rerank score、packing strategy、subquery 来源。

当前实现说明（2026-03-14）：

* 已在现有 Gradio UI 中落地基础版 citation rendering：答案正文与证据引用分区展示，citation 直接由 `GroundedAnswer.evidence` 渲染，并按 `source + section_path` 分组。
* 当前 Gradio 版本支持展示文件名、section path、page、doc_id、node_id 与 quote 原文。
* 更细粒度的交互式树定位（例如点击 citation 后联动展开文档树、逐条折叠/展开命中节点、富 debug drill-down）在 Gradio 中实现成本偏高，后续随 React UI 重构一并完成。

#### 4.3.4 增强 out_of_scope 的边界解释

既然项目已有 `corpus_profile.json` 驱动知识库边界提示，
则应进一步增强：

* 说明为什么越界
* 说明当前知识库覆盖什么
* 给出改写建议
* 给出用户下一步操作建议（上传哪些资料）

推荐实现说明：

1. 这一小节应与 4.4 的 corpus_profile 升级协同推进，但 **不必等 4.4 全部完成后再落地基础版本**。
2. 第一阶段可以先基于现有 `corpus_profile` 字段（`name / summary / coverage / usage_notes / source_examples`）增强 `out_of_scope_answer` prompt。
3. 第二阶段再接入扩展后的画像字段，例如：

   * `non_coverage`
   * `recommended_questions`
   * `forbidden_questions`
   * `domain_keywords`
   * `primary_entities`

4. 输出应尽量包含四个部分：

   * 为什么当前问题越界
   * 当前知识库主要覆盖什么
   * 用户可以怎样改写问题
   * 若确实想问该问题，建议补充哪些资料

5. out-of-scope 解释也应尽量结构化，至少内部保留 reason / boundary / suggestion / next_action 等字段，避免完全依赖 prompt 文风。

### 完成标准

* 聚合节点能接收结构化证据输入
* 答案输出具备 citation 和 confidence
* 超范围回答更可解释
* 证据采集主路径明确：`tool artifact -> wrap_tool_call middleware -> state -> grounded aggregation`
* `collect_answer` 如保留，只作为兜底，不再承担主证据建模职责

---

## 4.4 知识库画像升级：从静态描述到检索先验

### 任务目标

当前 `corpus_profile.json` 已用于路由边界判断。
需要把它进一步升级为 **retrieval prior** 和 **answer prior**。

### 必做事项

#### 4.4.1 扩展 corpus_profile 字段

新增：

```json
{
  "name": "...",
  "summary": "...",
  "coverage": "...",
  "non_coverage": "...",
  "recommended_questions": [...],
  "forbidden_questions": [...],
  "domain_keywords": [...],
  "preferred_answer_style": "...",
  "primary_entities": [...]
}
```

#### 4.4.2 在 query planning 中使用知识库画像

例如：

* 用 `domain_keywords` 扩展 query
* 用 `non_coverage` 限制越界问题
* 用 `primary_entities` 做实体归一化

#### 4.4.3 在 rerank 中引入画像加权

如果 passage 与画像中的核心实体或主题匹配，给予适度 boost。

### 完成标准

* corpus_profile 不再只是 UI 展示内容
* 在 routing、rewrite、rerank、answer style 中都被实际使用

---

## 4.5 存储层改造：为未来替换 SQLite / sqlite-vec / pgvector 做准备

### 任务目标

虽然当前项目使用 FAISS + BM25 已可运行，但需要抽象统一的检索存储接口，避免未来迁移困难。README 中当前索引输出仍以 `faiss/` 和 `bm25.pkl` 为主。

### 必做事项

#### 4.5.1 抽象统一 store 接口

设计：

```python
class NodeStore(Protocol):
    def save_nodes(...)
    def load_nodes(...)
    def get_node(node_id)
    def get_children(node_id)
    def get_parent(node_id)
```

```python
class VectorStore(Protocol):
    def add_embeddings(...)
    def search(...)
```

```python
class LexicalStore(Protocol):
    def build(...)
    def search(...)
```

#### 4.5.2 让 FAISS/BM25 成为实现，而不是系统假设

不要把业务逻辑写死在 FAISS/BM25 上；上层只依赖抽象。

#### 4.5.3 为 sqlite-vec 模式预留适配层

先不强制迁移，但要允许后续替换：

```env
VECTOR_BACKEND=faiss|sqlite_vec
```

### 完成标准

* 检索逻辑与具体存储解耦
* Node tree metadata 存储不依赖向量库本身

---

## 4.6 代码组织与检索管道可测试性提升

### 任务目标

提升代码可维护性与可测试性。当前 `agent/nodes.py`（375 行）将所有主图节点集中在一个文件中，随着功能增长需要拆分为独立模块。同时，`FusionRetriever` 内部的 `_retrieve_candidates / _dedupe_candidates / _rerank_candidates / _pack_context` 四个阶段虽然逻辑完整，但缺少独立单元测试覆盖。

### 背景说明：为什么不做 LangGraph 图拓扑重构

Milestone 2 已在 `FusionRetriever.retrieve()` 内部实现了完整的 `plan → retrieve → dedupe → rerank → pack` 管道，各阶段均有 debug 产物透传至 UI。将这些阶段提升为 LangGraph 图节点会：

1. 引入不必要的 state 序列化/反序列化开销 — 这些操作本质上是同步数据处理管道，不需要 LangGraph 的异步编排能力
2. 增加调试复杂度而不带来新功能
3. 当前主图仅 9 个节点，拓扑已足够清晰，不存在需要重构的混乱问题

因此本里程碑聚焦于**文件组织**和**测试覆盖**，不变更图拓扑。

### 必做事项

#### 4.6.1 拆分 `agent/nodes.py` 为 `agent/nodes/` 包

将当前 `agent/nodes.py` 中的各节点函数拆分为独立文件：

```text
agent/nodes/
├── __init__.py                  # re-export 所有节点函数，保持 graph.py 导入不变
├── inject_corpus_profile.py
├── summarize_history.py
├── decide_retrieval.py
├── plan_query.py
├── rewrite_query.py
├── direct_answer.py
├── out_of_scope_answer.py
└── aggregate_answers.py
```

要求：

* `agent/nodes/__init__.py` 必须 re-export 所有节点函数，使 `from agent.nodes import summarize_history` 等现有导入无需修改
* `agent/graph.py` 的导入路径不变
* 每个文件只包含一个节点函数及其直接依赖的私有辅助函数
* 拆分后运行全量测试，确保零回归

#### 4.6.2 补全 `FusionRetriever` 各检索阶段的单元测试

当前 `FusionRetriever` 的 `_retrieve_candidates`、`_dedupe_candidates`、`_rerank_candidates`、`_pack_context` 四个方法是检索链路的核心，需要独立单元测试覆盖。

在 `tests/` 下新增测试文件（建议 `tests/test_retrieval_pipeline.py`），至少覆盖：

**dedupe 阶段测试：**
* 相同文本去重：输入包含重复内容的候选，验证输出去重且保留最高分
* 相邻重叠合并：输入同一父节点下相邻 paragraph 候选，验证合并行为
* 空输入处理：输入空列表，验证不报错

**rerank 阶段测试：**
* title match boost：输入标题与 query 有重叠的候选，验证 boost 被正确应用
* corpus_profile boost：提供 corpus_profile，验证 domain_keywords / primary_entities / non_coverage 加权生效
* node_type_match boost：设置 preferred_node_types，验证匹配节点获得加分
* flashrank 降级：模拟 flashrank 不可用，验证 graceful fallback

**pack_context 阶段测试：**
* token budget 限制：验证打包结果不超出 token_budget
* 优先保留高分节点：验证高分候选优先进入 packed context
* 窗口扩展（expand_candidate）：验证 paragraph 候选在 summary intent 下可扩展为父 section

**集成测试：**
* 端到端 `FusionRetriever.retrieve()` 调用：给定 query 和 query_plan，验证返回 `PackedContext` 且 debug 字段完整

#### 4.6.3 确保现有节点函数的基础可测试性

对拆分后的各节点函数，验证以下条件成立（不要求写大量新测试，但需确认可测试性）：

* 每个节点函数签名为 `(state: GraphState) -> dict`，可直接传入 mock state 调用
* 不依赖全局可变状态（已满足，确认即可）
* 异常路径有 fallback 返回值（已满足，确认即可）

### 完成标准

* `agent/nodes.py` 已拆分为 `agent/nodes/` 包，所有现有导入保持兼容
* `FusionRetriever` 的 dedupe / rerank / pack_context 各阶段有独立单元测试
* 全量测试通过，无回归
* **不引入新的 LangGraph 图节点或子图**

---

## 4.7 评测体系建设：让优化可被验证

### 任务目标

为项目建立最小可用评测闭环，不然优化很容易停留在“感觉更高级”。

### 必做事项

#### 4.7.1 构建评测数据集

在 `tests/evals/` 或 `evals/` 下新增：

```text
evals/
  retrieval_cases.jsonl
  answer_cases.jsonl
  routing_cases.jsonl
```

每条样例至少包含：

* question
* expected_route
* gold_doc_ids / gold_node_ids
* reference_answer
* difficulty
* notes

#### 4.7.2 评测指标

至少实现：

**Routing**

* route accuracy

**Retrieval**

* recall@k
* MRR
* nDCG
* redundancy rate

**Answer**

* groundedness
* citation precision
* answer completeness
* hallucination rate（规则 + LLM judge）

#### 4.7.3 对比实验

至少比较三组：

1. baseline flat chunk RAG
2. flat + rerank
3. hierarchical RAG

#### 4.7.4 输出评测报告

CLI 示例：

```bash
python main.py eval --suite retrieval
python main.py eval --suite answer
```

输出 markdown/json 报告。

### 完成标准

* 能量化证明优化是否有效
* 能看出哪一阶段带来收益或退化

---

## 4.8 UI 与开发体验优化

### 任务目标

让 Gradio 工作台能展示 hierarchical RAG 的内部价值，而不只是最终答案。

### 必做事项

#### 4.8.1 知识库构建页增加索引模式选择

新增选项：

* Flat Chunk Mode
* Hierarchical Mode

并展示：

* 文档数
* section 数
* paragraph 数
* 叶子节点数
* 平均 tokens

当前实现说明（2026-03-14）：

* 已在 Gradio 知识库构建页增加 `Flat Chunk Mode / Hierarchical Mode` 切换。
* 已展示当前索引概览，包含可检测索引模式、文档数、section 数、paragraph 数、叶子节点数与平均 tokens。
* Flat 模式下由于当前索引产物不保留层级节点，section / paragraph / 叶子节点统计显示为 `N/A`。

#### 4.8.2 问答页增加调试面板

展示：

* route decision
* query plan
* rewritten queries
* retrieved candidates
* reranked top passages
* packed context
* citations

当前实现说明（2026-03-14）：

* 已在 Gradio 问答页增加调试面板，展示 `route decision`、`query plan`、`rewritten queries`、`retrieved candidates`、`reranked top passages` 与 `packed context`。
* citation 继续单独展示在“证据引用”面板，直接基于 `GroundedAnswer.evidence` 渲染。

#### 4.8.3 支持“查看命中的文档树位置”

例如：

```text
README.md
  > 检索路由说明
    > rewrite_query
```

当前实现说明（2026-03-14）：

* 已在 Gradio 中提供基础版“命中文档树位置”面板，按 `source -> section_path` 展示命中层级。
* 点击联动、树节点折叠展开、高亮回跳到原文位置等更强交互在 Gradio 中实现成本偏高，保留到未来 React UI 中完成。

### 完成标准

* 用户能观察系统为什么这样回答
* 有利于后续做 demo 和面试展示

---

# 5. 建议的代码目录调整

建议在现有结构上扩展，而不是大改目录。当前项目结构已包括 `agent/`、`core/`、`indexing/`、`llms/`、`ui/` 等模块。

建议新增或调整为：

```text
agentic_rag/
├── agent/
│   ├── graph.py
│   ├── states.py               # 注意：项目实际文件名为 states.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── inject_corpus_profile.py
│   │   ├── summarize_history.py
│   │   ├── decide_retrieval.py
│   │   ├── plan_query.py
│   │   ├── rewrite_query.py
│   │   ├── aggregate_answers.py
│   │   ├── direct_answer.py
│   │   └── out_of_scope_answer.py
├── indexing/
│   ├── parsers/
│   │   ├── base.py
│   │   ├── markdown_parser.py
│   │   ├── pdf_parser.py
│   │   └── txt_parser.py
│   ├── models/
│   │   ├── node.py
│   │   └── doc_tree.py
│   ├── builders/
│   │   ├── hierarchical_index_builder.py
│   │   └── flat_index_builder.py
│   ├── stores/
│   │   ├── node_store.py
│   │   ├── faiss_store.py
│   │   ├── bm25_store.py
│   │   └── sqlite_vec_store.py
│   ├── retrieval/
│   │   ├── planner.py
│   │   ├── fusion.py
│   │   ├── dedupe.py
│   │   ├── rerank.py
│   │   └── packer.py
├── evals/
│   ├── datasets/
│   │   ├── retrieval_cases.jsonl
│   │   ├── answer_cases.jsonl
│   │   └── routing_cases.jsonl
│   ├── metrics/
│   └── runner.py
```

---

# 6. 优先级排序

## 已完成（Milestone 1–3）

1. ✅ 建立 Node 数据模型与层级化解析
2. ✅ 建立 hierarchical indexing 模式
3. ✅ 检索链路加入 dedupe / rerank / pack_context
4. ✅ aggregate_answers 改为 grounded answer（含 citation rendering 基础版）

## P0：当前必须先做（Milestone 4–6）

1. corpus_profile 升级为 retrieval prior（扩展字段 + 接入 routing/rewrite/rerank）
2. 存储层抽象接口改造（NodeStore / VectorStore / LexicalStore Protocol）
3. `agent/nodes.py` 拆分为独立模块包 + `FusionRetriever` 各检索阶段补全单元测试

## P1：高优先级（Milestone 7–8）

1. 构建基础评测集与 baseline 对比实验
2. UI 调试面板（query plan / rerank / citations 可视化）
3. 评测指标实现（recall@k / MRR / nDCG / groundedness）

## P2：增强项

1. sqlite-vec 适配层实现
2. sentence-level leaf 支持
3. LLM-as-judge 评测
4. 自动窗口扩展与 section reconstruction

---

# 7. 交付物要求

以下为 **待完成** 交付物（Milestone 1–3 对应交付物已完成，不再列出）。

### 代码交付

* Milestone 4：corpus_profile 扩展字段 + 接入 routing / rewrite / rerank 的实际调用
* Milestone 5：NodeStore / VectorStore / LexicalStore Protocol 抽象层 + FAISS/BM25 适配实现
* Milestone 6：`agent/nodes.py` 拆分为 `agent/nodes/` 包 + `FusionRetriever` 各检索阶段单元测试
* Milestone 7：`evals/` 目录下评测数据集、指标实现与 runner 脚本
* Milestone 8：Gradio UI 索引模式选择 + 调试面板 + 文档树位置展示
* 所有新增模块有完整类型标注

### 文档交付

* 更新 README（反映 hierarchical 模式使用方式与新 CLI 参数）
* 新增 `docs/hierarchical_rag_design.md`（整体设计说明）
* 新增 `docs/eval_guide.md`（评测体系使用说明）

### 测试交付

* 各新增节点/模块的单元测试
* `FusionRetriever` 检索管道各阶段单元测试（dedupe / rerank / pack_context）+ 端到端集成测试
* 至少一组评测样例（retrieval_cases.jsonl / answer_cases.jsonl）
* 至少一份 baseline vs hierarchical 对比实验报告

### CLI / UI 交付

* `python main.py index ... --mode hierarchical`（已完成，确认可用）
* `python main.py ask ... --debug`（debug 参数输出 query plan / retrieval 中间产物）
* `python main.py eval --suite retrieval` / `--suite answer`
* UI 可查看 query plan / rerank / citations（调试面板）

---

# 8. 验收标准

项目优化完成后，应满足以下标准：

1. **索引阶段** ✅

   * 能构建 document/section/paragraph 层级节点
   * 向量与词法索引基于统一节点元数据

2. **检索阶段** ✅

   * 显式支持 `plan -> retrieve -> dedupe -> rerank -> truncate`
   * 检索结果冗余降低，上下文更连续

3. **生成阶段** ✅

   * 答案带证据引用（含基础版 citation rendering）
   * 能解释超范围判断原因

4. **工程阶段**（待完成）

   * 保持现有 LangGraph 架构风格与图拓扑不变
   * 保持 Gradio UI 可运行
   * 新旧模式兼容，不破坏 baseline
   * 存储层通过 Protocol 抽象解耦，FAISS/BM25 成为可替换实现
   * `agent/nodes.py` 拆分为 `agent/nodes/` 包，现有导入保持兼容
   * `FusionRetriever` 的 dedupe / rerank / pack_context 各阶段有独立单元测试

5. **画像与路由阶段**（待完成）

   * corpus_profile 扩展字段在 routing、rewrite、rerank、answer style 中实际生效
   * query planning 使用 domain_keywords / non_coverage / primary_entities

6. **评测阶段**（待完成）

   * 至少有一份 baseline vs hierarchical 的效果对比
   * 能量化验证 recall@k / MRR / groundedness 等核心指标
   * `python main.py eval` CLI 可运行并输出报告

7. **UI 可观测性阶段**（待完成）

   * 用户能在 Gradio 界面观察 query plan、rerank 结果、citations
   * 知识库构建页展示层级节点统计（文档数 / section 数 / paragraph 数）

---

# 9. 给编码智能体的执行建议

Milestone 1–3 已完成，执行从 Step 5 起继续推进：

```text
Step 1: 建 Node 模型与 parser                   ✅ 已完成
Step 2: 建 hierarchical index builder           ✅ 已完成
Step 3: 改 retrieval pipeline                   ✅ 已完成
Step 4: 改 aggregate answer schema              ✅ 已完成（含基础版 citation rendering）
Step 5: 升级 corpus_profile 字段并接入各链路    ← 当前起点
Step 6: 抽象存储层 Protocol，FAISS/BM25 适配
Step 7: 拆分 agent/nodes.py + 补全检索管道单元测试
Step 8: 构建评测集与运行基线对比实验
Step 9: 接 UI 调试面板与索引模式选择
Step 10: 补全 tests / evals，更新 README 与设计文档
```

执行原则：

* 每完成一步都运行测试
* 每一步都保留 fallback
* 不要先做 UI，再补底层
* 不要先上复杂 reranker，再补结构化索引
* 优先保证数据结构正确，再追求 fancy agent behavior

---

# 10. 一句话总结

这次优化的本质不是“再加几个 Agent 节点”，而是把你当前已经不错的 **Agentic RAG Demo**，升级成一个真正有工程内核的 **Hierarchical Agentic RAG 系统**：
**从平面 chunk 检索，升级为文档树检索；从简单召回，升级为多阶段检索治理；从文本拼接回答，升级为基于结构化证据的 grounded generation。**
