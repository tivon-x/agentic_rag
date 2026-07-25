# Agentic RAG V2 升级方案

> 修订日期：2026-07-25  
> 状态：待用户审批，只允许先执行 Core M1 至 M3  
> 推荐主线：论文证据底座 + metadata-prefixed hybrid retrieval + 有界补检 + 双模式科研工作区

## 1. 决策摘要

V2 不再按一次性重写推进。完整路线分为三层：

| 层级 | 范围 | 进入条件 | 是否属于首次审批 |
|---|---|---|---|
| Core | 可靠索引、论文目录、页码证据、Search、固定 V2 检索、精简评测 | 本方案获批 | 是 |
| Enhanced | 持久 run worker、有界补检、claim 校验、调试 trace | Core 检索达到 M3 门槛，用户再次确认 | 否 |
| Product | 多论文比较、研究结果保存、备份、部署收口 | Enhanced 的质量与延迟门槛成立，用户再次确认 | 否 |

首次实施边界只有 M1、M2、M3。它们完成后，产品已经能稳定导入论文、搜索个人论文库、查看原文页码、运行固定 RAG，并用可复现评测证明检索变化。M4 至 M6 的技术方向已经确定，但不会与 Core 捆绑实施。

V2 的四个核心亮点是：

1. 论文、章节、passage、页码和原文引用使用稳定标识，答案能回到证据页。
2. 检索表示与引用原文分离，通过 metadata prefix、BM25、dense、RRF、重排和受控扩展提升召回，同时保持引用不被改写。
3. 固定检索是正式基线，有界 adaptive 只在证据不足时进入，最多补检两轮。
4. 评测按 parser、retrieval、answer、route 分库，报告成对差异、坏例和成本，不再用一个小数据集支撑所有结论。

Docling、GraphRAG、多 Agent、RAPTOR、GROBID 服务、专用向量数据库和云任务队列不进入 V2。Docling 当前适合结构和版面解析，但官方仍把标题、作者、参考文献和语言元数据列为后续能力，且现有仓库尚无足够 parser gold 证明其收益值得承担依赖与运行成本。V2 Core 使用仓库已安装的 PyMuPDF4LLM，保留 legacy parser 回退。

关键决策依据可直接追溯到 [Docling 2.114.0 官方包说明](https://pypi.org/project/docling/2.114.0/)、[Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)、[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[LangGraph fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)、[AsyncSqliteSaver API](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/aio/AsyncSqliteSaver)、[JsonPlusSerializer API](https://reference.langchain.com/python/langgraph/checkpoint/serde/jsonplus/JsonPlusSerializer) 和 [ALCE 引用评测](https://aclanthology.org/2023.emnlp-main.398/)。完整来源与适用边界见 `phase1_research_report.md` 第 5 至 8 节。

## 2. 产品目标和非目标

### 2.1 目标用户和核心问题

用户是维护个人已读论文库的单用户。核心问题不是“让模型回答论文问题”，而是完成以下闭环：

1. 导入论文并知道解析、索引是否成功。
2. 按标题、作者、年份、章节、术语和自然语言检索。
3. 打开答案引用的论文页，核对原文。
4. 理解单篇论文，比较多篇论文的方法与实验。
5. 保存答案、证据和比较结果，之后能够重看。
6. 在技术调试模式中检查召回、重排、证据筛选和成本。

普通 flat RAG 不够的原因是论文问题经常包含章节语义、缩写、方法条件、表格结果和跨论文关系。仅按固定长度切块会丢失论文与章节上下文；仅输出答案文本也无法证明结论来自哪一页。

### 2.2 V2 目标

- local-first、单用户、本机文件和 SQLite，不依赖外部数据库或任务服务。
- 数字版 PDF 为主要输入，Markdown 和 TXT 继续兼容。
- 每条主要结论绑定可核验的 `evidence_id`、论文、章节、页码和原文。
- 导入、重试、索引版本切换、运行恢复具备明确状态和幂等边界。
- 固定与 adaptive 共用同一检索器、证据 schema 和评测集。
- 前端同时支持日常使用和面试调试，不复制 Zotero、Overleaf 或全网学术搜索。

### 2.3 非目标

- 不做多人协作、权限、云同步、浏览器插件和移动端。
- 不做全文 PDF 编辑、批注同步、参考文献管理器和论文写作工具。
- 不承诺扫描件、复杂跨页表格和公式语义与数字版 PDF 同等准确。
- 不构建引用关系图，不引入 GraphRAG 或图数据库。
- 不让 Agent 自由创建工具、无限循环或自动形成长期用户画像。
- 不把检索评测分数表述为生成答案质量。

### 2.4 前提失效处理

方案假设个人库以带文本层的数字版论文为主。M2 的 12 篇 parser gold 中，如果超过 3 篇无法满足页码正确率和文本完整性门槛，则停止结构化增强，不进入 M3。产品保留 legacy page-level 搜索和手工元数据编辑，不以增加 Docling、OCR 或外部服务来掩盖前提失效。

## 3. 范围和运行架构

### 3.1 三层范围

**Core**

- 修复配置路径、测试环境污染、并发索引和错误 SSE。
- 持久索引任务、单 worker、启动恢复和 immutable index version。
- 论文目录、元数据编辑、PyMuPDF4LLM parser adapter、legacy fallback。
- Search、固定 V2 检索、页码证据和 final-only answer。
- parser 与 retrieval 精简评测。

**Enhanced**

- 持久 run queue、LangGraph checkpoint、进程重启恢复。
- fixed、adaptive、refuse 三类路由。
- 最多两轮补检、claim-evidence 校验和最终拒答。
- 持久 trace 和技术调试面板。

**Product**

- 多论文比较、研究工作区、artifact 保存。
- 备份与恢复、Docker 和演示脚本。
- 不新增新的检索技术。

### 3.2 组件关系

```text
Next.js
  | REST + SSE
FastAPI
  |-- catalog / sessions / jobs / runs / evidence  --> data/app.db
  |-- index worker                                --> data/indexes/<version>/
  |-- run worker                                  --> data/checkpoints.db
  |-- parser adapter                              --> uploads + parsed artifacts
  `-- LangGraph                                   --> fixed/adaptive evidence graph

Index version
  |-- manifest.json
  |-- passages.jsonl
  |-- bm25.pkl
  |-- faiss/
  `-- integrity.json
```

FastAPI 进程内最多启动一个 index worker 和一个 run worker。SQLite 使用 WAL。两个 worker 都通过数据库 lease 领取任务，不依赖 `asyncio.create_task` 保存任务生命期。Next.js 不直接访问文件系统或索引。

### 3.3 数据流

导入：

1. `POST /api/index/files` 流式保存文件，计算 SHA-256，写 `indexing_jobs` 和 `index_job_items`。
2. index worker 领取 lease，解析到项目自有 `ParsedPaper` schema。
3. 元数据归一化并写 catalog；parser 输出落为可检查 artifact。
4. 为 passage 生成 `quote_text` 与 `retrieval_text`，并在调用 embedding provider 前执行应用侧长度校验。
5. 在临时版本目录构建 FAISS、BM25 和 manifest；manifest 固化 embedding provider、model、dimension、input mode 和代码版本。
6. 完整性校验通过后，用一次数据库事务切换 `active_index_version`。
7. 失败版本保留错误摘要，不影响当前 active version。

查询：

1. Core 直接运行 fixed pipeline，得到候选、重排结果和 evidence。
2. 证据不足时，Core 返回有限答案或拒答，不做隐式补检。
3. Enhanced 获批后，run worker 执行 adaptive graph。
4. 所有 claim 校验完成后才产生 `answer.final`。
5. 前端用 `paper_id + page_number` 打开原 PDF，并在侧栏显示 quote。

## 4. 数据模型和稳定标识

### 4.1 SQLite 迁移

继续使用原生 `aiosqlite`，不引入 SQLAlchemy 或 Alembic。新增 `api/db/migrations.py`，按整数版本在事务中执行迁移，并用 `schema_migrations(version, applied_at)` 记录。每个迁移只前进，不自动降级。回滚通过代码开关、active index pointer 和数据库备份完成。

Core 新增或改造：

| 表 | 关键字段 | 约束 |
|---|---|---|
| `papers` | `id`, `content_hash`, `title`, `authors_json`, `year`, `venue`, `doi`, `metadata_status`, `archived_at` | `content_hash` 唯一 |
| `paper_versions` | `id`, `paper_id`, `parser_name`, `parser_version`, `source_path`, `parsed_artifact_path`, `quality_json` | 每次重解析新建版本 |
| `sections` | `id`, `paper_version_id`, `parent_id`, `title`, `level`, `ordinal`, `page_start`, `page_end` | 顺序稳定 |
| `passages` | `id`, `paper_version_id`, `section_id`, `page_start`, `page_end`, `quote_text`, `retrieval_text`, `ordinal` | 原文和检索表示分开 |
| `index_versions` | `id`, `status`, `manifest_path`, `created_at`, `activated_at` | 同时只有一个 active |
| `indexing_jobs` | 现有字段加 `request_json`, `attempt_count`, `lease_owner`, `lease_expires_at`, `progress_json`, `active_version_before` | 状态机约束 |
| `index_job_items` | `job_id`, `paper_id`, `source_path`, `status`, `error_code`, `error_detail` | `(job_id, source_path)` 唯一 |
| `idempotency_records` | `scope`, `key`, `request_hash`, `response_json`, `expires_at` | `(scope, key)` 唯一 |
| `app_state` | `key`, `value_json`, `updated_at` | 保存 active index pointer |

Enhanced 再新增：

| 表 | 关键字段 | 作用 |
|---|---|---|
| `runs` | `id`, `session_id`, `status`, `input_json`, `history_snapshot_json`, `strategy`, `index_version`, `attempt_count`, `lease_owner`, `lease_expires_at`, `result_json`, `error_code` | 持久运行和恢复 |
| `run_events` | `run_id`, `seq`, `event_type`, `payload_json`, `created_at` | SSE 重连和 trace |
| `retrieval_candidates` | `run_id`, `round`, `passage_id`, `channel`, `rank`, `score_json` | 详细召回记录 |
| `evidence_items` | `run_id`, `evidence_id`, `passage_id`, `quote_text`, `score_json`, `accepted` | 证据账本 |
| `artifacts` | `id`, `session_id`, `type`, `title`, `content_json`, `created_at` | Product 保存结果 |

`chat_sessions.messages` 的 JSON 数组在 Core 保持兼容。M4 新建 `chat_messages(id, session_id, run_id, role, content_json, ordinal)` 并一次性迁移，之后停止写旧 JSON 字段。

### 4.2 标识规则

- `paper_id = sha256(raw_file_bytes)`，完全相同的文件重复上传返回现有论文。
- `paper_version_id = sha256(paper_id + parser_name + parser_version + normalization_version)`。
- `section_id = sha256(paper_version_id + normalized_heading_path + ordinal)`。
- `passage_id = sha256(paper_version_id + section_id + page_start + ordinal + sha256(quote_text))`。
- `evidence_id = "E-" + passage_id[:12]`，展示稳定，数据库仍保存完整 `passage_id`。
- `index_version` 是不可变 UUID，manifest 记录 parser、embedding provider/model/dimension/input mode/context-length check/max input chars、BM25 tokenizer、chunker、reranker 和代码版本。
- `run_id` 是 UUID。LangGraph `thread_id` 与 `run_id` 相同，但会话历史来自数据库快照，不由 checkpoint 承担。

Core 中的 `paper` 表示一个上传文件实体，`paper_versions` 表示该文件经过不同 parser 或 normalization 版本得到的解析版本，不表示论文内容修订版。同一论文的 arXiv 新版本、出版社版本或重新下载后字节不同的 PDF 会生成不同 `paper_id`，Core 不自动合并。DOI 和 arXiv ID 只作为元数据保存，后续允许用户手工合并文件实体。

只要 parser 或 normalization 版本变化，`paper_version_id` 和 passage ID 都会变化。旧 artifact 仍引用旧版本证据，UI 标记“证据来自历史索引”，不把它静默指向新 passage。

### 4.3 幂等边界

- `POST /api/index/files` 必须带 `Idempotency-Key`。相同 key 和相同 request hash 返回原 job；相同 key 和不同 request hash 返回 409。
- Enhanced 的 `POST /api/runs` 使用相同规则。
- `PATCH /api/papers/{paper_id}` 使用版本号 `If-Match` 做乐观并发控制。
- job retry 使用状态比较更新，仅允许 `failed -> queued`，重复 retry 返回当前 job。
- 最终 assistant message 以 `run_id` 唯一，恢复执行不会重复追加。

## 5. Parser、元数据和证据定位

### 5.1 仓库实测与选择

2026-07-25 对仓库 6 篇 PDF 做只读对照，包含双栏论文、表格、公式、80 页教程和 200 页书稿：

| 项目 | Legacy PyPDFLoader | PyMuPDF4LLM |
|---|---:|---:|
| 6 篇总耗时 | 11.96 秒 | 50.07 秒 |
| 6 篇总文本字符 | 864,302 | 892,635 |
| 单篇相对耗时 | 基线 | 约 3 至 14 倍 |
| 表格 Markdown | 无 | 抽查 3 篇分别得到 16、3、77 行 |
| 章节标题 | page-level，无语义章节 | 每篇只识别 1 至 3 个 Markdown heading，仍需归一化 |

21 篇现有评测 PDF 中，只有 6 篇有非空标题元数据，且其中 2 篇明显是 arXiv 标记或构建文件名；作者元数据更稀疏。因此 parser 选择不能代替元数据流程。

V2 Core 决定：

- 默认 parser：`PyMuPDF4LLMPaperParser`。
- 回退 parser：现有 `PdfHierarchicalParser`，重命名为 `LegacyPaperParser`。
- 统一输出：项目自己的 `ParsedPaper`、`ParsedSection`、`ParsedBlock`，业务代码不依赖第三方类型。
- 单文件超时：数字论文 180 秒，超过 100 页的长文档 600 秒。
- OCR：Core 不启用。无文本层页面标记 `needs_ocr`，仍可保留 PDF 和手工元数据。
- Docling：V2 不采用。其结构、表格、公式和本地运行能力保留为 V2 之外的备选，但元数据能力、依赖、模型下载和仓库实测不足以支持 V2 默认化。

### 5.2 结构归一化

`indexing/parsers/paper_parser.py` 定义 parser protocol。`indexing/parsers/structure_normalizer.py` 只做可复现规则：

1. 优先读取 Markdown heading。
2. 识别 `1 Introduction`、`2.3 Training`、`Appendix A` 等编号标题。
3. 排除 `# of parameters`、短公式、页眉和页脚。
4. 没有可靠标题时创建 `Page N` section，不能伪造语义章节。
5. 表格作为独立 block，保存 Markdown、caption、page number。
6. 公式保存可抽取文本和 page number，不宣称公式语义正确。
7. passage 不跨越可靠 section 边界；表格和 caption 不拆开。

自动质量检查：

- 成功抽取页数不少于 PDF 页数的 95%。
- 非空文本页比例不少于 90%，扫描件除外。
- PyMuPDF4LLM 总字符数不得低于 legacy 的 60%。
- 页码必须单调且落在 PDF 范围内。
- 任一检查失败则保存诊断并使用 legacy 结果，状态为 `degraded`。

### 5.3 元数据提取和用户校正

元数据不依赖 Docling，也不默认调用 LLM。优先级固定如下：

| 字段 | 提取顺序 | 无可靠值时 |
|---|---|---|
| title | 有效 XMP/PDF Info -> 首页面积最大的标题候选 -> 文件名 | `needs_review` |
| authors | 有效 XMP/PDF Info -> 首屏标题下方、摘要上方的候选行 | 空值，`needs_review` |
| year | XMP 日期 -> 前两页版权/会议年份规则 -> 文件名 | 空值 |
| venue | 前两页会议或期刊规则 | 空值 |
| DOI | 前两页 DOI 正则 | 空值 |

系统保存每个字段的 `source` 和 `confidence`。规则会拒绝 `untitled`、`Microsoft Word`、`arXiv:...`、`BookVersion.dvi` 等通用或构建标题。未验证元数据可以参与检索，但 UI 显示“待校正”。用户在论文详情页通过 `PATCH /api/papers/{paper_id}` 修改并标记 `metadata_status=verified`。元数据改变只重建 `retrieval_text` 和索引版本，不改变 `quote_text`。

### 5.4 Parser gold

固定 12 篇文档、48 个页面：

- 双栏正文 3 篇。
- 表格密集 3 篇，其中包含 1 个跨页表格测试 fixture。
- 公式密集 3 篇。
- 低文本或扫描 fixture 3 篇。

人工只标四项：阅读顺序、章节边界、表格边界、页码。标题、作者、年份和 DOI 单独标字段正确性。总标注和复核预算为 6 至 8 小时，由项目作者完成一次盲化复核，不称为双标注。

M2 门槛：

- 数字版页面 page number accuracy = 100%。
- 正文字符召回中位数不低于 legacy。
- 章节边界 F1 不低于 0.80。
- 表格边界 F1 不低于 0.75。
- 元数据字段不得用错误自动值覆盖空值；title accuracy 不低于 0.90。
- parser p95 不超过同文档 legacy 的 15 倍，且没有不可终止任务。

## 6. 固定检索主线

### 6.1 Metadata-prefixed retrieval

本方案不把以下做法称为 Anthropic Contextual Retrieval。Anthropic 的方法会为每个 chunk 生成一段 chunk-specific explanatory context；V2 不调用 LLM 生成这类内容。V2 的名称固定为 **metadata-prefixed retrieval**：

```text
[TITLE] <title>
[AUTHORS] <authors>
[YEAR] <year>
[SECTION] <heading path>
[BLOCK] <paragraph|table|formula>
<quote_text>
```

- `retrieval_text` 用于 BM25、embedding 和 rerank。
- `quote_text` 保持 parser 抽取的原文，只用于上下文和引用。
- 当前 OpenAI-compatible embedding provider 使用 raw-string input mode，即 `check_embedding_ctx_length=false`。因此 passage 与 metadata prefix 组合后的长度由应用侧负责校验，超限时确定性重切分或明确失败，不能直接发送超长输入。
- 无法确认的 metadata 留空，不让低置信值污染所有 passage。
- prefix 每个字段有开关，可以独立消融。

### 6.2 固定 pipeline

1. scope filter：按论文、标签、年份和用户选择范围过滤。
2. query normalization：保留英文缩写、连字符和数字；中文用 jieba，英文用小写词项规则。
3. BM25 top 40。
4. dense top 40。
5. 以 `passage_id` 去重，使用 RRF `k=60` 融合，不混合原始分数。
6. FlashRank 重排 top 30，取 top 8。
7. 对命中 passage 加同 section 的前后各 1 个邻居，去重后最多 12 个。
8. context pack 按 evidence 分数和 token budget 排序，保留论文、章节和页码。
9. 生成答案并输出 claim 级 evidence IDs。

Core 固定预算：

- retrieval 候选最多 80 个。
- rerank 最多 30 个。
- final evidence 最多 8 个，扩展后 context 最多 12 个。
- context 默认 8,000 tokens。
- Core 不改写查询，不做第二轮检索。

### 6.3 多粒度边界

Core 不使用 mean-pooled parent embedding 作为主召回。dense 与 BM25 都索引可引用 passage，section 只用于 metadata prefix、scope 和邻居扩展。当前 `hierarchical_index_builder.py` 保留兼容，但不作为 V2 推荐路径。

RAPTOR、multi-vector 和 GraphRAG 均不进入 V2。只有当固定测试集中的“跨章节综合”子集持续失败，且 passage expansion 无法改善时，才在 V2 之后重新评估摘要层。

### 6.4 缓存和索引版本

- cache key：`index_version + retriever_config_hash + normalized_query + scope_hash`。
- 只缓存 BM25/dense/RRF/rerank 的 passage IDs 和分数，TTL 30 分钟，进程内 LRU 256 项。
- active index 切换后旧 key 自然失效，不做全局清空。
- metadata、tokenizer、embedding 或 chunker 变化都生成新 index version。
- 每个版本构建到临时目录，校验后原子重命名并更新数据库 pointer。

## 7. Enhanced Agent 设计

### 7.1 策略和停止条件

M4 才启用三类策略：

| 策略 | 适用输入 | 行为 |
|---|---|---|
| `direct` | 寒暄、确认、对已有答案的格式调整，不产生新的论文事实 | 不检索，不允许新增事实 claim |
| `fixed` | 明确事实、定义、单篇定位 | 只运行一次固定 pipeline |
| `adaptive` | 多论文比较、跨章节综合、第一次证据覆盖不足 | 最多两轮检索 |
| `refuse` | 不在论文库、要求外部实时事实、证据无法支持 | 解释范围并拒答 |

adaptive 预算：

- query plan 最多 4 个子问题。
- 第一轮最多 4 次检索，可并发执行只读检索。
- 第二轮最多 2 次补检。
- 总 tool calls 不超过 6。
- 总 rerank passage 不超过 120。
- 总 evidence 不超过 12。
- 总上下文不超过 12,000 tokens。
- 任一轮 evidence IDs 与上一轮完全相同则停止。
- 覆盖分不再改善、预算用尽、用户取消或模型错误均停止。

### 7.2 Compact GraphState

`agent/states.py` 不再继承 `MessagesState` 保存完整会话和证据对象。V2 GraphState 只保留控制信息和小型结果：

| 字段 | 类型 | 上限 |
|---|---|---|
| `runId` | `str` | 1 |
| `sessionId` | `str` | 1 |
| `query` | `str` | 4,000 字符 |
| `historySummary` | `str` | 2,000 字符 |
| `scopeIds` | `list[str]` | 100 |
| `strategy` | `str` | 1 |
| `planItems` | `list[dict]` | 4 个，只含 id、query、status |
| `round` | `int` | 0 至 2 |
| `candidateIds` | `list[str]` | 30 |
| `evidenceIds` | `list[str]` | 12 |
| `coverage` | `dict[str, float]` | 每个 plan item 1 个值 |
| `budgets` | `dict[str, int]` | 固定字段 |
| `terminationReason` | `str` | 1 |
| `finalAnswer` | `dict` | 最终 answer、claims、citations |

完整候选分数、证据 quote 和 trace 分别存入 `retrieval_candidates`、`evidence_items`、`run_events`。节点按 ID 从 repository 加载。所有节点写入使用 `(run_id, round, item_id)` 唯一键 upsert，恢复执行不会生成重复记录。

checkpoint 显式构造：

```python
serde = JsonPlusSerializer(pickle_fallback=False)
checkpointer = AsyncSqliteSaver(connection, serde=serde)
```

生产环境同时设置 `LANGGRAPH_STRICT_MSGPACK=true`。GraphState 只允许 JSON 基础类型，不把 Pydantic model、Document、LLM client、完整消息或数据库连接放进状态。`InMemorySaver` 只用于单元测试。

### 7.3 Session memory 与 run recovery

本方案明确选择 `thread_id=run_id`：

- session history 存在 `chat_messages`。
- 创建 run 时，在事务中写入不可变 `history_snapshot_json`。
- graph 的 `thread_id` 只标识一次 run 的 checkpoint，不承担跨 run 会话记忆。
- 同一 session 同时只允许一个 `queued` 或 `running` run，数据库部分唯一索引保证；第二个请求返回 409。

`AsyncSqliteSaver` 只保存图 checkpoint，不会自动重新调度任务。恢复由 run worker 完成：

1. FastAPI 启动时生成 `worker_id`。
2. worker 每秒查询 `queued` 或 lease 已过期的 `running` run。
3. `BEGIN IMMEDIATE` 领取任务，写 30 秒 lease；每 10 秒 heartbeat。
4. 首次执行用 initial state 和 `thread_id=run_id` 调用 graph。
5. 恢复执行先检查同一 thread 是否已有 checkpoint；有则用相同 config 和空输入继续，没有则使用保存的 initial state 重启。
6. 最大尝试 2 次。第二次仍失败则写 `failed`，保留错误码和最近事件。
7. 只有 final answer、evidence 和 assistant message 在一个事务中写成功后，run 才变为 `completed`。
8. 成功 run 的 checkpoint 24 小时后调用 saver 的 thread delete API 清理；失败 run 保留 7 天。artifact 依赖的是数据库 evidence，不依赖 checkpoint。

run 的所有外部操作都是只读检索或幂等数据库 upsert。用户取消把状态改为 `cancel_requested`；节点在每个边界检查后以 `cancelled` 结束。

### 7.4 Claim 校验和输出

1. 生成结构化 claims，每个 claim 声明 evidence IDs。
2. 校验 evidence 是否存在、是否属于当前 index version、quote 是否支持 claim。
3. 不支持的 claim 触发一次定向补检，已经在第二轮则删除、降级措辞或拒答。
4. 引用完整性只检查主要事实 claim，不要求寒暄和过渡句引用。
5. 最终响应包含 answer、claims、citations、limitations 和 termination reason。

不向用户流式发送 provisional answer。SSE 只发送进度、已确认 evidence 和最终一次性答案：

```text
run.queued
run.started
plan.ready
retrieval.completed
evidence.accepted
validation.completed
answer.final
run.failed | run.cancelled
```

`answer.final` 必须在 claim 校验后产生。客户端断线后使用 `Last-Event-ID` 从 `run_events.seq` 继续，不会看到后续被替换的答案。

## 8. API 和前端

### 8.1 Core API

保留 `/api` 前缀：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/index/files` | 上传并创建持久 job，要求 `Idempotency-Key` |
| `GET` | `/api/indexing-jobs/{job_id}` | 状态、进度、失败原因 |
| `POST` | `/api/indexing-jobs/{job_id}/retry` | 失败任务重试 |
| `GET` | `/api/papers` | 分页、筛选、排序 |
| `GET` | `/api/papers/{paper_id}` | 元数据、解析质量、索引状态 |
| `PATCH` | `/api/papers/{paper_id}` | 校正元数据，要求 `If-Match` |
| `GET` | `/api/papers/{paper_id}/file` | 支持 Range 的 PDF 响应 |
| `GET` | `/api/search` | 关键词或自然语言检索，返回 evidence |
| `POST` | `/api/chat` | 兼容现有 fixed chat，只返回已完成答案 |
| `GET` | `/api/chat/{session_id}` | 会话历史 |
| `GET` | `/api/index-versions` | active 和历史版本 |

Core 删除现有“转发所有模型 token”的行为。`/api/chat/stream` 在 M1 兼容保留，但只发 `progress` 和一个 `answer.final`，M4 再迁移到 run API。没有后端持久化的 Settings 页面不进入 V2。

### 8.2 Enhanced API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/runs` | 创建 fixed/adaptive run |
| `GET` | `/api/runs/{run_id}` | 当前状态与最终结果 |
| `GET` | `/api/runs/{run_id}/events` | SSE，支持 `Last-Event-ID` |
| `POST` | `/api/runs/{run_id}/cancel` | 请求取消 |
| `GET` | `/api/runs/{run_id}/debug` | plan、候选、重排、证据、预算和耗时 |

### 8.3 前端交付层次

Core 只做 4 个用户表面，不建设 8 个空页面：

1. `/library`：论文列表、元数据状态、索引状态、重试。
2. `/papers/[id]`：PDF、元数据编辑、解析降级提示。
3. `/search`：全库搜索、scope filter、结果证据和页码跳转。
4. `/chat`：固定 RAG、会话历史、答案引用和页码跳转。

PDF 阅读器 Core 使用浏览器原生 PDF 或嵌入页，定位格式为 `file_url#page=N`。证据侧栏显示 quote、章节和页码。Core 不承诺 bbox 高亮；这比错误高亮更可信。

Enhanced 在 `/chat` 加普通模式和技术调试模式：

- 普通模式只显示进度、最终答案和 evidence cards。
- 调试模式显示 strategy、query plan、BM25/dense rank、RRF、rerank、证据接纳、补检原因、停止原因、耗时和 token。

Product 增加：

1. `/compare`：选择 2 至 5 篇论文和 1 至 6 个比较维度，每个单元格绑定 evidence。
2. `/workspace`：保存 answer、comparison 和证据集合。

不建设独立 Settings 页面。配置继续由 `.env` 和启动诊断管理，避免出现无后端语义的外壳页面。

## 9. 配置、依赖和本地运行

### 9.1 配置

所有设置进入 `core/settings.py`，不在路由读取环境变量。新增：

```text
APP_DB_PATH=data/app.db
CHECKPOINT_DB_PATH=data/checkpoints.db
INDEX_ROOT=data/indexes
UPLOAD_ROOT=data/uploads
PAPER_PARSER=pymupdf4llm
PARSER_TIMEOUT_SECONDS=180
LONG_DOCUMENT_TIMEOUT_SECONDS=600
INDEX_WORKER_LEASE_SECONDS=60
RUN_WORKER_LEASE_SECONDS=30
EMBEDDING_INPUT_MODE=raw
EMBEDDING_MAX_INPUT_CHARS=6000
RETRIEVAL_PIPELINE=v2_fixed
BM25_TOKENIZER=mixed
RRF_K=60
RERANK_TOP_N=30
FINAL_EVIDENCE_LIMIT=8
CONTEXT_TOKEN_BUDGET=8000
LANGGRAPH_STRICT_MSGPACK=true
```

Core 不需要新增 API Key。dense embedding 和生成仍使用现有 OpenAI-compatible 配置；无 Key 时 Search 的 BM25 和已存在的离线检索模式仍可运行。`EMBEDDING_INPUT_MODE=raw` 映射为 `check_embedding_ctx_length=false`；`EMBEDDING_MAX_INPUT_CHARS=6000` 是 provider 调用前的确定性应用侧上限。后续若切换 input mode 或长度上限，必须生成新 index version。API Key 不得写入 manifest。

### 9.2 依赖

Core：

- 继续使用已在 `pyproject.toml` 中的 `pymupdf4llm`、`pymupdf`、`aiosqlite`、`jieba`、FAISS、rank-bm25 和 FlashRank。
- 不引入 Docling、Alembic、SQLAlchemy、Redis、Celery、PostgreSQL 或前端 PDF SDK。

Enhanced：

- 新增并锁定 `langgraph-checkpoint-sqlite`，与当前 LangGraph 版本做最小导入和恢复测试。
- 官方不建议把 Async SQLite saver 用于通用高并发生产负载；本项目只在单用户、单 run worker、compact state 和独立 checkpoint DB 的边界内采用。
- 不新增任务服务。

Product：

- Docker 只打包现有 FastAPI 与 Next.js，SQLite、索引和上传目录通过 volume 持久化。

实施时锁文件解析失败视为里程碑失败，不允许临时替换核心技术。

## 10. 评测设计

### 10.1 数据集分开维护

| 数据集 | Dev | Test | 用途 | 人工预算 |
|---|---:|---:|---|---:|
| parser gold | 4 篇 | 12 篇、48 页 | 结构、页码、表格、元数据 | 6 至 8 小时 |
| retrieval | 16 问 | 48 问 | passage 召回与排序 | 6 至 8 小时 |
| answer | 8 问 | 24 问 | claim 支持、引用、完整性 | 8 至 10 小时 |
| route/refusal | 16 问 | 48 问 | fixed/adaptive/refuse 分类 | 2 至 3 小时 |

Core 只要求 parser gold、retrieval 和 8 条 answer smoke。Enhanced 才补齐 answer 与 route/refusal。四个数据集不做统一 70/15/15 切分。

Retrieval test 固定 4 个子集，每类 12 条：

- 精确术语与定义。
- 方法与章节定位。
- 实验数值与表格。
- 跨论文或跨章节问题。

Route test 固定 4 类，每类 12 条：direct、fixed、adaptive、refuse。类别均衡只用于策略评测，不代表真实流量。

作者完成首次标注。两周后随机隐藏系统名，复核 20% retrieval 和 answer 样本并记录修改率。这是单人一致性检查，不表述为独立双标注。

### 10.2 对照组

| 组 | Pipeline | 目的 |
|---|---|---|
| B0 | dense + BM25，无 rerank | 最低复杂度混合基线 |
| B1 | 当前 `flat_rerank` | 当前最强基线 |
| B2 | metadata prefix + mixed BM25 + dense + RRF + rerank | Core 推荐固定链路 |
| B3 | B2 + section neighbor expansion | 扩展净收益 |
| B4 | B3 + adaptive | Enhanced 净收益 |

所有组使用同一 parser 产物、同一 embedding provider/model/dimension/input mode、同一 reranker、同一 top-k 和同一测试集。B0 至 B3 必须从同一冻结配置重建索引，不得复用历史 fake embedding 索引或其他模型生成的 FAISS 索引。查询配置与 index manifest 不兼容时立即失败。对 B2 做以下单因素消融：

- 去掉 metadata prefix。
- 去掉 sparse。
- 去掉 dense。
- RRF 换回当前 min-max。
- 去掉 rerank。
- 去掉 neighbor expansion。

Adaptive 消融：

- 固定 B3。
- B3 + routing，不补检。
- B3 + 一轮补检。
- B3 + 两轮补检和 claim validation。

### 10.3 指标和报告方法

Retrieval：

- passage Recall@5、Recall@10、MRR@10、nDCG@10。
- paper Recall@10、section Recall@10。
- 每个问题记录 B1/B2/B3 的成对胜、平、负。
- paired bootstrap 95% 区间只作为不确定性描述，不以“提升 5 至 8 个百分点”作为硬承诺。

Answer：

- claim support precision。
- citation correctness。
- citation completeness。
- requirement coverage。
- unsupported major claim count。
- answer/refusal utility 人工 0、1、2 分。

Agent：

- route macro F1 和每类 confusion matrix。
- successful termination rate。
- 平均检索轮数、tool calls、重复检索率。
- p50、p95 latency、LLM input/output tokens。

### 10.4 发布门槛

M3 在 48 条 retrieval test 上必须同时满足：

- B2 的 Recall@10 不低于 B1。
- B2 相对 B1 至少 8 条提升 gold rank，退化不超过 4 条。
- 四个子集没有任何一个出现 Recall@10 下降 2 条以上。
- B2 p95 检索延迟不超过 B1 的 1.5 倍。
- B3 只有在跨章节子集至少改善 3 条、其他子集总退化不超过 1 条时才默认启用。

M4 必须满足：

- route test 每类 recall 不低于 0.75，macro F1 不低于 0.80。
- adaptive 在 24 条 answer test 中至少改善 5 条 requirement coverage，退化不超过 2 条。
- citation correctness 不低于 fixed B3。
- adaptive p95 总延迟不超过 fixed 的 2.5 倍，平均检索轮数不超过 1.5。
- 重启恢复测试不重复 assistant message、evidence 或 tool side effect。

未达到门槛时，默认策略保持 B1 或 B2 fixed，失败增强通过独立开关关闭。

## 11. 实施里程碑

完整路线预计修改或新增 35 至 45 个文件，超过 8 个文件，影响 `core/`、`indexing/`、`agent/`、`evals/`、`api/`、`web/` 和 `tests/`。每个里程碑控制在一个可审查主题内，完成后项目必须可运行。

### M1：运行与索引可靠性

**目标**

修复环境、SSE、持久 job 和并发索引问题，不改变检索算法。

**主要文件**

- `core/settings.py`
- `main.py`
- `api/main.py`
- `api/db/database.py`
- `api/db/models.py`
- 新增 `api/db/migrations.py`
- `api/routers/indexing.py`
- `api/models/indexing.py`
- `api/routers/chat.py`
- `api/models/chat.py`
- `api/services/graph_cache.py`
- `indexing/indexer.py`
- 对应 `tests/test_api.py` 和新增 worker/index version 测试

影响约 12 至 15 个文件，因为数据库、任务状态、索引写入和 API 生命周期必须一起闭环。

**改造**

- 合并 `main.py` 和 FastAPI 的 settings 加载边界，测试不再污染进程环境。
- 建立迁移表和 Core schema。
- 用单 index worker、lease、heartbeat 和启动扫描替换 `_BACKGROUND_TASKS`。
- index 写入临时版本，校验后切 active pointer。
- job 状态固定为 `queued -> running -> completed|failed|cancelled`。
- `/api/chat/stream` 只发进度和一次最终答案，不再转发所有 model token。
- 为上传 API 加 `Idempotency-Key`。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_api.py tests/test_settings.py tests/test_index_job_recovery.py tests/test_index_version.py tests/test_streaming.py -q
uv run --extra dev ruff check api core indexing main.py tests
```

**人工检查**

1. 上传两篇论文并在 running 时终止 API。
2. 重启后只恢复一个 worker，任务继续或明确失败。
3. active index 在新版本校验前不变化。
4. 重复 Idempotency-Key 不生成第二个 job。
5. SSE 中不存在路由或规划模型 token。

**回滚**

保留旧索引读取适配器和 `INDEX_WRITE_MODE=legacy` 一个版本周期。active pointer 可切回上一 ready version。数据库升级前自动复制 `sessions.db`；代码回滚不删除新增表。

### M2：论文目录与页码证据

**目标**

建立 paper、section、passage 和 metadata 闭环，交付 library、paper detail、Search 和页码跳转。

**主要文件**

- `indexing/parsers/base.py`
- `indexing/parsers/pdf_parser.py`
- 新增 `indexing/parsers/paper_parser.py`
- 新增 `indexing/parsers/pymupdf4llm_parser.py`
- 新增 `indexing/parsers/structure_normalizer.py`
- `indexing/models/node.py`
- `indexing/models/doc_tree.py`
- `indexing/builders/hierarchical_index_builder.py`
- `indexing/indexer.py`
- `api/db/database.py`
- `api/db/models.py`
- 新增 `api/routers/papers.py`
- 新增 `api/routers/search.py`
- `web/src/app/kb/page.tsx`，迁移为 `/library`
- 新增 `web/src/app/papers/[id]/page.tsx`
- 新增 `web/src/app/search/page.tsx`
- parser、metadata、paper API 测试

影响约 16 至 20 个文件。修改跨 parser、catalog、API 和三个产品页面，不能拆成只写后端却没有用户价值的半成品。

**改造**

- 加 parser protocol、PyMuPDF4LLM 默认实现和 legacy fallback。
- 落地结构归一化、质量检查和解析 artifact。
- 落地元数据优先级、来源、置信度和用户修正。
- 生成稳定 paper/section/passage IDs。
- Search 返回 paper、section、page、quote 和 score stage。
- PDF 用 Range 响应和 `#page=N` 定位。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_pdf_parser.py tests/test_parser_quality.py tests/test_metadata.py tests/test_paper_api.py tests/test_search_api.py -q
uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
uv run --extra dev ruff check indexing api tests
pnpm --dir web lint
pnpm --dir web build
```

**人工检查**

1. 双栏、表格、公式、长文和低文本 PDF 各导入一篇。
2. 错误 PDF title 不覆盖首屏正确标题，未知作者保持空值。
3. 修改 title 后新索引使用新 prefix，quote 不变化。
4. 从 Search 点击 evidence 打开正确论文页。
5. parser 降级状态和原因在 library 可见。

**回滚**

`PAPER_PARSER=legacy` 只影响新任务。M2 catalog 和 parsed artifact 不删除，上一 active index 继续可读。M2 parser 门槛失败则停止进入 M3。

### M3：固定 V2 检索和精简评测

**目标**

交付 metadata-prefixed retrieval、mixed tokenizer、RRF、rerank、可选邻居扩展和可复现检索报告。

**主要文件**

- `indexing/bm25_index.py`
- `indexing/retriever.py`
- `indexing/retrieval_pipeline.py`
- `indexing/stores/lexical_store.py`
- `indexing/stores/vector_store.py`
- `core/rag_answer.py`
- `core/settings.py`
- `evals/runner.py`
- `evals/metrics.py`
- `evals/datasets/`
- 新增 eval 配置和报告生成器
- `api/routers/search.py`
- `web/src/app/search/page.tsx`
- retrieval 与消融测试

影响约 12 至 16 个文件。所有检索阶段通过 config registry 组装，避免为每个实验复制 pipeline。

**改造**

- `retrieval_text` 与 `quote_text` 分离。
- 固化 embedding provider/model/dimension/input mode，并让 runner 校验 index manifest。
- 中英 mixed tokenizer。
- RRF 替换默认 min-max fusion。
- B0 至 B3 共用 runner。
- 建 48 条 retrieval test 和 8 条 answer smoke。
- 输出逐问题差异、目标子集、延迟和 bad cases。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_bm25_index.py tests/test_retriever.py tests/test_retrieval_pipeline.py tests/test_evals.py -q
uv run python -m evals.runner --config evals/configs/v2_b1.yaml
uv run python -m evals.runner --config evals/configs/v2_b2.yaml
uv run python -m evals.runner --config evals/configs/v2_b3.yaml
uv run python -m evals.build_report --runs artifacts/evals/v2_core
uv run --extra dev ruff check indexing core evals tests
```

**人工检查**

1. 检查至少 10 个 B1/B2 rank 变化案例。
2. 对表格、缩写、跨章节和中文术语各看 3 个 bad case。
3. 确认引用显示 quote_text，不显示 metadata prefix。
4. 默认 pipeline 只在 M3 门槛通过后从 B1 切到 B2 或 B3。

**回滚**

`RETRIEVAL_PIPELINE=v1_flat_rerank|v2_fixed|v2_expanded` 独立切换。每个 index version 记录 tokenizer 和 retrieval schema，不混用不兼容版本。

### M4：持久 run 与有界 adaptive

**进入条件**

M3 发布门槛全部通过，用户批准 Enhanced。

**主要文件**

- `agent/graph.py`
- `agent/states.py`
- `agent/edges.py`
- `agent/nodes/` 下新增或重构 plan、retrieve、score、validate、finalize 节点
- `agent/prompts.py`
- `agent/schemas.py`
- `agent/tools.py`
- 新增 `api/services/run_worker.py`
- 新增 `api/services/run_repository.py`
- 新增 `api/routers/runs.py`
- 新增 `api/models/runs.py`
- `api/main.py`
- `api/db/migrations.py`
- `web/src/app/chat/page.tsx`
- graph、budget、serializer、recovery、SSE 测试

影响约 18 至 22 个文件。它是 Enhanced 的单独架构变更，不与 Core 同批实施。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_agent_graph.py tests/test_agent_budget.py tests/test_claim_validation.py tests/test_run_recovery.py tests/test_run_streaming.py -q
uv run python -m evals.runner --config evals/configs/v2_adaptive.yaml
uv run --extra dev ruff check agent api tests
```

**必须通过的故障注入**

- 在第一轮检索后终止进程并重启。
- 在 answer 写入前终止进程并重启。
- 让 lease 过期并启动第二 worker 竞争。
- SSE 断线后用 `Last-Event-ID` 连接。
- 同一 session 并发创建两个 run。

结果必须是：最多一个 worker 持有 lease；run 从 checkpoint 恢复或以保存输入重启；assistant message、evidence 和事件不重复；并发第二 run 返回 409。

**回滚**

`ANSWER_STRATEGY=fixed` 完全绕过 adaptive 节点。Core fixed chat 保留一个版本周期。checkpoint 数据库可删除而不影响已完成 artifact。

### M5：调试 trace 和评测补齐

**进入条件**

M4 恢复测试通过。

**改造**

- 补齐 24 条 answer test 和 48 条 route/refusal test。
- `run_events`、candidate 和 evidence 形成可导出的 trace。
- chat 技术模式显示计划、排名、补检、停止原因、延迟和 token。
- trace 默认保存 7 天；用户保存的 artifact 不受影响。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_trace_repository.py tests/test_debug_api.py tests/test_route_eval.py -q
uv run python -m evals.runner --config evals/configs/v2_route.yaml
uv run python -m evals.runner --config evals/configs/v2_answer.yaml
pnpm --dir web lint
pnpm --dir web build
```

**回滚**

`DEBUG_TRACE=false` 停止写详细 candidate 和 trace，只保留 run 结果与错误。Agent 行为不依赖调试页面。

### M6：比较、Workspace 和部署

**进入条件**

M4 质量、延迟和成本门槛通过，用户批准 Product。

**改造**

- Compare 固定 2 至 5 篇论文、1 至 6 个维度。
- 每个比较单元格必须包含 evidence，缺证据显示“未找到”，不补写推测。
- 保存 answer、comparison 和 evidence collection。
- 提供 SQLite、uploads、indexes、parsed artifacts 的一致性备份命令。
- Docker 启动 FastAPI 和 Next.js，数据目录使用 volume。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_artifacts_api.py tests/test_compare.py tests/test_backup_restore.py -q
pnpm --dir web lint
pnpm --dir web build
docker compose up --build
```

**人工检查**

用 3 篇同主题论文生成方法、数据集、指标、局限四维比较，逐格打开 evidence。备份后在空数据目录恢复，论文、索引、会话和 artifact 数量一致。

**回滚**

Compare 和 Workspace 是独立路由及表，不影响 Search、fixed chat 或 adaptive graph。Docker 失败不阻止本机 `uv run uvicorn` 与 `npm run dev`。

## 12. 延迟、成本、失败恢复和保留

### 12.1 预算

| 路径 | 目标 |
|---|---|
| Search p95 | 本地 warm index 小于 1.5 秒 |
| Fixed 检索 p95 | 不超过 B1 的 1.5 倍 |
| Fixed 首个进度事件 | 小于 500 ms |
| Fixed 完整答案 p95 | 记录真实值，不设脱离模型的绝对承诺 |
| Adaptive p95 | 不超过 fixed 的 2.5 倍 |
| Parser | 单文档 180 秒，长文 600 秒硬超时 |

每个 run 记录模型、token、tool calls、各节点耗时和 index version。没有模型定价配置时只报告 token，不猜测货币成本。

### 12.2 失败恢复

| 失败 | 处理 | 用户可见结果 |
|---|---|---|
| 上传中断 | 临时文件未进入 job item | 可重新上传 |
| parser 失败 | legacy fallback 或 `needs_ocr` | 明确降级原因 |
| 索引 worker 崩溃 | lease 到期后启动扫描恢复 | job 回到 running |
| 新索引校验失败 | version 标 failed，active 不变 | 旧库继续可用 |
| LLM 失败 | Core 返回错误；Enhanced 最多恢复一次 | 不写空答案 |
| run worker 崩溃 | 同 run checkpoint 恢复 | 不重复消息与证据 |
| SSE 断线 | 从持久事件序号继续 | 不重跑 run |
| SQLite 损坏 | 从最近备份恢复 | 报告最后备份时间 |

### 12.3 保留

- 成功 checkpoint：24 小时。
- 失败 checkpoint 和 trace：7 天。
- 未保存 run events：7 天。
- inactive index version：保留最近 2 个。
- 上传原 PDF 和已保存 artifact：不自动删除。
- 清理命令先输出将删除的对象，要求显式确认。

## 13. 采用、暂缓和拒绝

### 13.1 采用

- PyMuPDF4LLM + deterministic normalizer + legacy fallback。
- metadata-prefixed retrieval、mixed BM25、dense、RRF、FlashRank。
- passage 检索、section 邻居扩展和稳定 evidence IDs。
- SQLite catalog、job/run lease、immutable index。
- compact GraphState、显式安全 serializer、run worker 恢复。
- final-only validated answer streaming。
- Core、Enhanced、Product 分层交付。

### 13.2 暂缓到 Enhanced 或 Product

- adaptive 补检和 claim validation。
- 详细 trace。
- Compare、Workspace、备份与 Docker。

暂缓项的接口、状态和门槛已在本方案确定，后续不需要重新选择技术，只需要用户确认是否扩大实施范围。

### 13.3 V2 拒绝

| 方案 | 原因 | 当前替代 |
|---|---|---|
| Docling 默认 parser | 元数据仍是后续能力，依赖重，仓库无收益证据 | PyMuPDF4LLM + legacy |
| Anthropic 式 LLM chunk context | 重建成本、可重复性和 token 成本过高 | 可消融 metadata prefix |
| RAPTOR | 聚类和摘要成本高，证据条件可能丢失 | section 邻居扩展 |
| GraphRAG | 个人库尚无图问题证据 | passage/section scope |
| 多 Agent | 增加协调、失败和评测面 | 单 graph 有界节点 |
| GROBID 服务 | Java/容器维护超出 local-first 首版 | parser adapter |
| PostgreSQL、Redis、Celery | 单用户规模没有必要 | SQLite lease worker |
| 专用向量数据库 | FAISS 规模足够，迁移无净收益 | immutable FAISS |
| provisional token stream | 校验后可能推翻已展示内容 | progress + answer.final |

## 14. 简历和演示口径

### 14.1 只能在实测后填写的简历表达

> 面向个人论文库设计 local-first 科研助手，构建可定位到论文与页码的稳定证据索引；通过 metadata-prefixed BM25+dense、RRF 与重排，在 48 条固定检索测试上将 passage Recall@10 从【B1 实测】提升到【B2 实测】，同时报告成对坏例和 p95 延迟。

> 在 LangGraph 中实现 fixed/adaptive 双策略、持久 run lease 和最多两轮补检，以 claim-evidence 校验控制引用和拒答；在独立 answer 与 route 数据集上报告引用正确性、完整性、route macro F1、恢复成功率和 token 成本。

方括号只能替换为真实结果。未完成 Enhanced 时不写第二条。

### 14.2 面试演示

Core 演示：

1. 上传一篇元数据错误的 PDF，展示任务状态和手工校正。
2. 搜索一个术语和一个表格数值，打开原 PDF 页。
3. 展示 B1 与 B2 的候选排名和 metadata prefix 消融。
4. 说明 `retrieval_text` 与 `quote_text` 为什么分开。

Enhanced 演示：

1. 提出跨论文比较问题。
2. 展示 fixed 证据不足后进入第二轮补检。
3. 查看 claim、evidence、停止原因和预算。
4. 终止并重启 API，展示同一 run 恢复且答案不重复。

## 15. 实施授权边界

本文件获批后，Codex 只执行 M1，再按 M1 验收结果进入 M2、M3。不得提前安装 Docling，不得新增外部服务，不得开始 M4 至 M6。每个里程碑开始前记录分支和工作区，保护用户已有未提交文件；每个里程碑结束后单独提交验收结果和坏例，等待用户决定是否继续。

## 16. 评审问题关闭矩阵

| 评审问题 | 修订结果 | 对应章节 |
|---|---|---|
| checkpoint 与 session memory、任务恢复混淆 | 明确 `thread_id=run_id`、历史快照、lease worker、启动扫描和两次尝试 | 4.2、7.3 |
| Docling 过早默认 | V2 拒绝 Docling；增加 legacy 与 PyMuPDF4LLM 实测、元数据链和用户校正 | 5、13 |
| GraphState 与 checkpoint 膨胀 | 状态只留 ID、预算、覆盖和 final result，详细记录落 DB；显式安全 serializer | 4.1、7.2 |
| 流式发送未校验答案 | 只发进度和 evidence，校验后一次发 `answer.final` | 7.4、8 |
| 评测规模和统计口径不匹配 | parser、retrieval、answer、route 分库；用成对胜负和目标子集退化数 | 10 |
| 范围接近重写 | Core、Enhanced、Product 分层，首次只批准 M1 至 M3 | 1、3、11、15 |
| 文件路径、幂等和 Settings 不完整 | 改用真实 `evals/`、`agent/nodes/`、`indexing/builders/` 和分层 API；补幂等表；移除 Settings 页面 | 4.3、8、11 |
| Contextual Retrieval 命名不准确 | 正式更名为 metadata-prefixed retrieval，并明确与 Anthropic 方法不同 | 6.1、13 |
