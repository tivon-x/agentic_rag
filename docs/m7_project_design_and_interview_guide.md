# Agentic RAG V2 项目设计与面试指南

> M7 文档交付。事实来源是当前 checkout、M1 至 M6 验收报告和冻结 artifacts。当前生产默认与里程碑状态以 [`v2_upgrade_plan.md`](research/v2_upgrade_plan.md) 顶部 YAML 为准。本指南不新增运行代码，也不把计划中的能力写成已交付能力。

## 1. 项目定位

Agentic RAG 是一个面向个人论文库的 evidence-first 学术 RAG 产品。用户把已经下载的论文放进本地库，系统负责解析、建立可回溯索引、搜索原文、按证据回答问题，并把答案带回论文和页码。

它解决的是“以后怎样快速找回论文中的依据”，不是全网论文发现、引用格式管理或自动写论文。当前产品界面是论文库、搜索、论文阅读、Chat，以及只读的评测展示页：`/library`、`/search`、`/papers/[id]`、`/chat`、`/evaluation`。

### 用户问题

- 论文来自用户自己的文件，来源边界比全网覆盖更重要。
- 搜索结果需要能回到论文、章节和页码，不能只给一段看似相关的文本。
- Chat 回答需要保留结构化 evidence，证据不足时要能显示限制，而不是补写来源。
- 解析失败、需要 OCR、索引失败和版本切换都要成为可见状态。

### 产品边界

- 运行时只在用户已导入的论文库内检索，不自动接入 Web search。
- 默认回答策略是 `fixed`，Adaptive 只保留为实验路径，不作为普通用户设置。
- Pipeline 名称和检索开关属于工程与评测层，普通用户不选择 B0、B1、B2、B3 或 S1。
- `pymupdf4llm` 失败时会显式走 legacy fallback；项目显示 `needs_ocr`，不承诺 OCR 已经完成。
- 不承诺公式语义解析、bbox 高亮、不同 PDF 修订版自动合并或多用户云端部署。

## 2. 系统总览

```text
上传文件
  -> SQLite 持久任务与幂等记录
  -> 解析与质量门
  -> papers / paper_versions / sections / passages
  -> 构建不可变 index version
  -> 校验 embedding 与 retrieval contract
  -> 激活版本
  -> FastAPI Search / Chat
  -> Next.js Library / Search / Paper / Chat
```

| 层 | 当前职责 | 主要入口 |
|---|---|---|
| Settings | 集中加载默认值、模型和索引合同 | [`core/settings.py`](../core/settings.py) |
| Parser | 从 PDF 生成页、章节、block、元数据和质量状态 | [`indexing/parsers/paper_parser.py`](../indexing/parsers/paper_parser.py)、[`indexing/paper_ingestion.py`](../indexing/paper_ingestion.py) |
| Catalog | 保存论文实体、解析版本、章节和 passage | [`api/db/papers.py`](../api/db/papers.py)、[`indexing/passages.py`](../indexing/passages.py) |
| Index | 保存 FAISS、BM25、manifest 和 active version | [`indexing/index_versions.py`](../indexing/index_versions.py)、[`indexing/indexer.py`](../indexing/indexer.py) |
| Retrieval | dense、BM25、fusion、dedupe、rerank、context packing | [`indexing/retrieval_pipeline.py`](../indexing/retrieval_pipeline.py)、[`indexing/retriever.py`](../indexing/retriever.py) |
| Agent | 路由、查询规划、检索工具和 grounded answer | [`agent/graph.py`](../agent/graph.py)、[`agent/nodes/`](../agent/nodes/) |
| API | 上传、任务、论文、搜索、Chat 和 SSE | [`api/routers/`](../api/routers/) |
| Web | 论文库、原页回跳、证据 Chat 和只读评测 | [`web/src/app/`](../web/src/app/) |
| Evaluation | KITE 公开 E2E 与内部检索诊断 | [`evals/`](../evals/) |

当前冻结的产品默认值如下：

| 项目 | 值 |
|---|---|
| Production pipeline | `v1_flat_rerank`，registry 名称 `b1` |
| Answer strategy | `fixed` |
| Parser | `pymupdf4llm`，失败时显式 legacy fallback |
| Index write mode | `versioned` |
| Embedding input | `raw` |
| Embedding max input | `6000` 字符 |
| Pipeline config hash | `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17` |

## 3. 数据与 evidence contract

### Parser 输出

项目自己的 parser protocol 统一保存 `ParsedPaper`。它包含：

- 页列表，页码、结构化文本、表格、源页 fingerprint 和确定性 `source_text`；
- 章节列表，标题层级、顺序、起止页、heading path 和 blocks；
- 论文元数据，标题、作者、年份、venue、DOI、arXiv ID，以及每个字段的来源和置信度；
- 质量结果，页覆盖、非空页比例、相对 legacy 的字符比、页码单调性、`needs_ocr` 和原因；
- parser 名称、版本、normalization 版本、状态、fallback 原因和耗时。

当前默认 parser 是 PyMuPDF4LLM 0.3.4 加 `structure-v1` 归一化。低置信元数据可以交给用户修正，不能把 parser 的猜测直接当成权威值。

### 稳定标识

| 对象 | 生成规则 | 用途 |
|---|---|---|
| `paper_id` | 上传文件字节的 SHA-256 | 区分不同文件实体，内容不同就不合并 |
| `paper_version_id` | `paper_id`、parser 名称与版本、normalization 版本的稳定 hash | 标识一次解析版本 |
| `section_id` | 解析版本、规范化 heading path、section ordinal 的稳定 hash | 在章节重建后保持可追踪 |
| `passage_id` | 解析版本、section、页码、全局 ordinal、quote hash 的稳定 hash | 绑定检索结果和 evidence |

稳定 ID 让 evidence 不依赖临时数组下标。文件移动、索引重建或元数据修正时，仍可以说明它属于哪篇论文、哪个解析版本和哪个 passage。

### `quote_text` 与 `retrieval_text`

`quote_text` 是 parser 产生的源文，用于 context、答案和用户可见引用。`retrieval_text` 可以带经过验证的标题、作者、年份、章节和 block 前缀，用于 dense 或 BM25 的检索表示。metadata prefix 不进入用户可见的 quote，也不能被当成论文原文。

passage 构建会先计算前缀，再按剩余空间切分 quote，完整 embedding 输入受 `EMBEDDING_MAX_INPUT_CHARS=6000` 保护。用户修改元数据时，在同一事务里重建 prefix、检查长度并写入 reindex job；失败会一起回滚。`quote_text` 不因元数据修正而变化。

### Chat evidence

`ChatEvidence` 至少保留 `node_id`、来源、章节路径、页码、quote，并在可用时带 `paper_id`、标题、score 和 relevance。Chat 路由只接受能和检索 artifact 中的 `node_id` 或 quote 对上的 evidence。模型自行写出的 paper ID、标题或 quote 无法绑定检索记录时会被丢弃，不会写进会话。

正常 SSE 顺序是：

```text
progress -> evidence -> answer.final
```

生成、检索或保存失败只发送 `stream-error`，不发送未持久化的 `answer.final`。旧会话没有 evidence 字段时仍然可读，前端不从答案句子里猜 citation。

## 4. B1 固定检索路径

B1 是 registry 中的 `v1_flat_rerank`，当前生产默认。它启用 dense 和 BM25，使用 `whitespace_v1` tokenizer 与 min-max fusion，随后执行 FlashRank rerank，不使用 metadata prefix，也不做邻居扩展。

```text
query normalize / plan
  -> dense recall + BM25 recall
  -> fusion
  -> dedupe
  -> rerank
  -> context packing
  -> grounded answer
```

`FusionRetriever` 会保留每个阶段的候选、分数、timing、query plan、去重信息和最终打包结果。检索调试可以看到阶段产物，用户界面只显示经过绑定的 evidence。

Pipeline registry 同时保存 index contract、query-time retrieval contract 和 config hash。加载 active index 时会校验 embedding provider、model、dimension、input mode、上下文检查、最大输入长度，以及 dense/sparse 的 retrieval schema。合同不一致就拒绝加载，要求重建，不静默降级。

## 5. 两层评测职责

### KITE：公开 E2E

M6 冻结了 KITE AI Papers snapshot：

| 项目 | 冻结值 |
|---|---|
| Upstream | [`D-Star-AI/KITE`](https://github.com/D-Star-AI/KITE)，commit `85e71ad63db9ea410eccbb0158f94e7d72462b99` |
| Query | `15` 题，SHA-256 `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3` |
| Corpus | `134` 个 PDF，manifest SHA-256 `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8` |
| Parser | PyMuPDF4LLM 0.3.4，`structure-v1` |
| Generation | `qwen3.7-plus`，fixed answer path |
| Judge | `qwen3.7-plus`，`kite-official-compatible-v1`，temperature `0` |

KITE 看最终回答的协议分数、延迟、上下文等 E2E 结果。报告必须同时保存 query、reference、rubric、answer、检索拥有的 evidence、judge 字段、错误和 provenance。当前分数是本项目固定 judge 协议下的比较结果，不与 KITE 上游的绝对分数直接比较。

### Internal Diagnostic：解释为什么

内部 Retrieval Dataset 继续用于 Recall@K、MRR、nDCG、context passage recall、阶段 latency 和坏例分析。它可以解释一个 pipeline 是否漏了术语、章节或论文，但不能把 retrieval 分数直接写成最终答案质量。

两层结果的关系是：KITE 决定最终产品表现是否值得采用，内部诊断帮助定位是哪一段召回或排序出了问题。两者都必须绑定冻结数据、parser artifact、index manifest、配置和代码版本。

## 6. 已验证的候选与失败决策

### S1：内部 holdout 未通过

S1 是 `v2_fixed_hybrid`，使用 dense 加 BM25、mixed tokenizer、min-max fusion，不使用 metadata prefix、rerank 或邻居扩展。

| 数据集 | Pipeline | Recall@10 | MRR@10 | nDCG@10 | Context recall | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| holdout | B1 | 0.875000 | 0.680258 | 0.710130 | 0.875000 | 389.0657 |
| holdout | S1 | 0.885417 | 0.766667 | 0.776056 | 0.854167 | 231.1416 |
| old dev | B1 | 0.614583 | 0.356622 | 0.405581 | 0.583333 | 424.5483 |
| old dev | S1 | 0.729167 | 0.435499 | 0.483281 | 0.697917 | 228.8369 |

S1 在 old dev 更好，但 holdout 的 context passage recall 低于 B1，冻结 gate 失败。因此 S1 保留为固定策略候选和诊断材料，B1 继续作为 fixed baseline。

### B0 至 B3：KITE 正式比较

四条 pipeline 使用相同 KITE snapshot、parser artifact、embedding、generation、judge 配置和 clean evaluation commit，均为 `formal_run=true`、`15/15` 有效。

| Pipeline | 平均协议分 | 有效题数 | p95 latency ms | 平均 context tokens |
|---|---:|---:|---:|---:|
| B0 | 4.2667 | 15/15 | 173270.2967 | 14450.6 |
| B1 | 6.0000 | 15/15 | 207923.7460 | 15810.1333 |
| B2 | 6.4000 | 15/15 | 187091.5157 | 18256.0 |
| B3 | 6.3333 | 15/15 | 176588.0740 | 17182.7333 |

晋级门槛要求：平均分至少提升 `0.5`、逐题 loss 不超过 `2`、至少 `4` 个 win，并满足 p95 与 context 的 `1.5x` 约束和 evidence contract。相对 B1：

- B2 是 `5` 胜、`7` 平、`3` 负，平均分只提升 `0.4`；
- B3 是 `5` 胜、`6` 平、`4` 负，平均分只提升 `0.3333`；
- B2 的 loss 是 `ai-papers-003/006/010`，B3 的 loss 是 `ai-papers-003/006/010/015`。

两者都没有 promotion candidate。生产默认继续 `b1 / v1_flat_rerank`，评测不会自动改线上配置。

### Adaptive：安全预算通过，质量门槛失败

M4.1.1 和 M4.1.2 都只在冻结 B1 contract 上比较补检路径，没有修改 dense、BM25、fusion、reranker 或 active index。两次结果都没有证明 Adaptive 的净收益。

最新 M4.1.2 的 route macro F1 是 `0.7029`，低于 `0.80`；fixed recall 是 `0.4167`，低于 `0.75`。answer 中 fixed coverage 为 `0.4375`，adaptive 为 `0.3125`；citation correctness 为 `0.2326` 和 `0.1687`。adaptive-eligible 只改善 `1` 条，退化 `4` 条，平均 rounds 为 `1.7917`。重复查询、轮数、tool call 和 termination 等安全预算通过，但质量门槛失败。

因此 `ANSWER_STRATEGY=fixed` 保持默认，M4.2 不启动。Adaptive 的上限仍是 3 个 requirement、2 轮、4 次 retrieval、12 条 evidence、12000 context tokens；这些是实验边界，不是产品承诺。

## 7. 产品链路与可靠性边界

### 上传与索引

- `POST /api/index/files` 强制 `Idempotency-Key`，相同 key 和相同 request hash 返回同一 job，不同 hash 返回冲突。
- 文件先写入隔离 staging，校验后才进入 job 目录；路径必须在 `UPLOAD_ROOT` 下，支持 `.pdf`、`.md`、`.txt`。
- versioned 模式由 SQLite 持久 job 和单个 leased worker 处理。worker 解析 catalog，构建临时 index version，校验后再激活。
- 新版本失败会保留失败信息，旧 active version 不被改写。lease 过期的 job 可以恢复或在达到 attempt 上限后失败。

### Active index 与回滚

SQLite `app_state` 是 active version 的权威状态，`active.json` 是可重建镜像。激活前会检查 manifest、FAISS、BM25、embedding contract 和 retrieval contract。切换旧的 ready version 使用：

```bash
python main.py activate-index <previous-version-id>
```

`INDEX_WRITE_MODE=legacy` 只读 legacy 索引，API 上传会返回 409；如果必须用 legacy CLI 写入，需要先停止 API。代码回滚使用独立 commit 的 `git revert`，不删除用户文件、版本索引或数据库备份。

### Search、Paper 与 Chat

- Search 返回论文、章节、页码、`quote_text`、vector/BM25/fusion/boost/final 分数和回跳链接。
- Paper 页面展示章节目录、页码、PDF 原页、解析状态、fallback 原因和可校正元数据。
- Chat 的结构化 evidence 按 assistant 回答分组，链接形如 `/papers/{paper_id}?page={page}`；页码或 paper ID 缺失时显示不可用状态，不猜测。
- `/evaluation` 只展示冻结的清洗结果，不触发模型、索引、参数编辑或 Pipeline 切换。

### 测试与正式结果

单元测试使用 FakeEmbeddings、fake model 或 monkeypatch，不发真实外部请求。正式 KITE 报告必须保存逐题 win/tie/loss、坏例、错误、延迟和 provenance。确定性测试、Fake 模型、UI smoke 和 dirty run 都不能写成完整 E2E Benchmark。

## 8. 为什么普通用户看不到 Pipeline 开关

Pipeline 不是一个只影响排序的 UI 偏好，它决定索引中的 dense/sparse 表示、tokenizer、fusion、rerank、context packing 和 manifest contract。切换到不兼容的 pipeline 可能让现有 index 无法安全查询，也会让已冻结的评测结论失去含义。

所以普通用户只看到稳定的 Search、Chat 和 evidence。候选 pipeline 由工程侧先冻结数据和门槛，完成逐题比较、证据审计和失败分析，再经过单独的生产批准。M6 的 B2/B3 就因为没有通过门槛而没有自动晋级。

## 9. 面试表达

### 30 秒版本

这是一个面向个人论文库的 evidence-first RAG 系统。上传的 PDF 先经过可降级的解析和质量检查，生成带稳定 ID 的论文、章节和 passage，再建立带 embedding contract 的不可变索引。线上默认使用 dense 加 BM25、fusion 和 rerank 的 B1 路径，Chat 只展示能回到论文和页码的结构化 evidence。检索候选不会凭感觉上线，而是在内部诊断集和 KITE AI Papers 的固定 E2E 结果上做晋级判断。

### 2 分钟版本

项目的核心约束是来源可回溯和结果可复核。数据层用文件字节 hash 标识论文，用 parser/version/normalization 生成解析版本，用 section 和 passage 的稳定 hash 绑定证据。`quote_text` 保留原文，`retrieval_text` 只给检索加入经过验证的元数据前缀，因此检索可以利用章节语境，用户看到的引用仍然是源文。

运行时，上传请求先经过路径和幂等校验，SQLite 保存 job，单个 worker 在 versioned 模式下构建新索引。新索引必须通过 FAISS、BM25、embedding 和 retrieval contract 校验后才能激活，SQLite 是 active pointer 的权威来源。查询经过 dense/BM25 召回、fusion、去重、FlashRank 重排和 context packing，最终由 fixed answer path 生成回答；Chat 只把 retrieval-owned evidence 传给前端。

评测分成两层。内部数据看 Recall、MRR、nDCG 和 context recall，用来定位检索问题；KITE 的固定 15 题、134 篇 PDF 结果看最终回答。B2 的平均分高于 B1，但提升只有 `0.4` 且有 `3` 个逐题 loss，没有达到晋级门槛，B1 因此继续生产。Adaptive 两次复验也没有证明质量净收益，所以默认仍是 fixed。

### 常见追问

| 问题 | 回答要点 |
|---|---|
| 为什么不直接把 PDF 切 chunk？ | 需要论文、章节、页码和稳定 evidence 地址。平面 chunk 无法可靠承担原页回跳和元数据修正。 |
| 为什么拆成两个文本字段？ | `retrieval_text` 服务召回，`quote_text` 服务源文和引用，避免 metadata prefix 泄露到用户答案。 |
| 为什么 B1 没有换成分数更高的 B2？ | B2 只高 `0.4` 分，低于 `0.5` 门槛，还有 `3` 个逐题 loss，不能只看平均分。 |
| 为什么没有上线 Adaptive？ | 两次冻结复验都未通过 route 或 answer 质量门槛。安全预算通过不等于质量通过。 |
| 模型能不能自己写引用？ | 不能成为可信来源。服务端把模型 evidence 和 retrieval-owned evidence 对齐，无法绑定的条目丢弃。 |
| 进程被杀时怎么办？ | job 和 lease 在 SQLite 中，启动会恢复过期任务；新版本激活前还会检查 worker lease，旧 active 不被半成品覆盖。 |
| 为什么不公开 KITE PDF？ | 论文再分发许可未作为 M7 已决条件。公开部署只使用许可明确的 demo corpus，KITE 通过外部获取、hash 和私有缓存参与评测。 |
| 下一步是不是部署？ | 不是。M8 已暂缓，当前不实施部署；如恢复，需要重新决定平台、持久盘、密钥、网络和数据许可。 |

## 10. 简历和面试中的安全表述

下面的数字都来自正式报告，可以直接作为事实素材，但仍需按目标岗位改写：

- 面向个人论文库构建 local-first evidence-first RAG，完成解析、稳定 passage ID、不可变索引版本、Search、Paper、Chat 和结构化 evidence 闭环。
- 在冻结的 KITE AI Papers snapshot 上对 B0 至 B3 做 `15` 题正式比较，B1 平均协议分 `6.0000`，B2 为 `6.4000`，但 B2 有 `3` 个逐题 loss 且未达到晋级门槛，生产默认保持 `v1_flat_rerank`。
- 将公开 E2E 评测和内部 retrieval diagnostic 分开，保留逐题结果、失败案例、parser/index/config hash 与运行 provenance，避免把 Recall 提升写成答案质量提升。
- 设计 `quote_text` / `retrieval_text` 双表示和 retrieval-owned evidence 校验，保证 metadata prefix 不进入用户可见引用，证据可以回到论文页码。

未知数字使用 `[待补充]` 或非量化表述。不要把一次 smoke、脏工作区结果、Fake 模型结果或 UI 检查写成正式线上收益。

## 11. 不应声称的内容

- “KITE 证明系统全面领先”。当前结果只支持固定 snapshot 和本项目 judge 协议下的 pipeline 比较。
- “B2/B3 已经上线”或“评测自动切换生产”。事实是 B1 仍为生产默认，promotion candidates 为空。
- “Adaptive 已经提升回答质量”。两次复验没有通过质量门槛。
- “项目支持 OCR、全网学术搜索、引用格式管理、公式语义解析或多用户云部署”。这些不在当前交付边界。
- “内部 Recall 等于最终回答质量”。两层评测承担不同职责。

## 12. 事实来源

### 当前计划与状态

- [`docs/research/v2_upgrade_plan.md`](research/v2_upgrade_plan.md)，M7 章节和顶部 YAML 状态区块。

### 代码合同

- [`core/settings.py`](../core/settings.py)，运行默认值和环境边界。
- [`indexing/parsers/paper_parser.py`](../indexing/parsers/paper_parser.py)，parser protocol 和稳定 hash。
- [`indexing/passages.py`](../indexing/passages.py)，section/passage ID、prefix 和 embedding 长度门。
- [`indexing/retrieval_pipeline.py`](../indexing/retrieval_pipeline.py)，B0、B1、B2、B3、S1 registry、contract 和 hash。
- [`indexing/retriever.py`](../indexing/retriever.py)，召回、fusion、dedupe、rerank、packing 和 trace。
- [`indexing/index_versions.py`](../indexing/index_versions.py)，manifest、兼容性校验、激活和回滚。
- [`api/services/index_worker.py`](../api/services/index_worker.py)、[`api/db/migrations.py`](../api/db/migrations.py)，任务、lease、迁移和恢复。
- [`api/routers/chat.py`](../api/routers/chat.py)、[`api/models/chat.py`](../api/models/chat.py)、[`core/rag_answer.py`](../core/rag_answer.py)，SSE、Chat evidence 和回答渲染。

### 冻结验收与 artifacts

- [`M1 验收`](implementation/m1_acceptance.md) 至 [`M2 验收`](implementation/m2_acceptance.md)：运行、索引、解析、目录和页码证据。
- [`M3.2 策略收口`](implementation/m3_2_strategy_acceptance.md)：B1 与 S1 的冻结诊断及固定 baseline 决策。
- [`M4.1`](implementation/m4_1_acceptance.md)、[`M4.1.1`](implementation/m4_1_1_retrieval_quality_acceptance.md)、[`M4.1.2`](implementation/m4_1_2_adaptive_eval_acceptance.md)：Adaptive 质量复验和失败边界。
- [`M5 固定产品`](implementation/m5_fixed_product_acceptance.md)、[`M5.1 Chat`](implementation/m5_1_chat_experience_acceptance.md)、[`M5.1 Web UI`](implementation/m5_1_web_ui_fix_acceptance.md)：产品、会话、SSE、evidence 和回跳。
- [`M6A KITE 数据`](implementation/m6a_kite_data_acceptance.md)、[`M6B B1 基线`](implementation/m6b_kite_b1_acceptance.md)、[`M6C Pipeline 比较`](implementation/m6c_kite_pipeline_acceptance.md)、[`M6D 只读展示`](implementation/m6d_evaluation_presentation_acceptance.md)。
- [`KITE 正式报告`](kite_benchmark_report.md)、[`生产 Pipeline 决策`](production_pipeline_decision.md)、[`KITE summary`](../artifacts/evals/kite/summary.json)、[`KITE manifest`](../artifacts/evals/kite/manifest.json)。
