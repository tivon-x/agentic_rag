# Agentic RAG V2 阶段一技术调研报告

> 调研日期：2026-07-25  
> 调研范围：当前 `master` 分支、工作区内尚未提交的 FastAPI/Next.js 改造、现有评测与 20 组外部一手资料  
> 本阶段边界：只审计、调研和设计，没有修改业务代码、测试、配置或用户已有改动

## 1. 调研结论

V2 应定位为面向个人已读论文库的 local-first 单用户科研助手。它需要完成一条日常可用的闭环：导入论文，确认解析与索引状态，搜索原文，基于证据回答单篇或多篇问题，跳回 PDF 原页核对，把答案和证据保存到研究工作区。

普通 RAG 在这个场景里不够，原因落在四个可测试的问题上：

- PDF 被切成缺少章节、页码和版面位置的文本块后，即使召回了相似内容，用户也很难核对原文。
- 学术问题常含缩写、术语、方法名和跨论文比较条件，单次 top-k 容易漏掉关键论文、章节或对照项。
- 多论文回答需要知道每个结论由哪些证据支持、哪些维度仍缺证据。只让模型自由调用一次检索工具，不能稳定控制覆盖率、成本和停止条件。
- 当前评测只能说明 6 个检索问题上的离线结果，无法证明真实生成质量、引用完整性、拒答能力或 Agent 策略收益。

推荐的四个关联亮点是：

1. **可回到论文页的稳定证据索引**，把论文、章节、段落、页码和原文做成稳定证据地址。
2. **metadata-prefixed passage retrieval 与受控扩展**，检索文本包含已验证的论文与章节元数据，引用文本保持原文不变，混合召回、RRF 融合、重排和相邻段扩展各自可以消融。
3. **证据充分度驱动的有界补检**，简单问题走固定 pipeline，比较和多跳问题才进入最多两轮的补检图，并记录停止原因和预算。
4. **答案、引用和运行轨迹共同评测**，按 claim 检查引用正确性与完整性，同时记录召回、拒答、延迟、token、工具调用和坏例。

GraphRAG、多 Agent、RAPTOR、ColBERTv2、Docling 默认解析、GROBID 服务和云端任务队列都不进入 V2 主线。它们各自有适用场景，但当前项目的评测规模、个人论文库体量和维护预算不足以证明这些复杂度有收益。V2 先交付 Core，再分别审批 Enhanced 和 Product，避免把可靠性、Agent 和完整工作区绑成一次重写。

## 2. 仓库与工作区基线

### 2.1 开始前状态

| 项目 | 审计结果 |
|---|---|
| 仓库根目录 | `C:\Users\27564\Documents\code\ai\rag\agentic_rag` |
| 当前分支 | `master` |
| 已修改文件 | `main.py`、`pyproject.toml`、`uv.lock` |
| 未跟踪内容 | `.sisyphus/`、`Agentic-RAG 项目指南.md`、`api/`、`tasks.md`、`tests/test_api.py`、`web/` |
| 保护措施 | 未清理、回退、覆盖、暂存或提交任何用户改动 |

审计阅读了根目录和子目录的 `AGENTS.md`、`README.md`、`tasks.md`、项目指南、评测指南与报告、依赖与部署文件、`agent/`、`core/`、`indexing/`、`llms/`、`evals/`、`ui/`，以及尚未提交的 `api/`、`web/` 和对应测试。以下结论以源码与实际命令为准，README 只用于交叉核对。

### 2.2 实际运行环境

| 检查项 | 实际结果 | 判断 |
|---|---|---|
| 系统 `python --version` | Python 3.10.11 | 不满足项目声明的 Python 3.12+ |
| `uv run python` | `.venv` 中 Python 3.12.11 | 项目环境可满足要求 |
| `uv run pytest -q` | 11 个收集错误 | 实际调用了缺依赖的全局测试环境 |
| `uv run --extra dev python -m pytest -q` | 154 passed，1 failed | 正确开发依赖下主体可运行，但全量测试不稳定 |
| 单独运行失败测试并清理环境变量 | 通过 | 失败来自跨测试环境污染 |
| `uv run --extra dev ruff check .` | 通过 | 静态检查当前通过 |
| `pnpm --dir web lint` | 通过 | 未提交前端的 lint 当前通过 |

全量测试失败是可复现问题。API 测试加载根目录 `.env` 时，自定义 dotenv 逻辑会修改进程级 `os.environ`，后续设置测试读到了用户的 `EMBEDDING_MODEL`，预期默认值因而变化。单独清理该变量后，失败用例通过。README 的 `uv sync --dev` 也与 `pyproject.toml` 不一致，开发依赖定义在 optional extra 中，正确入口应统一为 `uv sync --extra dev`。

## 3. 当前仓库真实能力

### 3.1 已经成立的能力

| 能力 | 源码证据 | 当前边界 |
|---|---|---|
| LangGraph 主图与条件路由 | `agent/graph.py`、`agent/edges.py` | 有 retrieve/direct/oos 路由和查询规划 |
| 并行子问题执行 | 主图通过 `Send` 分发 rewritten questions | 当前实现会与工具内的子查询循环叠加 |
| 混合检索 | `BM25Retriever`、FAISS、`FusionRetriever` | 融合使用逐查询 min-max 分数，稳定性有限 |
| FlashRank 重排 | 检索器中的可选 reranker | 可降级，已有一定测试覆盖 |
| 文档、章节、段落树 | `indexing/models/doc_tree.py`、`indexing/builders/hierarchical_index_builder.py` 与 parser | PDF 的 section 实际按页生成 |
| 上下文打包 | rerank、邻居扩展、token budget、证据分组 | 首段可能突破预算，扩展后膨胀控制不足 |
| 结构化证据 | 工具 artifact、`EvidenceCaptureMiddleware`、`GroundedAnswer` | 引用仍主要依赖模型输出，没有 claim 级校验 |
| 技术调试界面 | Gradio 展示路由、计划、召回、重排、上下文和引用 | 适合作为技术工作台，不是完整科研产品 |
| 离线评测框架 | 检索、答案、路由数据集和 leaderboard | 样本小，生成评测是抽取式 fallback |
| FastAPI/Next.js 雏形 | 未提交的 `api/`、`web/` | 已有聊天、上传和索引状态，产品闭环尚未完成 |

证据通过工具的 structured artifact 回写到图状态，是当前最有价值的 Agent 工程基础。它让检索证据可以脱离自然语言消息单独保存，也为后续 trace、引用校验和调试界面提供了结构化入口。

### 3.2 技术亮点分级

#### 已成立

- 混合召回、重排、上下文打包和结构化引用已经形成可运行链路。
- 图状态、路由、查询规划和并行分支都有实际代码，不是 prompt 中的概念演示。
- Gradio 调试面板能展示主要 RAG 阶段，适合保留为工程诊断入口。
- 测试覆盖了大部分索引、检索、图节点、评测和新增 API，正确依赖环境下有 154 个测试通过。

#### 待验证

- `flat_rerank` 在现有 6 个检索问题上优于 flat，但样本不足，不能推断对真实论文库有效。
- FlashRank 对不同语言、公式密集段落和学术术语的收益尚无分层实验。
- query planning 对跨论文比较是否提高覆盖率没有独立对照。
- corpus profile 的术语增强可能提高领域召回，也可能把无关论文推到前面。
- 尚未提交的 API 和前端能跑通基础流程，但缺少重启恢复、并发索引安全、证据阅读和日常产品流程验证。

#### 当前不成立

- **hierarchical 优于 flat 不成立**。现有报告中 hierarchical 的 Recall@k 为 0.8333，低于 flat_rerank 的 1.0；MRR 和 nDCG 也更低。
- **语义层级检索不成立**。父节点虽然有均值向量，实际索引默认只有 paragraph；父节点召回依赖全树扫描和词项重叠。
- **生成 groundedness 已被证明不成立**。离线模式把检索原句抽出来作为答案，groundedness=1.0 是构造结果，不能代表 LLM 生成。
- **生产级 Agent 终止控制不成立**。中间件的计数存在实例字段中，会跨请求、跨子查询共享，且在 agent 执行后检查，不能可靠阻止循环。
- **可恢复的异步索引不成立**。API 使用进程内 `asyncio.create_task` 和 set；进程重启后任务丢失，也没有单写者和原子版本切换。
- **正确的最终答案流式输出不成立**。SSE 收集所有 `on_chat_model_stream` 事件，会把路由、规划、改写、子 Agent 和聚合模型的 token 拼到同一答案。

### 3.3 V2 能力分类

| 类别 | V2 内容 | 是否作为技术亮点 |
|---|---|---|
| 技术能力 | 稳定证据索引、metadata-prefixed hybrid retrieval、有界补检、claim 级引用与评测 | 是，必须用原理、源码和实验说明 |
| 产品能力 | 论文库、PDF 阅读器、Search、单篇问答、多篇比较、研究工作区、历史结果 | 是，说明这些技术如何进入日常流程 |
| 工程配套 | SQLite、索引版本、任务恢复、缓存、SSE、Docker、环境统一 | 否，作为可靠性与可复现基础 |
| 展示载体 | FastAPI、Next.js、Gradio 调试台 | 否，框架与页面数量不算技术贡献 |

## 4. 失败模式与根因

### 4.1 索引与解析

当前 PDF parser 把每一页当成 section，标题是 `Page N`。论文的 Abstract、Method、Experiments、Limitations 等语义没有进入树结构，表格、公式、标题层级和引用关系也没有成为独立元素。`doc_id` 来自解析后文件路径的 SHA1，同一文件内容变化时 ID 不变，移动文件时 ID 改变，无法稳定表示论文和版本。

索引器向已有 FAISS 追加随机 UUID，没有去重、删除、事务和 index version。多文件上传会并发加载并写回同一组 FAISS、BM25 pickle 和 JSON tree，存在丢更新与文件损坏风险。FAISS 加载允许危险 pickle 反序列化，只应读取本机可信索引。

### 4.2 召回、融合与上下文

BM25 对语料和查询直接 `.split()`，中文文本和连字符学术术语的词项质量较差。融合器把每个查询内的 FAISS 距离和 BM25 分数分别做 min-max，再按 alpha 加权。候选分数相等时全部变为 0，不同查询之间的分数也不可比。RRF 更适合这里只依赖排名、不假设异构分数可校准的情况。

hierarchical 路线只索引叶节点。查询父节点时遍历整个树并做词项覆盖，随后再扩展邻居和父级，召回、扩展和打包的职责混在一起。当前主图把每个 rewritten question 发给子 Agent，工具又遍历完整 `query_plan.subqueries`，一个计划会产生乘法式重复检索。

打包器允许第一个 passage 突破 token budget，邻居扩展发生在重排之后但缺少独立预算，长页面可能占满上下文。`retrievalEvidence`、`packedContexts` 等状态包含排名信息，却没有每阶段耗时、模型 token、缓存命中、预算消耗和 index version。

### 4.3 Agent 策略

Agent 只有一个 `search_relevant_chunks` 工具。它无法明确表达查元数据、限定论文、查看相邻段、检查证据覆盖、补查缺失维度和结束任务。自由 ReAct 循环承担了本可由图状态和确定性节点完成的工作。

`FallbackMiddleware` 把 `iteration_count` 和 `tool_call_count` 放在中间件实例上。多个请求或并行子问题共享计数，后一次请求可能继承前一次预算。计数在 `after_agent` 更新，只能在一次 agent 已经运行后返回 fallback，不能作为循环中每步的强制预算。

当前 confidence 主要来自模型自评或证据数量启发式，没有与证据覆盖率、引用校验或历史准确率校准。direct answer、历史摘要等路径的失败恢复也不一致。

当前 `GraphState` 继承 `MessagesState`，同时保存完整消息、`retrievalEvidence`、`packedContexts`、`evidenceGroups` 和最终答案。若直接换成 SQLite checkpointer，这些候选、原文和排名会在多个 checkpoint 中重复写入，状态会随轮次快速膨胀。V2 需要让图状态只保存 ID、预算、覆盖率和当前结果，把候选、证据与 trace 放到数据库表中。

现有 API 用 `session_id:消息数` 作为 `thread_id`，每次提问都是新 thread，因此 InMemorySaver 既不能承担连续会话，也不能在进程重启后恢复。更重要的是，checkpointer 只保存图状态，不会自动重新调度未完成任务。真正的恢复需要持久 `runs` 表、lease、启动扫描和显式 worker；session history 应从数据库快照加载，不能混同于 run checkpoint。

### 4.4 评测

| 数据集 | 样本数 | 分布 | 主要缺口 |
|---|---:|---|---|
| retrieval | 6 | easy 2、medium 3、hard 1 | 只有 gold doc id，无 gold passage/section |
| answer | 5 | easy 2、medium 2、hard 1 | 离线抽取，无真实生成与 claim 标注 |
| routing | 6 | 全部 easy | 每类 2 条，不能覆盖含糊、拒答和复杂度路由 |
| 评测语料 | 25 篇 | 项目内固定语料 | 缺少扫描件、双栏、表格、公式和跨语言分层 |

现有 baseline 的主要数字如下：

| 配置 | Recall | MRR | nDCG | Citation precision | Completeness |
|---|---:|---:|---:|---:|---:|
| baseline_flat | 1.0000 | 0.8500 | 0.8681 | 0.2833 | 0.2842 |
| flat_rerank | 1.0000 | 1.0000 | 1.0000 | 0.4333 | 0.1882 |
| hierarchical | 0.8333 | 0.8333 | 0.8084 | 0.3000 | 0.1351 |

这轮评测使用 `FakeEmbeddings`，其向量来自哈希字节，不代表语义相似度。结果主要反映 BM25、FlashRank 和小样本偶然性。answer leaderboard 已标注 fallback-only，这个边界需要在 V2 中继续保留，不能把离线抽取分数写成生成质量。

### 4.5 后端与前端

CLI 默认把 `base_dir` 解析成 `core/`，因此数据位于 `core/data`；API 显式传仓库根目录，数据位于根目录 `data/`。两个入口会读写不同索引。配置又同时来自 `core/settings.py`、`core/config.py`、`llms/llm.py`、`indexing/embeddings.py` 和 `main.py` 的 dotenv，复现路径不唯一。Docker 健康检查导入了不存在的顶层 `settings` 模块，也只启动 Gradio。

API 上传会把整个文件读进内存，没有大小、MIME、文件名和路径校验。后台任务没有重试、取消、阶段进度和启动恢复。完成状态不检查 `Indexer.index()` 的实际结果。

聊天 SSE 会把所有 `on_chat_model_stream` token 当作最终答案。即使只过滤 answer node，如果答案在 claim validation 前发送，用户仍可能先看到错误版本再被替换。V2 应只流式发送进度和已确认 evidence，在校验完成后一次发送 `answer.final`。

Next.js 当前只有主页、聊天页和知识库页。知识库能上传和轮询任务，聊天能建立会话并消费 EventSource，但证据以纯文本显示，无法打开 PDF 原页。尚无论文库、元数据编辑、全文搜索、单篇阅读、多论文比较、研究工作区、历史结果保存、失败重试和普通/调试模式。Gradio 可以继续作为技术工作台，Next.js 应承担正式产品。

## 5. 外部研究的可迁移结论

### 5.1 通用 RAG

Anthropic 的 Contextual Retrieval 会调用模型，为每个 chunk 生成一段解释该 chunk 在整篇文档中位置和含义的专属上下文，再同时用于 embedding 与 BM25。其公开实验中，contextual embedding 与 contextual BM25 把 top-20 失败率降低 49%，再加重排降低 67%。该数字来自其数据和模型，不能直接当作本项目目标。

V2 不实现这项方法，也不沿用它的名称。论文库当前先采用可重复、低成本的 **metadata-prefixed retrieval**：把经过验证的标题、作者、年份、章节路径和 block type 加到 `retrieval_text`，`quote_text` 保持原文。它同样处理段落脱离语境的问题，但没有 chunk-specific LLM context，预期收益必须由本项目消融证明。

Parent-child 与 multi-vector 适合“短文本容易匹配、长上下文适合阅读”的矛盾。当前项目已有树和扩展器，不需要重新引入 LangChain classic 的 `MultiVectorRetriever`。V2 应让 section heading、caption 和 paragraph 各自产生可检索表示，命中后通过稳定 parent id 回到原文，再按预算扩展。

上下文压缩采用可追溯的选择式压缩：重排后只保留支持问题的 passage，再按 token budget 加入标题、caption 和相邻段。V2 不用 LLM 改写或摘要原始证据，因为压缩文本会让 PDF 原句和模型看到的内容不一致。长答案需要的覆盖由多组短证据完成，不把整篇论文塞进上下文。

RAPTOR 的递归聚类与摘要适合长文档的整体性、多步问题，但索引阶段需要额外 embedding、聚类和摘要生成。个人论文库先用论文元数据、章节结构、段落和受控扩展覆盖同类问题；只有全库技术演进问题在固定与 Agent pipeline 上持续失败时，才值得评估递归摘要。

GraphRAG 的 global search 会在社区报告上做 map-reduce，官方文档也明确标注资源密集。微软源码还为 map/reduce 分别统计 LLM 调用、token 和延迟。个人论文库当前没有稳定的实体关系抽取评测，也不需要为“全库主题总结”承担知识图谱构建、社区发现和摘要更新成本。

### 5.2 Agentic RAG

Adaptive-RAG 根据问题复杂度选择 no retrieval、single-step 或 iterative retrieval，解决简单问题被过度处理、复杂问题检索不足的冲突。V2 采用同一原则，但不训练独立复杂度分类器。现有结构化路由模型增加 `fixed` 与 `adaptive` 策略即可，并由评测数据检查路由是否值得保留。

CRAG 在召回结果上增加轻量评估器，再按置信度触发纠正。它使用 web search 扩展静态语料，论文库产品不应越过用户已读库边界。可迁移部分是证据评分和补救动作：改写查询、限定论文、查章节标题、扩展相邻段；找不到时明确拒答。

Self-RAG 通过训练模型生成 reflection tokens 来决定检索和评判证据。当前项目使用通用 OpenAI-compatible 模型，没有训练数据和微调链路，不能把 prompt 反思等同于 Self-RAG。V2 只采用按需检索、相关性和支持度检查的设计原则。

PaperQA2 的代码把 paper search、gather evidence、generate answer、complete/reset 分成不同工具，状态显式记录论文数、证据数和当前成本。测试覆盖崩溃恢复、增量索引、并发、超时、路径、token 与成本。当前项目不需要复制它的外部文献搜索，但应采用工具边界、证据状态和终止动作。

OpenScholar 的结果同时显示两件事：自反馈、重排和多样检索能改善科学综述；只追求引用准确率会牺牲覆盖率。Nature 论文报告 PaperQA2 的引用准确率可匹配或超过 OpenScholar，但经常依赖少量论文，覆盖和组织较弱。V2 因而必须同时测 citation correctness 和 completeness，不能用单一 groundedness 决定质量。

memory 只保存会话消息、压缩后的历史摘要和用户显式保存的 artifact，不从对话自动提取长期事实，也不把旧答案当新问题的证据。human-in-the-loop 放在产品边界：用户可以修改元数据、选择论文范围、重试失败任务和确认保存结果；单次问答图不在中途暂停等待审批，以免增加恢复和交互复杂度。

LangGraph persistence 文档说明 checkpointer 按 `thread_id` 保存每个 super-step 的状态，fault-tolerance 文档说明恢复仍由应用重新调用同一 thread。由此得到两个边界：checkpoint 不是 session memory，checkpoint 也不是任务队列。V2 选择 `thread_id=run_id`，会话历史由数据库快照提供；持久 run worker 用 lease、heartbeat 和启动扫描重新调用同一 run。GraphState 只保存小型控制字段，并用 `JsonPlusSerializer(pickle_fallback=False)` 和 strict msgpack 禁止不安全回退。

`AsyncSqliteSaver` 官方参考同时提醒，SQLite 写性能不适合通用生产负载。这个结论与本项目并不矛盾：V2 只服务单用户、本机单 run worker、compact state，并把 checkpoint 放在独立数据库中。若产品边界变成多用户或并发服务，这个前提失效，必须重新选择持久层，不能继续把 SQLite 描述成通用生产方案。

### 5.3 论文解析

Docling 提供本地 PDF 解析、文档层级、reading order、表格结构、公式和页级来源坐标。它仍是有价值的长期候选，但不能直接成为 V2 默认 parser。Docling 2.114.0 的[官方 PyPI 页面](https://pypi.org/project/docling/2.114.0/)仍把 title、authors、references 和 language metadata extraction 列为后续能力；仓库 Issues 也有[特定 PDF 长时间卡住](https://github.com/docling-project/docling/issues/2109)和[表格解析错误](https://github.com/docling-project/docling/issues/2028)实例。采用它仍需额外元数据链、模型下载、超时和降级逻辑。

本轮补做了仓库实测。对 6 篇现有 PDF，legacy PyPDFLoader 总耗时 11.96 秒、抽取 864,302 个字符；PyMuPDF4LLM 总耗时 50.07 秒、抽取 892,635 个字符，单篇耗时约为 legacy 的 3 至 14 倍。抽查 3 篇表格论文时，PyMuPDF4LLM 能产生 Markdown table rows，但每篇只得到 1 至 3 个 Markdown heading，无法单独恢复可靠章节。21 篇现有评测 PDF 中只有 6 篇带非空 title metadata，其中 2 篇还是 arXiv 标记或构建文件名。

因此 V2 Core 使用已经安装的 PyMuPDF4LLM，加确定性结构归一化和 legacy fallback。标题、作者、年份、venue 和 DOI 走独立提取优先级，低置信字段留空并交给用户修正，不能把 parser 输出直接当权威元数据。Docling 不进入 V2；GROBID 也不进入，因为 Java 或容器服务超出单用户维护预算。

表格和公式在 V2 中先解决“可检索和可定位”，不做数学推导。表格保存 Markdown、caption 和页码，公式保存可抽取文本和页码。Core 不启用 OCR，也不承诺 bbox 高亮；无文本层页面标记 `needs_ocr`，扫描件单独报告。

### 5.4 评测

ALCE 把引用质量拆成正确性和完整性，并指出即使最佳模型也经常缺少完整引用支持。RAGAS 适合无参考答案的快速回归，RAGChecker 更适合定位 retriever 和 generator 分别失败在哪里。V2 采用人工 gold、确定性检索指标、LLM judge 和人工复核四层评测，RAGAS 只能作为辅助 judge，不能取代 gold passage 与 claim-evidence 标注。

检索指标要按 paper、section、passage 三层计算 Recall@k、MRR 和 nDCG。回答指标要按原子 claim 检查支持、矛盾、引用精度、引用完整性和答案要点覆盖。Agent 还要测策略选择、补检增益、新证据率、停止原因、轮数和成本。拒答单独计算 precision、recall、F1。

原方案用 120 条混合样本同时承担 retrieval、answer、route 和拒答，并不足以支撑细分子集和“提升 5 至 8 个百分点”的统计结论。修订后分别维护 parser gold、48 条 retrieval test、24 条 answer test 和 48 条均衡 route/refusal test。paired bootstrap 区间只描述不确定性，发布门槛使用成对胜负和各目标子集退化数。Core 人工标注预算约 12 至 16 小时，Enhanced 再增加约 10 至 13 小时；项目作者两周后盲化复核 20% 样本，这只是单人一致性检查，不冒充双标注。

### 5.5 产品交互

Zotero 把 annotation 与 citation 一起写入笔记，并支持 “Show on Page” 回到原 PDF 页。V2 不做文献管理器或批注系统，但证据卡必须完成同类闭环：点击后打开对应论文和页码，并在侧栏显示原文。Core 不承诺版面坐标高亮。

Onyx 把 Search 和 Chat 分开，前者适合用户验证召回和筛选，后者负责综合回答。这个分工适合论文库。Core 只交付 Library、Paper、Search 和 Chat 四个表面；Compare、Workspace 和详细 trace 分别进入 Product 与 Enhanced。RAGFlow 强调文档解析和可视化流程，但完整部署依赖多个服务；V2 只吸收解析状态、失败原因和可观察检索过程。Paperlib 的元数据管理说明论文库需要标题、作者、年份、venue、标签与附件状态，但本项目不扩张成引用格式、同步或插件生态。

## 6. 来源目录与证据等级

以下 20 组来源均在本轮访问。论文、官方文档和项目源码承担主要判断，产品页面只用于交互参考。

| # | 来源 | 类型 | 本报告采用的证据 | 抽查范围 |
|---:|---|---|---|---|
| 1 | [Self-RAG](https://arxiv.org/abs/2310.11511) | 论文 | 按需检索和反思 token 需要训练 | 摘要、方法边界 |
| 2 | [Corrective RAG](https://arxiv.org/abs/2401.15884) | 论文 | 证据评估、纠正动作、web fallback | 摘要、方法边界 |
| 3 | [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) | 论文与代码入口 | 复杂度路由 no/single/iterative | 论文页、官方代码入口 |
| 4 | [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html) | 论文 | 递归聚类摘要与多层检索 | ICLR 摘要、实验声明 |
| 5 | [ColBERTv2](https://arxiv.org/abs/2112.01488) | 论文 | token 级 late interaction 的效果与存储代价 | 摘要、压缩结论 |
| 6 | [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | 作者工程文章 | 上下文 BM25/embedding、重排和失败率 | 方法、消融、成本说明 |
| 7 | [GraphRAG 查询文档](https://microsoft.github.io/graphrag/query/overview/) 与 [global search 源码](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/query/structured_search/global_search/search.py) | 官方文档与源码 | local/global/DRIFT 边界、map-reduce 成本统计 | 查询架构、核心 search 模块 |
| 8 | [PaperQA2 论文](https://arxiv.org/abs/2409.13740)、[工具源码](https://github.com/Future-House/paper-qa/blob/main/src/paperqa/agents/tools.py)、[Agent 测试](https://github.com/Future-House/paper-qa/blob/main/tests/test_agents.py) | 论文、源码、测试 | 科学检索 Agent 的工具边界、证据状态、恢复和成本测试 | 工具、状态、并发、恢复、测试 |
| 9 | [OpenScholar](https://www.nature.com/articles/s41586-025-10072-4) | 同行评审论文 | 多论文覆盖、引用、重排、自反馈与评测规模 | 方法、对照、限制和成本 |
| 10 | [Docling 技术报告](https://research.ibm.com/publications/docling-technical-report)、[2.114.0 PyPI](https://pypi.org/project/docling/2.114.0/)、[v2 文档源码](https://github.com/docling-project/docling/blob/main/docs/v2.md)、[PDF 卡住 issue](https://github.com/docling-project/docling/issues/2109)、[表格解析 issue](https://github.com/docling-project/docling/issues/2028) | 技术报告、官方包说明、源码文档与 Issues | 本地结构解析、元数据路线、表格、grounding、依赖和解析局限 | converter、Coming soon 列表、chunker、开放缺陷 |
| 11 | [GROBID 原理](https://grobid.readthedocs.io/en/latest/Principles/) 与 [仓库](https://github.com/grobidOrg/grobid) | 官方文档与源码仓库 | 学术 TEI、引用解析、坐标和服务成本 | 架构、处理链、部署要求 |
| 12 | [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | 论文 | 引用正确性、完整性与自动评测 | 指标定义、实验结论 |
| 13 | [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) | 论文 | 无参考答案的检索与生成评测 | 指标范围和限制 |
| 14 | [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html) | 论文 | retriever/generator 细粒度诊断 | 摘要、指标框架 |
| 15 | [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)、[AsyncSqliteSaver](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/aio/AsyncSqliteSaver)、[JsonPlusSerializer](https://reference.langchain.com/python/langgraph/checkpoint/serde/jsonplus/JsonPlusSerializer) | 官方文档与 API 参考 | thread、checkpoint、恢复调用、SQLite saver 和安全 serializer 边界 | persistence、resume、serde 构造 |
| 16 | [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming) | 官方文档 | token 来自任意节点，按 node/tag 过滤和 custom event | message/custom stream、过滤 |
| 17 | [LangChain MultiVectorRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_vector/MultiVectorRetriever) | 官方参考 | 子向量召回父文档的标准模式 | 接口和 id mapping |
| 18 | [Zotero PDF Reader](https://www.zotero.org/support/pdf_reader) | 官方产品文档 | annotation、citation、Show on Page | 阅读、笔记、原页回跳 |
| 19 | [Onyx](https://github.com/onyx-dot-app/onyx) 与 [Internal Search](https://docs.onyx.app/overview/core_features/internal_search) | 仓库与官方文档 | Search/Chat 分工、过滤与重型服务边界 | 产品流程、部署结构 |
| 20 | [RAGFlow](https://github.com/infiniflow/ragflow) 与 [Paperlib](https://github.com/Future-Scholars/paperlib) | 开源仓库 | 文档流程可视化、论文元数据管理、部署复杂度 | 目录、核心能力、依赖与测试入口 |

## 7. 代表性方案对比

| 方案 | 最强项 | 适用条件 | 与当前项目的关系 | 决定 |
|---|---|---|---|---|
| Anthropic Contextual Retrieval | LLM 生成 chunk-specific context | 接受索引期模型成本和上下文漂移 | 提供设计启发，但当前实现不是该方法 | V2 不采用 |
| Metadata-prefixed retrieval | 检索表示加入已验证元数据 | 有稳定 title、section 和 passage schema | 低成本解决部分语境丢失 | 采用，可消融 |
| Parent-child / multi-vector | 小块匹配，父级返回 | parent id 与原文映射稳定 | 可复用现有树，但要重做检索职责 | 采用 |
| Adaptive-RAG | 按复杂度选择成本不同的策略 | 有路由评测和多种可靠策略 | 适合 fixed 与 bounded adaptive 分流 | 采用原则 |
| CRAG | 低质量召回后触发纠正 | 有证据评分和可用补救动作 | web fallback 不符合私有论文库 | 采用评分与补检，拒绝 web |
| Self-RAG | 模型内生的检索与自我批评 | 能训练专用模型与 reflection token | 当前通用模型无法复现 | 不采用完整方法 |
| PaperQA2 | 科学搜索工具与证据状态清晰 | 可访问外部论文和高预算模型 | 工具边界和恢复测试可迁移 | 采用工程模式 |
| OpenScholar | 大规模科学语料、专用 retriever、自反馈 | 大数据、训练和算力 | 是研究上限，不是本地产品模板 | 采用评测思想 |
| RAPTOR | 长文档多层摘要 | 全局与多步问题多，索引预算足 | 当前 25 到数百篇库收益未知 | 暂缓 |
| GraphRAG | 实体关系与全库主题总结 | 关系问题高频且可评测 | 索引和查询成本过高 | 拒绝 V2 |
| ColBERTv2 | token 级细粒度匹配 | 能接受多向量存储和模型维护 | 先验证简单混合检索上限 | 暂缓 |
| PyMuPDF4LLM | 本地文本和表格 Markdown | 接受离线索引耗时，配结构归一化 | 已在仓库依赖内，实测文本和表格优于 legacy | 采用，legacy fallback |
| Docling | 本地结构、表格、公式、坐标 | 接受较重依赖与模型下载 | 元数据仍需另建链路，仓库无收益实测 | 拒绝 V2 默认化 |
| GROBID | 学术元数据、参考文献与 TEI | 能维护 Java/Docker 服务 | 超出单用户本地维护预算 | 拒绝 V2 |

不同来源的结论并不完全一致。RAPTOR 和 GraphRAG 强调更高层摘要对全局问题的价值，Contextual Retrieval 强调模型生成的局部 chunk context。它们适用问题不同：前两者针对全局主题和跨片段推理，后者针对局部证据召回。个人论文库先用 metadata prefix 和 section expansion 建立低成本基线，不借用 Contextual Retrieval 的实验数字。PaperQA2 更偏引用精度，OpenScholar 的对照显示它可能牺牲覆盖；V2 因而把精度和覆盖拆开报告。

### 7.1 候选技术逐项筛选

| 候选 | 失败场景与当前缺口 | 论文库适配和改动 | 基线、指标与预期 | 延迟、维护与退出 | 面试可讲性 |
|---|---|---|---|---|---|
| Metadata-prefixed retrieval | 局部段落缺论文、章节语境；当前叶节点只有正文 | 适合，改 passage schema、BM25、embedding、rerank input | B1 对 B2，消融 prefix；看 passage Recall/nDCG，不预设提升幅度 | 增加索引文本和少量 rerank token；`METADATA_PREFIX=false` 可关 | 能准确区分于 Anthropic 方法，并展示 schema、排名和消融 |
| Parent-child / multi-vector | 命中短句后缺实验条件；当前父节点不进向量索引 | 适合，增加 section/title/caption retrieval unit 与 parent id | 移除 section unit/neighbor expansion；看 Recall、完整性和上下文 token | 增加 unit 数与扩展；扩展窗口设 0 即移除 | 能解释“小块匹配、父级阅读”和当前实现差异 |
| RRF hybrid | dense/BM25 分数不可比；当前 min-max 对平分和查询变化敏感 | 适合，替换 fusion，不改存储 | min-max 对 RRF；看 nDCG、稳定性和跨语言子集 | 计算开销可忽略；`RETRIEVAL_PIPELINE=v1` 回退 | 公式、源码和排名例子都清楚 |
| FlashRank rerank | 初召回噪声进入上下文；当前小样本收益待验证 | 适合本地单机，保留现实现 | no-rerank 消融；看 nDCG、citation completeness、p95 | 运行时增加一次模型推理；不可用时跳过 | 可展示候选前后排名和延迟权衡 |
| 选择式上下文压缩 | 整页或邻居占满预算；当前首段可突破预算 | 适合，重写 packer，quote 不改写 | no-expansion、不同 token budget；看完整性、噪声和 token | 不增加 LLM；打包器可切回 top-k | 能说明为什么不做 LLM 摘要证据 |
| Adaptive-RAG 原则 | 简单题过度规划、复杂题单次检索不足；当前只有 retrieve/direct/oos | 适合，route 增加 fixed/adaptive | all-fixed、all-adaptive、auto；看策略准确率、覆盖和 token | 多一次 route，复杂题多轮；`ANSWER_STRATEGY=fixed` 关闭 | 可从路由、状态和分层实验说明 |
| CRAG 原则 | 低质量证据直接生成；当前没有 coverage 与纠正动作 | 适合库内补检，不采用 web | fixed 对 bounded corrective；看覆盖、拒答、每轮新增证据 | 最多两轮和 evidence judge；预算开关可关 | 能展示缺失维度、补检动作和停止原因 |
| Self-RAG | 模型需要判断何时检索与反思；当前通用模型只靠 prompt | 完整方法不适合，需要训练 reflection token 模型 | 不进入实现实验，只把 fixed/adaptive 作为替代对照 | 训练、部署和数据成本高，拒绝后无残留 | 可以准确说明论文方法与本项目没有实现它 |
| RAPTOR | 长论文全局、多步问题可能漏召回；当前层级只有页树 | 当前收益未知，需聚类、摘要和摘要版本 | 触发后对 B2 做 recursive-summary 消融；看 evolution 子集 Recall/coverage | 索引 LLM 成本和维护高，可作为独立 index feature | 能讲原理，但没有实验前不列为项目亮点 |
| GraphRAG | 全库主题和实体关系问题；当前没有关系层 | 当前个人库不适合，要增加实体、关系、社区和 map-reduce | 只有关系 gold 达到触发条件才与 B2 对比 | 新索引管线、LLM 成本高，可独立服务但本期拒绝 | 能解释适用条件和拒绝依据，不冒充已实现 |
| ColBERTv2 | 术语与长段落需要 token 级匹配；当前单向量可能丢细节 | 先验证简单 hybrid，上限不足再引入模型与多向量存储 | 触发后与 B2 比术语/公式子集 Recall 和磁盘 | 模型、运行时和索引空间增加，独立 retriever 可移除 | 原理可讲，当前没有本项目实验 |
| PyMuPDF4LLM | 章节、表格和阅读顺序丢失；当前 PDF section 等于 page | 已安装，改 parser adapter、normalizer、schema 和 catalog | legacy 对 PyMuPDF4LLM，12 篇 parser gold；看 section/table F1、页码和耗时 | 实测慢 3 至 14 倍；legacy fallback | 能展示 parser output、质量门和 PDF 回跳 |
| Docling | 需要更强 layout、公式和坐标 | 能解决部分问题，但元数据仍需独立处理 | V2 不做实现实验，现阶段只保留来源证据 | 依赖、模型和超时维护高，拒绝 V2 后无残留 | 能说明官方能力边界与为什么没默认采用 |
| GROBID | 需要参考文献、TEI 和引用上下文；当前没有 | 能解决但超出首版故事，要新增 Java/Docker 服务 | 若未来做引用关系图，再比较 metadata/reference F1 | 常驻服务和模型维护高，本期零接入 | 能解释它更强的领域与为什么没选 |

## 8. 技术采用、暂缓与拒绝清单

### 8.1 采用

| 技术 | 失败场景 | 当前为何不足 | 主要改动 | 实验与目标 | 代价与可移除性 |
|---|---|---|---|---|---|
| PyMuPDF4LLM + 结构归一化 | 页面顺序、章节和表格丢失 | 当前 PDF section 等于 page | parser adapter、normalizer、paper/section/passage、parser version | 12 篇解析 gold，section/table F1、页码准确率 | 实测较慢；配置可切回 legacy |
| 稳定证据地址 | 文件移动或重建后引用失效 | `doc_id` 依赖路径，passage UUID 随机 | content hash、paper id、version、source span | 重建后 evidence id 一致率 100% | 多一层迁移；旧索引可只读保留 |
| Metadata-prefixed passage retrieval | 术语段落脱离论文与章节语境 | 叶节点只有局部文本 | `retrieval_text` 与 `quote_text` 分离 | remove-prefix 消融，Recall/nDCG | 索引文本变长；单开关关闭 |
| BM25 + dense + RRF + rerank | 异构分数难校准、专业术语漏召回 | `.split()` 与 min-max fusion | 中英 tokenizer、RRF、候选去重、rerank | remove-sparse/dense/rerank 消融 | 多一次 rerank 延迟；各阶段独立关闭 |
| 受控 parent/neighbor expansion | 命中一句但缺实验条件 | 当前扩展与层级召回混合 | 命中后按 section/order 扩展并重新预算 | no-expansion 消融，完整性与 token | 上下文增加；配置可为 0 |
| 有界 adaptive evidence loop | 比较题缺论文或维度仍直接生成 | 单工具 ReAct 无证据覆盖状态 | strategy、evidence ledger、coverage、max rounds、stop reason | fixed 对 adaptive，覆盖与成本 | 只对复杂题启用；可全局切 fixed |
| claim 级引用与拒答 | 引用存在但不支持具体结论 | 当前 groundedness 粗粒度 | claims、evidence ids、validator、unsupported claims | citation correctness/completeness、refusal F1 | 增加一次校验；可退回引用段落模式 |
| SQLite catalog、job/run worker、checkpoint | 重启丢任务和图状态 | 内存任务与 InMemorySaver | app DB、lease worker、独立 checkpoint DB、启动扫描 | kill/restart、重复请求和并发测试 | 无新服务；Enhanced 可整体关闭 |
| Compact GraphState | 完整候选和证据在 checkpoint 重复写入 | 当前状态包含 messages、evidence 和 packed context | 图中只留 ID、预算、覆盖率和 final result | checkpoint 大小、恢复一致性 | 详细记录移到 DB，可独立保留或清理 |

### 8.2 暂缓

| 技术 | 暂缓依据 | 重新评估触发条件 |
|---|---|---|
| RAPTOR | 摘要索引成本高，现有 hierarchy 尚未证明价值 | 全库演进类 gold 上，V2 fixed/adaptive Recall@20 低于 0.75，且缺失来自高层概念 |
| ColBERTv2 | 新模型、存储和检索运行时增加维护面 | 混合检索在术语与公式问题上连续两轮实验无提升，且 reranker 无法补救 |
| 引用关系图 | 参考文献解析与实体消歧尚无数据 | 用户故事中出现可量化的引用链追踪需求，并完成 50 篇 gold |
| OCR 全面开启 | 数字 PDF 会增加无必要延迟和误差 | 页面无文本层时自动启用，扫描件单独报告 |
| 外部学术搜索 | 产品边界是个人已读库 | 后续明确扩展成发现产品，并单独处理版权、去重和来源可信度 |

### 8.3 拒绝

| 方案 | 拒绝原因 | 轻量替代 |
|---|---|---|
| GraphRAG | 实体抽取、社区构建、摘要更新和 map-reduce 超出当前场景与预算 | paper/section metadata、受控查询分解和多论文证据表 |
| 多 Agent 协作 | 没有独立角色和不可并行的工具价值，增加轨迹和失败组合 | 单图中并行执行确定性子查询 |
| Docling 默认 parser | 官方元数据仍是后续能力，依赖和模型更重，仓库没有对照收益 | PyMuPDF4LLM + deterministic normalizer + legacy fallback |
| GROBID 常驻服务 | Java/Docker 服务与模型维护不符合单机简洁部署 | PyMuPDF4LLM + legacy parser |
| Celery/Redis/Kafka | 单用户索引只需持久任务和单写者 | SQLite job queue + 进程内 worker |
| PostgreSQL/专用向量数据库 | 个人库规模下没有证据表明 FAISS 不够 | SQLite metadata + immutable FAISS/BM25 versions |
| Self-RAG 训练 | 没有训练数据、GPU 和专用模型维护目标 | 结构化证据评分和有界补检 |
| 自动 web fallback | 越过个人库边界，答案来源不可控 | 明确提示库内证据不足 |

## 9. 推荐产品定位和使用故事

目标用户是已经下载并读过一批论文，希望日后快速找回证据、比较方法和整理研究结论的个人用户。产品不负责替用户发现全网论文、管理引用格式或写 LaTeX。

完整使用故事如下：

1. 用户批量导入 PDF，任务列表展示哈希去重、解析、切分、embedding、索引、激活等阶段。失败项显示页码、原因和重试按钮。
2. 论文库展示标题、作者、年份、venue、标签、解析状态和索引版本。用户可以按元数据筛选，也可以搜索原文。
3. 搜索页返回论文、章节、页码和原句。点击证据后，阅读器打开对应页，侧栏保留原文；用户可以把证据保存到工作区。
4. 单篇页面提供概要、章节目录和对当前论文提问。回答只能引用该论文，并标明缺少证据的部分。
5. Product 阶段允许用户选中 2 到 5 篇论文进入比较工作区，指定 1 到 6 个维度。系统先生成 query plan，再显示每个维度的证据覆盖。
6. 普通模式只显示进度、答案和证据。调试模式展示策略、子查询、召回候选、RRF 排名、重排、证据评分、补检轮次、停止原因、延迟、token 和估算成本。
7. 会话、答案、证据集合和比较表可以命名保存。再次打开时仍能回到当时的 index version；索引更新后可以主动选择重跑。

## 10. 面试价值

这条主线能从三个层面讲清楚：

- **原理**，为什么学术段落需要文档与章节上下文，为什么 RRF 比逐查询 min-max 更稳，为什么引用准确率与覆盖率要分开，为什么 Agent 只处理复杂或证据不足的问题。
- **源码**，稳定 evidence id、parser adapter、immutable index version、检索各阶段、compact GraphState、run worker、停止条件、安全 checkpoint 和自定义 SSE 事件都有明确模块边界。
- **实验**，先在 retrieval 数据集上比较 flat、flat_rerank 和 V2 fixed，再在独立 answer/route 数据集上比较 fixed 与 adaptive，并分别移除 metadata prefix、sparse/dense、rerank、neighbor expansion 和 evidence loop。

简历表达应使用可复现数字，不写“实现先进 Agentic RAG”。合适的模板是：

> 面向个人论文库设计 local-first 科研助手，构建可回跳 PDF 原页的稳定证据索引；通过 metadata-prefixed BM25+dense、RRF 与重排，在 48 条固定检索测试上报告 passage Recall@10、成对坏例和 p95 延迟。Enhanced 完成后，再在独立 answer 与 route 数据集上报告引用正确性、完整性、拒答 F1、恢复成功率和 token 成本。

当前阶段还不能填写提升百分比。只有 V2 评测运行后，才能把实测数字写进简历和演示。
