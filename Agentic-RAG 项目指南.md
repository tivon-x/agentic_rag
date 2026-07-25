# Agentic RAG 项目指南

这份指南的目标是帮你在 AI 应用开发面试里把这个项目讲清楚。重点不是背 LangChain、LangGraph、FAISS 这些名字，而是把系统链路、设计取舍、线上风险和后续演进讲成一条能追问下去的因果链。

## 开场怎么讲

30 秒版本：

> 这是一个面向本地知识库问答的 Agentic RAG 项目。我没有把它做成简单的 PDF 聊天，而是把问答拆成了知识库边界注入、路由决策、查询规划、问题改写、混合检索、重排、上下文打包、证据聚合和结构化回答。系统用 LangGraph 表达多分支流程，用 LangChain agent 执行检索子任务，用 BM25+FAISS 做混合召回，并提供 Gradio、FastAPI、SSE 前端和离线评测，方便定位问题到底出在路由、检索还是回答。

90 秒版本：

> 用户问题进来后，系统先把 `corpus_profile` 注入状态，明确这套知识库覆盖什么、不覆盖什么。然后 `summarize_history` 压缩历史上下文，`decide_retrieval` 判断是直接回答、超范围拒答，还是进入检索。进入检索后，`plan_query` 会生成 intent、subqueries 和偏好的文档粒度，`rewrite_query` 把问题改写成最多 3 个自包含子查询，再通过 LangGraph 的 `Send` 并发派发给 research-search 子代理。
>
> 子代理不是直接回答，它通过 `search_relevant_chunks` 工具调用检索器。检索器内部做 FAISS 向量召回和 BM25 词法召回，再做去重、规则加权、可选 Flashrank 重排和 token budget 下的 context packing。最后 `aggregate_answers` 不直接拼 chunk，而是把证据组织成 `GroundedAnswer`，包括 answer、reasoning_summary、evidence、confidence 和 limitations。UI 里能看到 route、query plan、rerank、packed context 和 citation，所以这个项目可解释、可调试，也能用 eval runner 做 routing、retrieval、answer 三类评测。

一句话定位：

> 这个项目的价值在于把普通 RAG 的黑盒链路拆成了可路由、可观察、可评测的多阶段系统。

## 面试官真正想听什么

面经里最关键的考察点是两条线：

- 这个系统上线会不会出事。
- 你知不知道自己用了什么，也知道自己没用什么。

所以介绍项目时不要按技术栈报菜名。更好的顺序是：

1. 用户输入是什么，系统输出是什么。
2. 系统如何判断是否该检索。
3. 检索问题如何拆、如何改写。
4. 工具调用在哪里发生，结果怎么回到状态里。
5. 最终回答如何和证据绑定。
6. 失败时如何兜底。
7. 如何评测，如何定位问题。
8. 当前短板是什么，下一步怎么补。

这套顺序能让面试官看到你是在讲系统，不是在朗读依赖列表。

## 项目全链路

### 入口层

核心入口在 `main.py`：

- `index`：构建索引，支持 `flat` 和 `hierarchical`。
- `ask`：命令行问答，在线模式走 LangGraph，离线模式走 retrieval-only answer。
- `ui`：启动 Gradio 工作台。
- `api`：启动 FastAPI 服务。
- `eval`：运行 routing、retrieval、answer 评测。

FastAPI 入口在 `api/main.py`，它挂了四组路由：

- `health`
- `corpus`
- `chat`
- `indexing`

前端在 `web/`，是 Next.js + React。聊天页先调用 `/api/chat` 创建或追加消息，再用 `/api/chat/stream` 建立 SSE 连接接收 token、citations 和 done 事件。

面试里可以这样说：

> 这个项目有两套交互入口。Gradio 更像调试工作台，可以看 route、plan、rerank 和 citation；FastAPI + Next 前端更接近产品化入口，支持会话持久化、文件上传索引和 SSE 流式回答。

### 索引构建链路

flat 模式：

1. `Indexer.index()` 接收文件或目录。
2. `data_processor` 读取 PDF、TXT、Markdown。
3. `Chunker` 做分块，默认是 `RecursiveChunker`。
4. chunk 写入 FAISS。
5. 从向量库取回全部文档，构建 BM25 bundle。
6. FAISS 和 `bm25.pkl` 持久化到索引目录。

hierarchical 模式：

1. 根据文件后缀选择 parser。
2. Markdown 会按标题解析成 `document -> section -> paragraph`。
3. PDF 会按 `document -> page section -> paragraph` 建树。
4. `HierarchicalIndexBuilder` 给 leaf node 生成 embedding。
5. 父节点可以用 mean pooling 聚合子节点 embedding。
6. `JsonNodeStore` 保存 `nodes.jsonl` 和 `doc_trees.json`。
7. 指定 leaf node type 的节点转成 `Document` 写入 FAISS。

这里最值得讲的是：

> 分层索引不是为了炫技。它解决的是“检索粒度”和“回答粒度”不一致的问题。检索时 paragraph 粒度更准，回答时可能需要 section 或相邻 sibling 才有完整上下文。

### 问答执行链路

主图定义在 `agent/graph.py`：

```text
START
-> inject_corpus_profile
-> summarize_history
-> decide_retrieval
-> direct_answer / out_of_scope_answer / plan_query
-> rewrite_query
-> agent（多个子查询并发派发）
-> aggregate_answers
-> END
```

关键状态定义在 `agent/states.py`：

- `routingDecision`
- `routingReason`
- `corpusProfile`
- `corpusProfileData`
- `conversation_summary`
- `queryPlan`
- `rewrittenQuestions`
- `retrievalEvidence`
- `packedContexts`
- `evidenceGroups`
- `groundedAnswer`

`agent_answers`、`retrievalEvidence`、`packedContexts`、`evidenceGroups` 使用了自定义 reducer，正常情况下追加，遇到 `__reset__` 时清空。这能避免多子查询并发写 state 时互相覆盖。

面试里可以这样说：

> LangGraph 在这里的价值不是“能连节点”，而是它给了一个显式状态机。每个节点只负责一件事，每个中间产物都能进 state，后面 UI 和评测才能把链路拆开看。

## 每个关键模块怎么讲

### `corpus_profile` 是知识库边界，不是装饰字段

`core/corpus_profile.py` 定义了知识库画像：

- name
- summary
- coverage
- non_coverage
- usage_notes
- recommended_questions
- forbidden_questions
- domain_keywords
- primary_entities
- preferred_answer_style

它参与了四处逻辑：

- `decide_retrieval`：先用 profile 做规则级越界判断，再交给 LLM 做结构化路由。
- `plan_query`：把匹配到的领域关键词和实体加进 query plan。
- `rewrite_query`：扩展子查询，提高召回概率。
- `aggregate_answers`：把 preferred answer style 传给最终聚合 prompt。

可以这样回答：

> 很多 RAG 项目把知识库描述当 UI 文案，但这里它是参与运行的先验。它告诉系统这套语料能回答什么、不能回答什么、哪些实体重要、答案应该怎么组织。

### 路由决策

`decide_retrieval` 输出三类结果：

- `retrieve`
- `direct_answer`
- `out_of_scope`

先用 `analyze_corpus_profile_match()` 判断 forbidden 和 non-coverage，再用 `RetrievalDecision` 做结构化 LLM 输出。结构化输出失败时默认走 retrieval。

面试里可以强调：

> 失败时默认检索是保守策略。宁可让系统从知识库里找证据，也不要在路由解析失败时直接编一个答案。

### 查询规划和改写

`QueryPlan` 包含：

- `intent`：fact、summary、compare、multi_hop、definition
- `subqueries`：1 到 3 个子查询
- `preferred_node_types`：document、section、paragraph

`rewrite_query` 会把子查询再改写成自包含问题，并结合 profile 扩展关键词。

可以这样说：

> query plan 的作用是把用户的自然语言问题转成检索策略。比如 summary 问题更偏 section，fact 问题更偏 paragraph；multi-hop 问题需要拆成多个子查询，而不是只拿原问题去搜一次。

### 子代理和工具调用

`agent/research_search_agent.py` 通过 LangChain `create_agent` 创建 research-search agent。这个 agent 有几个 middleware：

- `QueryPlanMiddleware`：把主图生成的 query plan 传给工具层。
- `SummarizationMiddleware`：上下文接近 token 上限时压缩历史。
- `FallbackMiddleware`：限制模型迭代次数和工具调用次数。
- `EvidenceCaptureMiddleware`：把工具返回的 artifact 写回 graph state。
- `collect_answer`：在 agent 结束后收集最终回答。

工具只有一个：

- `search_relevant_chunks`

工具返回两部分：

- 给模型看的序列化文本。
- 给系统看的 structured artifact，包括 passages、evidence、packed_context、debug。

这个设计很适合面试：

> 我没有只把工具结果作为纯文本塞回模型，而是把工具返回拆成 content 和 artifact。content 给模型推理，artifact 给系统做 citation、debug 和最终聚合。这是把 LLM 消费和工程系统消费分开。

### 检索器设计

`FusionRetriever.retrieve()` 的流程是：

```text
normalize_query_plan
-> retrieve_candidates
-> dedupe_candidates
-> rerank_candidates
-> pack_context
```

候选召回有两类：

- FAISS 向量召回。
- BM25 词法召回。

融合公式：

```text
score = alpha * vector_score + (1 - alpha) * bm25_score
```

如果是 hierarchical 模式，并且 query plan 偏好 section 或 document，还会从 `node_store` 里做结构化节点召回，用 lexical overlap 补充候选。

去重策略：

- 优先看 `node_id`。
- 再看标准化文本。
- 相同文本但不同节点会合并分数。
- 相邻 sibling 不强行去掉，因为后面可能需要窗口合并。

重排策略：

- 标题匹配加分。
- node type 匹配加分。
- corpus profile 关键词加分。
- primary entity 加分。
- non-coverage 命中且和 query 无关时惩罚。
- 可选 Flashrank 精排。

上下文打包：

- 受 `token_budget` 控制。
- 按分数优先选候选。
- summary 类问题可能把 paragraph 扩展成 parent section。
- paragraph 命中时可能合并前后 sibling，形成连续窗口。

可以这样回答“为什么要 hybrid search”：

> 技术文档里有很多精确术语、缩写、模型名和函数名。向量检索擅长语义相似，但可能漏精确词；BM25 擅长精确词，但不理解同义表达。混合检索能降低单一路线的失误。

可以这样回答“为什么要 rerank”：

> 第一阶段召回追求快和全，排序目标比较粗。rerank 追求候选之间的细粒度相关性。这个项目先用规则增强补充领域先验，再用 Flashrank 做轻量模型重排，成本比让大模型处理大量候选更可控。

### Grounded Answer

最终回答结构是 `GroundedAnswer`：

- `answer`
- `reasoning_summary`
- `evidence`
- `confidence`
- `limitations`

`aggregate_answers` 会把 `evidenceGroups`、`packedContexts`、`retrievalEvidence` 一起交给聚合模型。如果结构化聚合失败，会退回到 extractive fallback，从已捕获证据里拼出保守答案。

面试里可以这样说：

> 我更关心答案和证据能不能绑定，而不是答案看起来流畅。结构化回答让 UI 能展示引用，评测能计算 citation precision，也能在证据不足时显式暴露 limitations。

### 可观测性

Gradio 调试面板展示：

- route decision
- query plan
- rewritten queries
- retrieved candidates
- reranked top passages
- packed context
- citation
- 命中文档树位置

FastAPI + Next 前端展示：

- SSE token 流。
- citations 事件。
- SQLite 会话持久化。
- indexing job 状态查询。

可以这样说：

> RAG 出错不能只看最后答案。这个项目把中间状态暴露出来后，可以判断是路由错、query plan 错、召回没命中、rerank 排错，还是最终聚合没用好证据。

### 评测体系

`evals/runner.py` 把评测拆成三类：

- routing：看 route accuracy。
- retrieval：看 recall@k、MRR、nDCG、redundancy rate。
- answer：看 groundedness、citation precision、answer completeness、hallucination rate。

支持三个 variant：

- `baseline_flat`
- `flat_rerank`
- `hierarchical`

当前基线报告是离线模式，使用 `FakeEmbeddings`，LLM answer generation disabled。结果里 `flat_rerank` 在 retrieval 指标上最好，`hierarchical` 暂时没有跑赢。

面试里要讲清楚：

> 当前评测不能拿来吹生成质量，因为 answer 是 offline extractive fallback。它更适合作为回归基线，帮我比较检索配置和防止改动后指标倒退。

## LangChain、LangGraph、DeepAgents 怎么讲

### LangChain 在项目里的角色

项目里 LangChain 主要做三件事：

- 统一模型和消息接口。
- 提供 `create_agent` 的工具调用循环。
- 提供工具、middleware、structured output 等 agent 基础能力。

按 LangChain 官方文档，`create_agent` 是基于 LangGraph 构建的 graph-based agent runtime，agent 会在模型节点、工具节点和 middleware 之间循环，直到输出最终结果或达到迭代上限。

这个项目的用法是：

- 主流程不用 LangChain chain，而是自己写 LangGraph。
- 子任务用 LangChain `create_agent`，让它负责工具调用循环。
- 最终结构化聚合用 Pydantic schema 约束输出。

面试里可以这样说：

> 我没有把所有流程都塞进一个 LangChain agent。主流程需要清晰的业务分支，所以用 LangGraph；子查询检索适合工具调用循环，所以用 `create_agent`。

### LangGraph 在项目里的角色

LangGraph 解决的是状态、分支和可恢复执行的问题。官方文档里，graph 配置 checkpointer 后，每一步会保存 state snapshot，支持 human-in-the-loop、memory、time travel 和 fault tolerance。

这个项目当前用了 `InMemorySaver`：

- 本地开发方便。
- 可以用 `thread_id` 区分运行线程。
- 不适合生产持久会话。
- 进程重启后 checkpoint 丢失。

可以这样回答“为什么不用一条 chain”：

> 这里有直接回答、超范围拒答、检索回答三条分支，还有多子查询并发派发和证据聚合。一条 chain 会把控制流藏在代码里，调试时只能看最终结果。LangGraph 让每一步的输入输出都进入 state，UI 和 eval 才能复用这些中间产物。

可以这样回答“是不是过度设计”：

> 如果项目只是“模型调用、检索、再模型调用”三步，LangGraph 可能偏重。但这里的图节点对应真实业务分支：是否检索、是否越界、如何拆 query、如何并发检索、如何聚合证据。它不是为了框架而框架。

### DeepAgents 怎么对照讲

这个项目没有直接使用 `deepagents` 包，这一点面试时不要说错。

DeepAgents 官方定位是 agent harness，内置任务规划、文件系统上下文管理、subagent spawning 和长期记忆，底层仍基于 LangChain 和 LangGraph。它适合更复杂的多步骤任务，尤其是需要子代理隔离上下文、文件系统状态和长期任务管理的场景。

本项目和 DeepAgents 的关系可以这样讲：

> 我这个项目没有直接引入 DeepAgents，但已经有一些相似的工程思想：主图负责高层编排，research-search agent 负责子任务；工具结果通过 artifact 回传，避免主模型只拿到纯文本；SummarizationMiddleware 控制上下文膨胀。不同的是，DeepAgents 还内置 todo、filesystem、subagent spawning 和长期记忆，而这个项目目前只需要检索型子代理，所以我选择了更轻的 LangGraph + LangChain agent 组合。

如果面试官追问“以后会不会引入 DeepAgents”，可以这样答：

> 如果项目从知识库问答升级到长程研究任务，比如需要读大量文件、维护中间草稿、拆多个研究子代理、跨轮保留任务状态，那 DeepAgents 更合适。当前项目的任务边界是 RAG 问答，检索管道已经比较明确，引入 DeepAgents 反而可能增加不必要的抽象。

## 高频追问与回答

### 1. 为什么不是普通 RAG？

普通 RAG 是固定流水线：query -> retrieve -> prompt -> answer。这个项目的问题类型更复杂，需要先判断：

- 是否该检索。
- 是否超出知识库边界。
- 是否需要拆成多个子查询。
- 检索粒度该偏 paragraph 还是 section。
- 最终答案是否有证据支撑。

所以它更像 Agentic RAG，用 agent/graph 做动态决策和证据聚合。

### 2. 为什么主图和子代理要分开？

主图负责稳定的业务流程，子代理负责有不确定性的工具调用。这样边界更清楚：

- 主图：路由、规划、派发、聚合。
- 子代理：围绕某个子查询检索和生成中间答案。
- 检索器：负责确定性候选处理。

如果全塞进一个 agent，route、retrieval debug、citation 都会混在消息历史里，后面很难评测。

### 3. 这是 ReAct 还是 Plan-Execute？

更准确地说是混合形态：

- 主图先做 plan 和 rewrite，接近 Plan-Execute。
- research-search agent 内部是工具调用循环，接近 ReAct。
- 检索器内部是确定性 pipeline，不交给模型自由决定。

可以这样说：

> 我没有让模型在所有层面自由 ReAct。高层控制流用图固定下来，检索内部用确定性代码，只有子查询工具调用留给 agent 循环。这样可控性更强。

### 4. 如何限制 agent 循环失控？

项目里有几层限制：

- `MAX_ITERATIONS`
- `MAX_TOOL_CALLS`
- `MAX_CONTEXT_TOKENS`
- `KEEP_MESSAGES`
- `FallbackMiddleware`
- `SummarizationMiddleware`

如果达到上限，fallback 会基于已有消息给出保守答案。

更进一步可以补：

- 检测同一工具同参重复调用。
- 给工具设置超时。
- 对写操作增加幂等键。
- 把异常分类为网络、配置、解析、业务越界。

### 5. checkpoint 怎么讲？

项目现在用了 `InMemorySaver`，它能支持本地线程级状态，但不是生产级持久化。

如果面试官问 checkpoint，要分清三件事：

- LangGraph checkpoint 是图运行状态快照。
- API 的 SQLite chat session 是产品层会话历史。
- 当前项目没有把 checkpoint 持久化到 SQLite/Postgres。

可以这样答：

> 当前版本的 checkpoint 是开发态选择。真正生产化我会换成 SQLite/Postgres checkpointer，并明确 thread_id 和业务 session_id 的映射。外部副作用也要有幂等设计，避免 replay 或 resume 时重复执行。

### 6. memory 怎么讲？

当前项目有短期上下文处理：

- 对话历史进入 messages。
- `summarize_history` 对最近历史做摘要。
- 子代理有 `SummarizationMiddleware`。
- API 用 SQLite 保存聊天消息。

但它没有做长期用户记忆。

可以这样说：

> 这个项目的 memory 主要是会话内上下文和图状态，不是跨用户长期记忆。长期记忆需要单独的 namespace、写入触发、删除权和隐私策略，不能把每轮对话都塞进向量库。

### 7. 结构化输出失败怎么办？

项目里多个节点都有 fallback：

- route 失败默认 retrieve。
- plan 失败默认 fact + 原问题 + paragraph。
- rewrite 失败默认原子查询。
- aggregate 失败从 evidence 里做 extractive fallback。
- out_of_scope 失败用固定模板。

可以这样说：

> 结构化输出能提高可控性，但不能假设永远成功。关键节点都有保守默认值，保证系统能退化运行。

### 8. 为什么要把检索 pipeline 放在 `FusionRetriever`，而不是拆成 LangGraph 节点？

这是一个很好的取舍点。

可以答：

> 检索内部的 retrieve、dedupe、rerank、pack 是同步确定性数据处理，不需要单独变成图节点。把它们放在 `FusionRetriever.retrieve()` 里，调用成本更低，接口更简单。真正需要图表达的是业务分支和多子查询派发。

### 9. hierarchical 为什么没有跑赢 flat_rerank？

可以答：

> 分层索引带来的收益不一定直接体现在当前离线检索指标上。现在 embedding 入库主要还是 leaf node，父节点更多用于上下文扩展和路径展示。当前评测集也偏检索命中，不一定能体现 section-level packing 对最终答案完整性的收益。这个结果说明 hierarchical 还需要继续调权重、结构化节点召回和 packing 策略，而不是说明分层方向没价值。

### 10. 如何定位一次坏答案？

按链路排查：

1. 看 route decision，确认是否走错分支。
2. 看 query plan，确认 intent 和 preferred node types 是否合理。
3. 看 rewritten queries，确认是否改写偏题。
4. 看 retrieved candidates，确认召回是否覆盖 gold source。
5. 看 reranked top passages，确认排序是否把关键证据排上来。
6. 看 packed context，确认是否被 token budget 丢掉。
7. 看 groundedAnswer.evidence，确认最终回答是否引用了正确证据。

这比只说“调 prompt”更有工程说服力。

### 11. 如何优化准确率？

按层回答：

- 数据侧：补充语料、清洗 PDF、保留标题层级、维护 corpus profile。
- 索引侧：优化 chunk、中文分词、leaf node type、增量索引。
- 检索侧：调 `alpha`、`k`、`fetch_k`、Flashrank top_n、结构化节点召回。
- 规划侧：优化 intent 和 subquery 生成。
- 生成侧：改进 grounded aggregation prompt 和 citation 校验。
- 评测侧：难例入库，按 routing/retrieval/answer 拆指标。

### 12. 如何生产化？

可以分四层说：

- 存储层：持久 checkpointer、pgvector 或 sqlite-vec、可用的 sqlite node store、索引版本管理。
- 运行层：trace id、节点级耗时、工具错误率、token 成本、SSE 断连恢复、队列化索引任务。
- 安全层：租户隔离、文件扫描、上传大小限制、RBAC、敏感字段脱敏。
- 质量层：线上 bad case 回流、人工抽检、离线集版本化、灰度发布。

### 13. 如果问 MCP、Skill、Tool 的区别

结合项目讲，不要泛泛背概念：

- Tool：项目里的 `search_relevant_chunks`，是 agent 可调用的原子能力。
- Skill：高频任务的工作方式，比如“如何构建某类知识库”“如何做检索调试”，更像流程、约束和工具使用说明。
- MCP：宿主和外部能力之间的协议层，适合把搜索、数据库、文件系统等工具以统一 schema 暴露给 agent。

可以这样说：

> 当前项目只用了 LangChain Tool，没有接 MCP。后续如果要接企业内部系统，我会优先把权限、schema、审计和超时定义清楚，再考虑 MCP 接入，而不是把一百个工具一次性塞给模型。

### 14. 如果问流式响应

项目里 FastAPI 的 `/chat/stream` 使用 SSE：

- 先根据 session_id 取聊天历史。
- 在线模式用 `graph.astream_events()` 监听 `on_chat_model_stream`。
- 每个 chunk 发 `token` 事件。
- 最后发 `citations` 和 `done`。
- 离线模式一次性返回 retrieval-only answer。

可以补充生产注意点：

- Nginx 要关闭代理缓冲。
- 需要心跳避免连接长时间空闲。
- 前端要处理断线、重复 done、用户取消。
- 后端要区分模型流式失败和图执行失败。

## 可以主动讲的源码级细节

- `GraphState` 用 camelCase，符合项目约定，也方便 LangGraph state 暴露到 UI。
- `route_after_rewrite` 返回 `list[Send]`，每个 rewritten query 派发一个 research-search agent。
- `EvidenceCaptureMiddleware` 用 `Command(update=...)` 把工具 artifact 写回 state。
- `ToolFactory` 用 `contextvars.ContextVar` 保存 active query plan，避免并发子查询互相串状态。
- `FusionRetriever` 先归一化向量分数和 BM25 分数，再用 `alpha` 融合。
- `_expand_candidate()` 能把 paragraph 扩展成 parent section 或 sibling window。
- `JsonNodeStore` 会缓存 nodes 和 trees，并按 mtime 判断是否需要重载。
- API 上传文件后用 `asyncio.create_task()` 后台索引，完成后会 `invalidate_graph_cache()`。
- `get_cached_graph()` 用索引路径、模型和 embedding 配置计算 fingerprint，但生产上还可以把 corpus profile 版本也纳入 fingerprint。
- `FakeEmbeddings` 是确定性 hash embedding，方便离线测试，不代表真实语义效果。

## 当前短板要主动讲

主动讲短板不丢分，关键是讲出下一步。

### 中文 BM25 分词还不够好

`BM25Bundle` 当前用 `split()` 做 tokenization。英文文档够用，中文连续文本会吃亏。虽然 `SemanticNLTKChunker` 里用了 jieba，但 BM25 语料本身没有统一使用中文分词。

下一步：

- 给 BM25 增加语言感知 tokenizer。
- 中文走 jieba 或更适合领域词的分词器。
- 技术缩写、函数名、模型名保留原样。

### checkpointer 不是生产级

`agent/graph.py` 使用 `InMemorySaver`。

下一步：

- 接 SQLite/Postgres checkpointer。
- 明确 `thread_id` 和业务 `session_id` 的映射。
- 对 replay/resume 下的副作用做幂等。

### 部分存储适配器只是预留

`sqlite_vec_store.py` 还是 `NotImplementedError`。

下一步：

- 真正实现 sqlite-vec 或 pgvector。
- 把 FAISS、BM25、NodeStore 的生命周期统一到索引版本。

### 文件上传链路还比较轻

API 当前会 `await upload.read()` 一次性读入内存，然后后台 task 索引。

下一步：

- 流式落盘。
- 上传大小限制。
- 文件类型检测和安全扫描。
- 任务队列和重试。
- 索引进度更细粒度上报。

### eval 数据量还小

当前评测集可做回归，但不能代表真实线上表现。

下一步：

- 扩充难例、长尾例和越界例。
- 标注 gold doc/node。
- 加线上 bad case 回流。
- 区分离线 extractive fallback 和在线 grounded generation。

## 简历怎么写

可以写成 4 条：

- 构建基于 LangGraph 的 Agentic RAG 流程，将知识库问答拆分为路由决策、查询规划、问题改写、混合检索、证据聚合和结构化回答，支持直接回答、超范围拒答和检索问答三类分支。
- 设计 BM25+FAISS 混合检索与多阶段 retrieval pipeline，包含候选召回、去重、规则增强、Flashrank 重排和 token budget 下的上下文打包，支持 flat 与 hierarchical 两种索引模式。
- 引入 `corpus_profile` 作为知识库边界和检索先验，参与路由、query plan、query rewrite 和回答风格控制，降低越界问答和无关召回。
- 搭建 Gradio 调试工作台、FastAPI SSE 接口和离线 eval runner，将质量评估拆成 routing、retrieval、answer 三层，便于定位坏例来源和做配置对比。

## 项目讲述模板

### 业务背景

普通 RAG Demo 往往只验证“能不能搜到点东西”，但真实知识库问答要解决三个问题：边界、证据和可调试性。边界是系统要知道自己能回答什么，证据是最终答案要能追溯来源，可调试性是坏答案出现时能定位是哪一层错了。

### 我的设计

我用 LangGraph 把主流程拆成状态图，用 `corpus_profile` 明确知识库范围，用 query plan 和 rewrite 提高检索质量，用 BM25+FAISS 做混合召回，再通过 rerank 和 context packing 控制进入模型的证据。最后用 `GroundedAnswer` 约束输出，确保答案、证据、置信度和局限性一起返回。

### 我做的取舍

主流程用 LangGraph，因为它有真实分支和多子查询派发。检索内部不用图节点，因为 retrieve、dedupe、rerank、pack 是确定性同步管道，放在 `FusionRetriever` 里更直接。子查询检索用 LangChain agent，因为工具调用循环交给成熟框架更合适。

### 当前结果

离线评测里 `flat_rerank` 比 `baseline_flat` 的 MRR 和 nDCG 更好，说明重排对检索排序有效；`hierarchical` 当前还没跑赢，说明分层索引需要继续优化结构化召回和上下文扩展策略。这个结果对我有价值，因为它避免了只凭直觉说 hierarchical 一定更好。

### 后续计划

我会优先补中文 BM25 分词、持久 checkpointer、索引版本管理、生产级文件上传和更完整的 bad case 评测集。

## 面试前 10 分钟速记

- 不要先报框架名，先讲用户问题进来后的链路。
- 项目核心是可路由、可观察、可评测的 Agentic RAG。
- LangGraph 管主流程，LangChain agent 管子查询工具循环。
- DeepAgents 没有直接使用，只能说设计思想有相似处。
- `corpus_profile` 是知识库边界和检索先验。
- 检索链路是 BM25+FAISS -> dedupe -> heuristic boosts -> Flashrank -> context packing。
- hierarchical 的价值是组织证据和扩展上下文，不是天然提升所有指标。
- `GroundedAnswer` 让答案、证据、置信度、局限性绑定。
- 当前 checkpointer 是 `InMemorySaver`，生产要换持久化。
- 当前 BM25 对中文还弱，生产要做分词优化。
- eval 分 routing、retrieval、answer 三层，不把所有问题都归因给 prompt。

## 官方资料参考

- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangChain Agents: https://docs.langchain.com/oss/python/langchain/agents
- LangChain Structured Output: https://docs.langchain.com/oss/python/langchain/structured-output
- Deep Agents Overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents Subagents: https://docs.langchain.com/oss/python/deepagents/subagents
