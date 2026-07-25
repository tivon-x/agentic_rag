# Agentic RAG V2 第二阶段 Goal 提示词

## 使用方式

- 每次只创建一个 Goal，不要同时启动多个里程碑。
- 按 M1、M2、M3、M4、M5、M6 的顺序执行。
- M1 至 M3 属于 Core。M4、M5、M6 必须满足进入条件并获得用户再次授权。
- `docs/research/v2_upgrade_plan.md` 是唯一实施方案。`tasks.md`、`.sisyphus/` 和旧项目指南只作为历史材料，发生冲突时不得覆盖 V2 方案。
- 每个 Goal 完成后停止，等待用户评审，不得自动进入下一个 Goal。
- 所有 Goal 都禁止自动推送远端、创建 PR、安装 Docling 或新增外部服务。

## Goal 1：M1 运行与索引可靠性

```text
目标：在 C:\Users\27564\Documents\code\ai\rag\agentic_rag 仓库中完成 Agentic RAG V2 的 M1“运行与索引可靠性”，建立可靠、可恢复、可回滚的 Core 运行底座。

开始前必须读取并遵守：
1. 仓库根目录 AGENTS.md。
2. docs/research/v2_upgrade_plan.md，重点阅读第 4、8、9、10、11、12、15 节。
3. docs/research/phase1_research_report.md。
4. web/AGENTS.md。修改前端前还要读取当前安装版本 node_modules/next/dist/docs/ 中与改动相关的 Next.js 文档。

当前基线：
- 工作分支应为 codex/v2-core。
- pre-V2 baseline commit 为 5983aca。
- baseline 后端测试结果为 154 passed、1 failed。失败用例是 tests/test_settings.py::test_load_settings_defaults，原因是测试没有隔离 EMBEDDING_MODEL 环境变量。
- baseline Ruff 通过，pnpm --dir web lint 和 pnpm --dir web build 通过。
- 已知安全问题：api/routers/indexing.py 直接使用 upload.filename 拼接保存路径，必须在 M1 中修复路径穿越风险，并增加回归测试。

执行边界：
- 只实施 M1，不改变检索算法、parser、chunk schema、Agent 路由或回答策略。
- 不开始 M2 至 M6，不安装 Docling，不新增 Redis、Celery、PostgreSQL、任务服务或其他外部基础设施。
- 复用现有 FastAPI、SQLite、Indexer、Settings 和前端代码，不另建平行实现。
- 所有配置继续通过 AppSettings，禁止在路由中直接读取环境变量。
- 保护所有用户文件。开始时运行 git status --short --branch -uall；如果出现不属于本 Goal 的新增改动，保留并绕开，发生路径冲突时停止并说明。

必须完成：
1. 修复环境复现和测试环境污染，保证 dev extra 下完整测试使用 Python 3.12+，Settings 测试不受宿主环境变量影响。
2. 建立明确的数据库 migration/version 机制，并在迁移 sessions.db 前创建可恢复副本。
3. 将 _BACKGROUND_TASKS 替换为单 index worker、SQLite lease、heartbeat、启动扫描和有限重试。
4. indexing job 状态统一为 queued -> running -> completed|failed|cancelled。
5. 上传写入安全临时目录，校验文件名、后缀、大小和最终解析路径，禁止目录穿越。重复 Idempotency-Key 不创建第二个 job。
6. 新索引写入不可变临时版本，校验成功后原子切换 active pointer；失败时旧 active index 不变。
7. 合并 main.py 与 FastAPI 的 Settings 加载边界，避免测试和启动方式产生不同配置。
8. /api/chat/stream 只发送进度、证据和一次最终答案，不转发任意节点的原始模型 token。
9. 保留旧索引读取适配器和 INDEX_WRITE_MODE=legacy 回滚路径。
10. 补齐 API、migration、worker recovery、index version、上传安全、幂等和 SSE 测试。

验证：
- 运行 uv run --extra dev python -m pytest -q，完整测试必须通过。
- 运行 uv run --extra dev ruff check .，必须通过。
- 运行 pnpm --dir web lint 和 pnpm --dir web build，必须通过。
- 手工故障注入：上传两篇论文，在 indexing running 时终止 API，再启动；确认最多一个 worker 恢复任务。
- 确认新索引校验前 active pointer 不变化。
- 确认重复 Idempotency-Key 不产生第二个 job。
- 确认恶意文件名无法逃出 UPLOAD_ROOT。
- 确认 SSE 不包含路由、规划或中间生成 token。

交付：
- 创建 docs/implementation/m1_acceptance.md，记录基线、数据库迁移、状态机、故障注入、自动测试、人工检查、已知坏例、回滚方法和实际修改文件。
- 仅在所有 M1 验收通过后创建一个独立 commit，不推送远端。
- 最终回复给出 commit、测试结果、人工检查结果、剩余风险和 M2 是否具备进入条件。
- 完成后停止，不得自动执行 M2。只有目标真实完成且没有剩余必做项时才把 Goal 标记为 complete。
```

## Goal 2：M2 论文目录与页码证据

```text
目标：完成 Agentic RAG V2 的 M2“论文目录与页码证据”，把现有通用文档 RAG 升级成可管理、可搜索、可回到原 PDF 页面的个人论文库产品。

进入条件：
- M1 已完成，docs/implementation/m1_acceptance.md 存在且结论允许进入 M2。
- M1 的完整测试、Ruff、前端 lint/build 和恢复测试均已通过。
- 用户已明确要求执行本 Goal。
- 如果进入条件不满足，停止并报告，不得绕过。

开始前读取 AGENTS.md、web/AGENTS.md、docs/research/v2_upgrade_plan.md、docs/research/phase1_research_report.md 和 M1 验收报告。V2 方案优先于 tasks.md、.sisyphus/ 和旧项目指南。

执行边界：
- 只实施 M2，不改变默认检索融合算法，不开始 Adaptive Agent、run checkpoint、trace、Compare 或 Workspace。
- 默认 parser 使用 PyMuPDF4LLM + deterministic structure normalizer，保留 legacy fallback。
- 不安装 Docling，不承诺 OCR、公式语义解析、bbox 高亮或自动合并同一论文的不同 PDF 修订版。
- 修改前端时读取当前 Next.js 版本文档，交付可实际使用的 Library、Paper Detail 和 Search，不做纯后端半成品。

必须完成：
1. 建立 parser protocol、PyMuPDF4LLM parser、结构归一化、质量检查、超时和 legacy fallback。
2. 建立 paper、paper_version、section、passage 和 parsed artifact 数据闭环。
3. paper 表示上传文件实体；paper_versions 表示 parser/normalization 解析版本。不同字节的 PDF 暂不自动合并。
4. 元数据提取按 PDF metadata、首屏启发式、文件名依次降级，保存字段来源和置信度；支持用户修改 title、authors、year、venue、DOI 或 arXiv ID。
5. 生成稳定的 paper、paper_version、section、passage ID，保存 page_start、page_end、quote_text 和 retrieval_text。
6. Search API 返回 paper、section、page、quote 和各评分阶段，PDF API 支持 Range 请求和 #page=N 跳转。
7. 将 /kb 迁移为 /library，新增 /papers/[id] 和 /search。页面必须展示解析状态、降级原因、元数据置信度、用户校正入口和原页跳转。
8. 修改元数据后，新的 retrieval_text prefix 生效，但 quote_text 保持原文。
9. 建立 parser gold：4 篇 dev、12 篇 test、共 48 个重点页面。覆盖双栏、表格、公式、长文、低文本和错误 metadata。不得根据 test 结果反向修改 test 标注。
10. 增加 parser、metadata、paper API、Search API、Range 和页面跳转测试。

验证：
- uv run --extra dev python -m pytest tests/test_pdf_parser.py tests/test_parser_quality.py tests/test_metadata.py tests/test_paper_api.py tests/test_search_api.py -q
- uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
- uv run --extra dev ruff check indexing api tests
- pnpm --dir web lint
- pnpm --dir web build
- 另外运行完整后端测试，确认 M1 没有回归。

人工检查：
- 分别导入双栏、表格、公式、长文和低文本 PDF。
- 错误 PDF title 不能覆盖首屏可信标题，未知作者保持空值。
- 修改 title 后重新索引，retrieval_text 使用新 prefix，quote_text 不变化。
- 从 Search 结果打开正确论文页。
- parser 失败、fallback 和 needs_ocr 原因对用户可见。

交付：
- 创建 docs/implementation/m2_acceptance.md，记录 parser 对照、gold 数据、失败案例、页面截图或路径、测试结果、回滚方法和修改文件。
- 仅在 M2 验收通过后创建独立 commit，不推送。
- 如果 parser 质量门槛失败，保留 PAPER_PARSER=legacy 和上一 active index，并明确停止进入 M3。
- 完成后停止，不得自动执行 M3。
```

## Goal 3：M3 固定 V2 检索与精简评测

```text
目标：完成 Agentic RAG V2 的 M3“固定 V2 检索与精简评测”，通过可复现实验证明 metadata-prefixed hybrid retrieval、mixed BM25、RRF、rerank 和可选 section neighbor expansion 的净收益。

进入条件：
- M2 已完成，docs/implementation/m2_acceptance.md 存在且 parser 门槛通过。
- 用户已明确要求执行本 Goal。
- 如果 M2 失败或 parser artifact 不稳定，停止，不得使用临时数据绕过。

开始前读取 AGENTS.md、docs/research/v2_upgrade_plan.md 第 6、10、11 节、phase1_research_report.md、M1/M2 验收报告。V2 方案是唯一实施依据。

执行边界：
- 只实施固定检索和 Core 评测，不增加 query routing、multi-query、自纠错循环、claim validation、run worker 或详细 Agent trace。
- B0 至 B3 必须共用相同 parser artifact、embedding、reranker、top-k 和 test set。
- 不能因为某个方案技术上更新就设为默认，默认切换只由冻结评测门槛决定。
- 不使用 Anthropic 式 LLM chunk context，正式名称保持 metadata-prefixed retrieval。

必须完成：
1. 分离 retrieval_text 与 quote_text，metadata prefix 只进入检索表示，引用始终使用原文。
2. 实现中英 mixed tokenizer。
3. 建立可配置、可测试的固定 pipeline registry。
4. B0 为 dense + BM25 无 rerank；B1 为当前 flat_rerank；B2 为 metadata prefix + mixed BM25 + dense + RRF + rerank；B3 为 B2 + section neighbor expansion。
5. RRF 使用方案确定的 k=60；召回、融合、重排、扩展和 context packing 均记录阶段性结果。
6. 建立 48 条冻结 retrieval test，四类各 12 条：精确术语与定义、方法与章节定位、实验数值与表格、跨论文或跨章节问题。
7. 建立 8 条 answer smoke，但不得用它代替正式 answer test。
8. 对 B2 做去 metadata prefix、去 sparse、去 dense、RRF 换回 min-max、去 rerank 的单因素消融；neighbor expansion 单独作为 B3。
9. 报告 Recall@5/10、MRR@10、nDCG@10、paper Recall@10、section Recall@10、逐问题胜平负、目标子集、p50/p95 延迟和坏例。
10. 不填虚构提升百分比，不用统一综合分掩盖子集退化。

发布门槛：
- B2 Recall@10 不低于 B1。
- B2 相对 B1 至少 8 条改善 gold rank，退化不超过 4 条。
- 四个子集没有任何一个出现 Recall@10 下降 2 条以上。
- B2 p95 检索延迟不超过 B1 的 1.5 倍。
- B3 只有在跨章节子集至少改善 3 条、其他子集总退化不超过 1 条时才设为默认。
- 未达到门槛时保持 B1 或 B2，不得为了完成 Goal 强行启用更复杂方案。

验证：
- uv run --extra dev python -m pytest tests/test_bm25_index.py tests/test_retriever.py tests/test_retrieval_pipeline.py tests/test_evals.py -q
- uv run python -m evals.runner --config evals/configs/v2_b1.yaml
- uv run python -m evals.runner --config evals/configs/v2_b2.yaml
- uv run python -m evals.runner --config evals/configs/v2_b3.yaml
- uv run python -m evals.build_report --runs artifacts/evals/v2_core
- uv run --extra dev ruff check indexing core evals tests
- 运行完整后端测试、前端 lint 和 build，确认 M1/M2 无回归。

人工检查至少 10 个 B1/B2 rank 变化案例，并对表格、缩写、跨章节、中文术语各检查 3 个坏例。确认引用不显示 metadata prefix。

交付：
- 创建 docs/implementation/m3_acceptance.md，保存配置、数据集版本、逐问题结果、消融报告、坏例、延迟、默认 pipeline 决策和回滚方式。
- 仅在实现和评测可复现后创建独立 commit，不推送。
- 完成后停止。明确报告 Core 是否通过、默认选择 B1/B2/B3，以及 M4 是否具备进入条件。
- 即使 M3 通过，也不得自动执行 M4，必须等待用户再次批准 Enhanced。
```

## Goal 4：M4 持久 Run 与有界 Adaptive Agent

```text
目标：在 Core 检索基线已经被实验证明后，完成 Agentic RAG V2 的 M4“持久 run 与有界 Adaptive Agent”，实现可恢复、可终止、受预算约束的 Agentic RAG。

进入条件：
- docs/implementation/m3_acceptance.md 存在。
- M3 的 B2/B3 发布门槛通过，固定默认 pipeline 已冻结。
- 用户明确批准进入 Enhanced 并要求执行本 Goal。
- 任一条件不满足就停止，不得自行放宽门槛。

开始前读取 AGENTS.md、web/AGENTS.md、v2_upgrade_plan.md 第 4、7、8、10、11、12 节，以及 M1 至 M3 验收报告。

执行边界：
- 只实施 M4，不实现完整调试工作台、Compare、Workspace、Docker 或外部任务服务。
- 保留 Core fixed chat 作为独立回滚路径。
- 不引入多 Agent、GraphRAG、RAPTOR、Redis、Celery 或 PostgreSQL。
- 所有 prompt 放在 agent/prompts.py，结构化输出放在 agent/schemas.py，节点保持普通函数，路由判断放在 agent/edges.py。
- GraphState 只保存 JSON 基础类型和紧凑控制状态，禁止保存 Document、完整候选、完整会话、LLM client 或数据库连接。

必须完成：
1. 实现 direct、fixed、adaptive、refuse 四类策略。direct 不允许产生新的论文事实；adaptive 只用于多论文比较、跨章节综合或固定检索证据不足。
2. query plan 最多 4 个子问题，最多两轮检索，总 tool calls 不超过 6，总 rerank passage 不超过 120，总 evidence 不超过 12，总上下文不超过 12,000 tokens。
3. evidence ID 与上一轮完全相同、覆盖不再改善、预算耗尽、取消或模型错误时可靠终止。
4. 实现 compact GraphState。详细 candidate、evidence quote 和事件分别落到 retrieval_candidates、evidence_items、run_events。
5. 使用 thread_id=run_id。session history 来自创建 run 时写入的不可变 history_snapshot_json，不由 checkpoint 承担。
6. 使用 AsyncSqliteSaver、JsonPlusSerializer(pickle_fallback=False) 和 LANGGRAPH_STRICT_MSGPACK=true。生产依赖锁定到包含严格序列化安全修复的兼容版本。
7. 实现 run worker、SQLite lease、10 秒 heartbeat、启动扫描、最多两次尝试和幂等 upsert。
8. final answer、evidence 和 assistant message 必须在同一事务成功后才能把 run 标记为 completed。
9. 支持 cancel_requested、SSE Last-Event-ID 重连和事件序号恢复。
10. 生成结构化 claims，每个主要事实 claim 声明 evidence IDs；校验证据存在、属于当前 index version 且 quote 支持 claim。
11. 不向用户发送 provisional answer。SSE 只发送进度、确认后的 evidence、validation.completed 和一次 answer.final。
12. 通过 ANSWER_STRATEGY=fixed 完整绕过 adaptive。

验证：
- uv run --extra dev python -m pytest tests/test_agent_graph.py tests/test_agent_budget.py tests/test_claim_validation.py tests/test_run_recovery.py tests/test_run_streaming.py -q
- uv run python -m evals.runner --config evals/configs/v2_adaptive.yaml
- uv run --extra dev ruff check agent api tests
- 运行完整后端测试、前端 lint/build。

必须通过故障注入：
- 第一轮检索后杀进程并重启。
- answer 事务写入前杀进程并重启。
- lease 过期后启动第二 worker 竞争。
- SSE 断线后用 Last-Event-ID 继续。
- 同一 session 并发创建两个 run。

成功条件：
- 最多一个 worker 持有 lease。
- run 从 checkpoint 恢复或使用保存输入幂等重启。
- assistant message、evidence、事件和只读 tool side effect 不重复。
- 同 session 的第二个并发 run 返回 409。
- Agent 在所有测试中都能在预算内终止。

交付：
- 创建 docs/implementation/m4_acceptance.md，记录状态机、GraphState 大小、恢复协议、故障注入、预算、评测、坏例和回滚方式。
- 仅在 M4 验收通过后创建独立 commit，不推送。
- 完成后停止，不自动执行 M5。
```

## Goal 5：M5 Trace、可解释性与完整评测

```text
目标：完成 Agentic RAG V2 的 M5“调试 trace 和评测补齐”，让每次回答都能追溯策略、计划、候选、重排、证据、补检、停止原因、延迟和 token，并用独立数据集证明 Adaptive Agent 的净收益。

进入条件：
- M4 已完成，docs/implementation/m4_acceptance.md 存在。
- M4 的 serializer、lease、恢复、SSE 重连、并发和幂等测试全部通过。
- 用户已明确要求执行本 Goal。

开始前读取 AGENTS.md、web/AGENTS.md、v2_upgrade_plan.md 第 7、8、10、11、12 节，以及 M1 至 M4 验收报告。

执行边界：
- 只实施 trace、技术模式和 Enhanced 评测，不实现 Compare、Workspace、备份或 Docker。
- Trace 只用于观察和诊断，Agent 决策不得依赖调试页面是否开启。
- 不泄露 prompt、API Key、完整环境变量或不必要的用户原文。
- trace 默认保存 7 天，用户明确保存的 artifact 不受自动清理影响。

必须完成：
1. run_events、retrieval_candidates 和 evidence_items 组成可导出的稳定 trace。
2. Chat 提供普通模式和技术模式。技术模式展示 route、query plan、每轮候选排名、融合与重排分数、接受/拒绝证据、补检原因、预算、停止原因、节点延迟和 token。
3. 所有页面优先展示用户可理解的论文、章节、页码和 quote，内部 ID 只作为辅助信息。
4. 完成 24 条 answer test 和 48 条 route/refusal test。Route 四类 direct、fixed、adaptive、refuse 各 12 条。
5. Answer 指标包括 claim support precision、citation correctness、citation completeness、requirement coverage、unsupported major claim、answer/refusal utility。
6. Agent 指标包括 route macro F1、每类 confusion matrix、successful termination rate、平均检索轮数、tool calls、重复检索率、p50/p95 latency 和 LLM input/output tokens。
7. 完成 Adaptive 消融：固定 B3、B3 + routing、B3 + 一轮补检、B3 + 两轮补检和 claim validation。
8. 没有模型定价配置时只报告 token，不猜测货币成本。
9. DEBUG_TRACE=false 时停止写详细 candidate 和 trace，但保留 run 结果、错误和正常 Agent 行为。

质量门槛：
- route 每类 recall 不低于 0.75，macro F1 不低于 0.80。
- Adaptive 在 24 条 answer test 中至少改善 5 条 requirement coverage，退化不超过 2 条。
- citation correctness 不低于 fixed B3。
- Adaptive p95 不超过 fixed 的 2.5 倍，平均检索轮数不超过 1.5。
- successful termination 和故障恢复不得出现重复消息、证据或工具副作用。
- 未达到门槛时默认保持 fixed，不能为了展示 Agent 而强制启用 adaptive。

验证：
- uv run --extra dev python -m pytest tests/test_trace_repository.py tests/test_debug_api.py tests/test_route_eval.py -q
- uv run python -m evals.runner --config evals/configs/v2_route.yaml
- uv run python -m evals.runner --config evals/configs/v2_answer.yaml
- uv run --extra dev ruff check .
- uv run --extra dev python -m pytest -q
- pnpm --dir web lint
- pnpm --dir web build

人工检查：
- 普通用户看不到不必要的内部调试噪声。
- 技术模式能从最终 claim 反向定位到 evidence、passage、排名阶段和原 PDF 页。
- 关闭 DEBUG_TRACE 后回答结果不改变。
- Trace 保留和清理不会删除用户保存的 artifact。

交付：
- 创建 docs/implementation/m5_acceptance.md，保存完整评测结果、confusion matrix、Adaptive 消融、trace 示例、隐私检查、坏例、默认策略决定和回滚方法。
- 仅在 M5 验收完成后创建独立 commit，不推送。
- 完成后停止，不自动执行 M6。
```

## Goal 6：M6 Compare、Workspace 与部署

```text
目标：完成 Agentic RAG V2 的 M6“比较、Workspace 和部署”，把已经通过质量门槛的科研助手收口为可长期个人使用、可备份恢复、可本地部署的产品。

进入条件：
- M4 和 M5 已完成，对应验收报告存在。
- Adaptive 的质量、延迟、成本和恢复门槛通过；如果 Adaptive 未通过，Product 必须继续使用 fixed 默认策略。
- 用户明确批准 Product 并要求执行本 Goal。

开始前读取 AGENTS.md、web/AGENTS.md、v2_upgrade_plan.md 第 3、8、11、12、14、15 节，以及 M1 至 M5 验收报告。

执行边界：
- 只实现 Compare、Workspace、artifact、备份恢复和本地部署收口。
- 不增加用户系统、团队协作、云服务、Redis、Celery、PostgreSQL、专用向量数据库或 SaaS 计费。
- Docker 只封装现有 FastAPI 与 Next.js；SQLite、uploads、indexes 和 parsed artifacts 通过 volume 持久化。
- Compare 不允许在缺证据时生成推测。

必须完成：
1. Compare 支持固定选择 2 至 5 篇论文、1 至 6 个比较维度。
2. 每个比较单元格必须绑定 evidence；没有证据时显示“未找到”，不得用模型常识补写。
3. 支持保存 answer、comparison 和 evidence collection，并能从 Workspace 重新打开。
4. Artifact 保存稳定 evidence ID、paper/version、index version 和生成配置，原始 quote 可追溯到 PDF 页。
5. 提供一致性备份和恢复命令，覆盖 SQLite、uploads、indexes 和 parsed artifacts。
6. 备份命令默认只读检查并列出范围；覆盖或删除现有数据前必须显式确认。
7. 恢复前校验 manifest、版本和目标目录，失败时不得留下半恢复状态。
8. Docker Compose 启动 FastAPI 和 Next.js，所有数据目录使用明确 volume；本地 uv/pnpm 运行方式仍然可用。
9. 更新 README，给出安装、索引论文、Search、Chat、Compare、备份、恢复和 Docker 的真实命令。
10. 产品页面保持 Library、Search、Chat、Compare、Workspace 的清晰分工，不把技术 trace 强塞给普通用户。

验证：
- uv run --extra dev python -m pytest tests/test_artifacts_api.py tests/test_compare.py tests/test_backup_restore.py -q
- uv run --extra dev python -m pytest -q
- uv run --extra dev ruff check .
- pnpm --dir web lint
- pnpm --dir web build
- docker compose up --build -d
- 检查 FastAPI 和 Next.js 健康状态后运行 docker compose down，禁止使用 -v，保留持久化数据。

人工检查：
- 用 3 篇同主题论文生成方法、数据集、指标、局限四维比较，逐格打开 evidence 和原 PDF 页。
- 制造一个无证据比较项，确认显示“未找到”。
- 保存 answer、comparison 和 evidence collection，重启后仍可打开。
- 备份后在空数据目录恢复，论文、索引、会话和 artifact 数量及 manifest 一致。
- Docker 停止并重启后数据仍存在。
- Docker 失败时，本机 uv run uvicorn 和 pnpm --dir web dev 仍能运行。

交付：
- 创建 docs/implementation/m6_acceptance.md，记录产品流程、备份恢复验证、Docker 验证、数据一致性、截图或页面路径、测试结果、坏例和回滚方法。
- 更新面向用户的 README，但不要提前撰写第三阶段的完整源码指南。
- 仅在 M6 验收通过后创建独立 commit，不推送。
- 完成后停止，并报告第二阶段是否整体完成、哪些功能采用 fixed 或 adaptive 默认策略，以及第三阶段项目指南可以依赖的最终事实材料。
```
