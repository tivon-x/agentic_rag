# Agentic RAG V2 第二阶段 Goal 提示词

> 当前状态：M1 至 M3.2、M4.1、M4.1.1、M4.1.2 均已完成。M4.1 系列未通过质量门槛，`ANSWER_STRATEGY=fixed` 保持默认，M4.2 已终止且不得执行。

## 使用方式

- 每次只创建一个 Goal，不要同时启动多个里程碑。
- 已完成 Goal 仅保留为历史执行记录，不能再次运行或覆盖冻结数据、报告和结论。
- 后续只按 M5、M6 的顺序执行，并且每个 Goal 都需要用户再次授权。
- `docs/research/v2_upgrade_plan.md` 是唯一实施方案。`tasks.md`、`.sisyphus/` 和旧项目指南只作为历史材料，发生冲突时不得覆盖 V2 方案。
- 每个 Goal 完成后停止，等待用户评审，不得自动进入下一个 Goal。
- 所有 Goal 都禁止自动推送远端、创建 PR、安装 Docling 或新增外部服务。

## Goal 1：M1 运行与索引可靠性（已完成）

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
- M1 实施起点 commit 为 324e7c0，其中包含 OpenAI-compatible embedding provider 的 raw-string 输入修复。
- 当前后端测试结果为 155 passed、1 failed。失败用例是 tests/test_settings.py::test_load_settings_defaults，原因是测试没有隔离 EMBEDDING_MODEL 环境变量。
- baseline Ruff 通过，pnpm --dir web lint 和 pnpm --dir web build 通过。
- 已知安全问题：api/routers/indexing.py 直接使用 upload.filename 拼接保存路径，必须在 M1 中修复路径穿越风险，并增加回归测试。

执行边界：
- 只实施 M1，不改变检索算法、parser、chunk schema、Agent 路由或回答策略。
- 不开始 M2 至 M6，不安装 Docling，不新增 Redis、Celery、PostgreSQL、任务服务或其他外部基础设施。
- 复用现有 FastAPI、SQLite、Indexer、Settings 和前端代码，不另建平行实现。
- 所有配置继续通过 AppSettings，禁止在路由中直接读取环境变量。
- embedding raw-string 行为必须成为显式配置和索引契约，不能继续作为 indexing/embeddings.py 中未记录的隐藏细节。
- 保护所有用户文件。开始时运行 git status --short --branch -uall；如果出现不属于本 Goal 的新增改动，保留并绕开，发生路径冲突时停止并说明。

必须完成：
1. 修复环境复现和测试环境污染，保证 dev extra 下完整测试使用 Python 3.12+，Settings 测试不受宿主环境变量影响。
2. 建立明确的数据库 migration/version 机制，并在迁移 sessions.db 前创建可恢复副本。
3. 将 _BACKGROUND_TASKS 替换为单 index worker、SQLite lease、heartbeat、启动扫描和有限重试。
4. indexing job 状态统一为 queued -> running -> completed|failed|cancelled。
5. 上传写入安全临时目录，校验文件名、后缀、大小和最终解析路径，禁止目录穿越。重复 Idempotency-Key 不创建第二个 job。
6. 将 embedding provider、model、dimension、input mode、context-length check 和 max input chars 纳入 AppSettings。raw-string provider 使用 check_embedding_ctx_length=false，应用侧默认 EMBEDDING_MAX_INPUT_CHARS=6000；不得把 API Key 或其他凭据写入 manifest。
7. 新索引写入不可变临时版本，manifest 记录 embedding provider、model、dimension、input mode、context-length check、max input chars 和代码版本；校验成功后原子切换 active pointer，失败时旧 active index 不变。
8. 加载索引时校验当前 embedding 配置与 manifest。model、dimension 或 input mode 不兼容时拒绝加载并要求重建，不能静默查询旧向量。
9. 合并 main.py 与 FastAPI 的 Settings 加载边界，避免测试和启动方式产生不同配置。
10. /api/chat/stream 只发送进度、证据和一次最终答案，不转发任意节点的原始模型 token。
11. 保留旧索引读取适配器和 INDEX_WRITE_MODE=legacy 回滚路径。
12. 补齐 API、migration、worker recovery、index version、embedding compatibility、上传安全、幂等和 SSE 测试。

验证：
- 运行 uv run --extra dev python -m pytest -q，完整测试必须通过。
- 运行 uv run --extra dev ruff check .，必须通过。
- 运行 pnpm --dir web lint 和 pnpm --dir web build，必须通过。
- 手工故障注入：上传两篇论文，在 indexing running 时终止 API，再启动；确认最多一个 worker 恢复任务。
- 确认新索引校验前 active pointer 不变化。
- 确认 embedding model、dimension 或 input mode 改变后旧索引不会被静默加载。
- 确认重复 Idempotency-Key 不产生第二个 job。
- 确认恶意文件名无法逃出 UPLOAD_ROOT。
- 确认 SSE 不包含路由、规划或中间生成 token。

交付：
- 创建 docs/implementation/m1_acceptance.md，记录基线、数据库迁移、状态机、故障注入、自动测试、人工检查、已知坏例、回滚方法和实际修改文件。
- 仅在所有 M1 验收通过后创建一个独立 commit，不推送远端。
- 最终回复给出 commit、测试结果、人工检查结果、剩余风险和 M2 是否具备进入条件。
- 完成后停止，不得自动执行 M2。只有目标真实完成且没有剩余必做项时才把 Goal 标记为 complete。
```

## Goal 2：M2 论文目录与页码证据（已完成）

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
6. 在调用 embedding provider 前对 retrieval_text 做确定性的长度校验。由于 raw-string 模式关闭 LangChain 自动切分，passage 和 metadata prefix 的组合必须受 AppSettings 中 EMBEDDING_MAX_INPUT_CHARS=6000 的默认硬上限保护；超限时重新切分或明确失败，不能把超长输入直接交给 provider。
7. Search API 返回 paper、section、page、quote 和各评分阶段，PDF API 支持 Range 请求和 #page=N 跳转。
8. 将 /kb 迁移为 /library，新增 /papers/[id] 和 /search。页面必须展示解析状态、降级原因、元数据置信度、用户校正入口和原页跳转。
9. 修改元数据后，新的 retrieval_text prefix 生效，但 quote_text 保持原文。
10. 建立 parser gold：4 篇 dev、12 篇 test、共 48 个重点页面。覆盖双栏、表格、公式、长文、低文本和错误 metadata。不得根据 test 结果反向修改 test 标注。
11. 增加 parser、metadata、passage 长度边界、paper API、Search API、Range 和页面跳转测试。

验证：
- uv run --extra dev python -m pytest tests/test_pdf_parser.py tests/test_parser_quality.py tests/test_metadata.py tests/test_paper_api.py tests/test_search_api.py -q
- uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
- uv run --extra dev ruff check indexing api tests
- npm --prefix web run lint
- npm --prefix web run build
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

## Goal 3：M3 固定 V2 检索与精简评测（已完成）

```text
目标：完成 Agentic RAG V2 的 M3“固定 V2 检索与精简评测”，通过可复现实验证明 metadata-prefixed hybrid retrieval、mixed BM25、RRF、rerank 和可选 section neighbor expansion 的净收益。

进入条件：
- M2 已完成，docs/implementation/m2_acceptance.md 存在且 parser 门槛通过。
- 用户已明确要求执行本 Goal。
- 如果 M2 失败或 parser artifact 不稳定，停止，不得使用临时数据绕过。

开始前读取 AGENTS.md、docs/research/v2_upgrade_plan.md 第 6、10、11 节、phase1_research_report.md、M1/M2 验收报告。V2 方案是唯一实施依据。

执行边界：
- 只实施固定检索和 Core 评测，不增加 query routing、multi-query、自纠错循环、claim validation、run worker 或详细 Agent trace。
- B0 至 B3 必须共用相同 parser artifact、embedding provider、model、dimension、raw-string input mode、reranker、top-k 和 test set。
- 所有 B0 至 B3 索引必须从同一冻结配置重新构建。不得复用历史 fake embedding 索引或其他 embedding 模型生成的 FAISS 索引。
- 不能因为某个方案技术上更新就设为默认，默认切换只由冻结评测门槛决定。
- 不使用 Anthropic 式 LLM chunk context，正式名称保持 metadata-prefixed retrieval。

必须完成：
1. 分离 retrieval_text 与 quote_text，metadata prefix 只进入检索表示，引用始终使用原文。
2. 实现中英 mixed tokenizer。
3. 建立可配置、可测试的固定 pipeline registry。
4. B0 为 dense + BM25 无 rerank；B1 为当前 flat_rerank；B2 为 metadata prefix + mixed BM25 + dense + RRF + rerank；B3 为 B2 + section neighbor expansion。
5. 每次评测报告必须记录 embedding provider、model、dimension、input mode、索引 manifest 和代码 commit，查询 embedding 与索引 embedding 不一致时立即失败。
6. RRF 使用方案确定的 k=60；召回、融合、重排、扩展和 context packing 均记录阶段性结果。
7. 建立 48 条冻结 retrieval test，四类各 12 条：精确术语与定义、方法与章节定位、实验数值与表格、跨论文或跨章节问题。
8. 建立 8 条 answer smoke，但不得用它代替正式 answer test。
9. 对 B2 做去 metadata prefix、去 sparse、去 dense、RRF 换回 min-max、去 rerank 的单因素消融；neighbor expansion 单独作为 B3。
10. 报告 Recall@5/10、MRR@10、nDCG@10、paper Recall@10、section Recall@10、逐问题胜平负、目标子集、p50/p95 延迟和坏例。
11. 不填虚构提升百分比，不用统一综合分掩盖子集退化。

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
- 完成后停止。明确报告 Core 是否通过、默认选择 B1/B2/B3，以及后续策略收口是否具备进入条件。
- 即使 M3 通过，也不得自动执行后续里程碑，必须等待用户再次批准。
```

## Goal 4A：M4.1 有界 Adaptive 质量闭环（已完成，未通过）

```text
目标：冻结 M3.2 选出的 B1 检索链路，在不继续调检索参数的前提下，证明“证据缺口判断 + 一次定向补检”是否比 fixed B1 更好。M4.1 只解决策略质量，不实现持久 run、worker、checkpoint、SSE 重连或调试工作台。

进入条件：
- docs/implementation/m3_2_strategy_acceptance.md 存在，且 m3_strategy_closed=true、m4_entry_ready=true。
- artifacts/evals/v2_m3_2/m4_fixed_baseline.json 存在，selected_pipeline_name=v1_flat_rerank，pipeline_config_hash=ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17。
- formal holdout run count=1，metadata prefix leak=0，active index 未改变。
- 用户明确批准执行 M4.1。任一条件不满足就停止，不得重新运行或修改 M3.2 holdout。

开始前必须读取：
1. 仓库根目录 AGENTS.md。
2. docs/research/v2_upgrade_plan.md 第 6、7、9、10、11、12、15 节。
3. docs/implementation/m3_2_strategy_acceptance.md。
4. docs/implementation/m3_2_strategy_per_question.md。
5. artifacts/evals/v2_m3_2/m4_fixed_baseline.json。
6. docs/implementation/m4_1_adaptive_handoff.md。

执行边界：
- 所有事实型问题的每次检索都必须使用冻结 B1 contract，不新增或调整 dense、BM25、fusion、reranker、top-k、context packing 参数。
- M3.2 holdout 已经看过结果，只能用于总结失败类型，不得充当 M4.1 最终测试集。
- 先冻结 M4.1 route 和 answer 数据集及其 SHA-256，再实现或调试策略；冻结后不得根据 test 结果修改问题、标签、gold、阈值或评分器。
- M3 困难标签只用于编写数据集，禁止成为运行时硬编码路由规则。
- 保留现有 fixed graph 和 fixed chat。Adaptive 通过 ANSWER_STRATEGY=adaptive 显式启用，M4.1 完成后默认值仍为 fixed。
- 不新增数据库 migration、run worker、checkpoint saver、SSE 协议或前端技术模式。
- 所有 prompt 放在 agent/prompts.py，结构化输出放在 agent/schemas.py，路由函数放在 agent/edges.py。

必须完成：
1. 建立独立、紧凑的 AdaptiveGraphState，不能破坏现有 fixed GraphState 回滚路径。
2. direct 只处理寒暄、确认和格式调整，不检索，也不产生新的论文事实；refuse 处理超出论文库、要求外部实时事实或两轮后仍无证据的请求。
3. 事实型问题先拆成最多 3 个可检查需求，再执行第一轮 B1 检索。不能只根据问题表面复杂度决定 adaptive。
4. 证据充分性输出必须逐项声明 requirement、evidence IDs、coverage 和 missing reason。确定性校验负责 evidence 存在性、当前 index version、quote 非空、页码可定位和 ID 完整性；quote 是否语义支持 claim 由结构化模型判断，并在验收中单独报告误判，不能宣称为确定性证明。
5. 第一轮不足时，只为缺失需求生成最多 1 个补检查询。最多 2 轮、总 tool calls 不超过 4、总 evidence 不超过 12、总上下文不超过 12,000 tokens。
6. evidence IDs 无变化、coverage 无提升、补检 query 与已有 query 完全重复、预算耗尽、取消或模型错误时停止。
7. 第二轮仍不足时，只能输出带 limitations 的有限回答或 refuse，不允许第三轮检索。
8. 主要事实 claim 必须声明 evidence IDs。不支持的 claim 删除、降级措辞或转为 limitations。
9. 新建 48 条 route test，direct、fixed、adaptive、refuse 各 12 条；新建 24 条 answer test，其中 12 条覆盖 M3 暴露的困难类型，12 条为独立问题。
10. 固定 B1 与 adaptive 使用同一问题、history、scope、index version、模型和评分口径，保存逐题结果、路由混淆矩阵、轮数、tool calls、延迟、token、证据和停止原因。

质量门槛：
- route 每类 recall 不低于 0.75，macro F1 不低于 0.80。
- adaptive 相对 fixed B1 至少改善 5 条 requirement coverage，退化不超过 2 条。
- citation correctness、citation completeness 和主要事实支持率均不低于 fixed B1。
- unsupported major claim count 不高于 fixed B1。
- successful termination rate=100%，平均检索轮数不超过 1.5，总 tool calls 始终不超过 4。
- exact duplicate query + scope 次数为 0，coverage 不再提升时能够停止。
- adaptive p95 总延迟不超过 fixed B1 的 2.5 倍。

验证：
- uv run --extra dev python -m pytest tests/test_agent_graph.py tests/test_agent_budget.py tests/test_claim_validation.py tests/test_route_eval.py -q
- uv run python -m evals.runner --config evals/configs/v2_m4_1_route.yaml
- uv run python -m evals.runner --config evals/configs/v2_m4_1_answer.yaml
- uv run --extra dev ruff check agent core evals tests
- uv run --extra dev python -m pytest -q
- npm --prefix web run lint
- npm --prefix web run build

交付：
- 创建 docs/implementation/m4_1_acceptance.md，记录数据集 hash、评分口径、B1/adaptive 逐题结果、混淆矩阵、预算、延迟、token、坏例、误判、默认策略和回滚方式。
- 只有所有质量门槛通过时才写 m4_1_quality_passed=true 和 m4_2_entry_ready=true；否则两者均为 false，保持 ANSWER_STRATEGY=fixed。
- 仅在实现和评测可复现后创建一个独立 commit，不推送。
- 完成后停止，不得自动执行 M4.2。
```

## Goal 4A.1：M4.1.2 Adaptive 场景对齐复验（已完成，未通过）

```text
目标：在不调整冻结 B1 检索链路的前提下，建立新的、面向“首轮部分覆盖且一次定向补检可补齐缺口”的 M4.1.2 冻结评测，验证 bounded Adaptive 是否在其真正适用场景中优于同一 B1 的一轮 fixed。M4.1.1 已失败且不得改写；本 Goal 只复验策略质量，不实施 M4.2。

进入条件：
- HEAD 含 M4.1.1 收口提交 `cec8918`，且 `docs/implementation/m4_1_1_retrieval_quality_acceptance.md` 存在。
- B1 baseline 仍为 `v1_flat_rerank`，pipeline config hash 为 `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`；active index 未改变。
- 用户明确批准执行 M4.1.2，并在真实模型调用前再次授权。
- 任一条件不满足即停止；不得重跑 M3.2 holdout、不得覆盖 M4.1.1 结果。

开始前必须读取：
1. AGENTS.md。
2. docs/research/v2_upgrade_plan.md 第 6、7、9、10、11、12、15 节。
3. docs/implementation/m3_2_strategy_acceptance.md。
4. artifacts/evals/v2_m3_2/m4_fixed_baseline.json。
5. docs/implementation/m4_1_1_retrieval_quality_protocol.md。
6. docs/implementation/m4_1_1_retrieval_quality_acceptance.md。
7. docs/implementation/m4_1_2_adaptive_eval_handoff.md。

执行边界：
- 所有事实检索均校验并使用冻结 B1 contract；禁止修改 dense、BM25、fusion、reranker、top-k、context packing、embedding 或 active index。
- 不修改 M4.1.1 数据、标签、gold、grader、报告或结论；新数据集与 M3.2 holdout、M4.1.1 原问题均不得重复或近似改写。
- 先创建并冻结 M4.1.2 route/answer 数据、评分协议、阈值及 SHA-256，再修改任何 Adaptive prompt、route、planner、assessor、follow-up、answerer 或 grader。
- M3 困难标签、authoring snapshot、case ID、gold 和 route 标签只能用于离线数据编写与审计，禁止成为运行时路由规则。
- 保留 fixed graph/chat，默认始终为 `ANSWER_STRATEGY=fixed`；不新增 migration、worker、checkpoint、SSE 或前端模式。
- prompts 位于 agent/prompts.py，结构化输出位于 agent/schemas.py，路由函数位于 agent/edges.py。

必须完成：
1. 新建 `m4_1_2_route_v1.json`（48 条，四类各 12）和 `m4_1_2_answer_v1.json`（至少 24 条）；answer 至少有 12 条 adaptive-eligible、8 条 fixed-eligible、4 条明确证据不足题。
2. 可使用一次只读 B1 authoring snapshot 挑选题：fixed 类确认首轮覆盖全部 requirements；adaptive 类确认首轮缺至少一项且库中存在不同定向 query 可发现的页码可定位证据。记录 snapshot，但不得调用 Adaptive 挑题。
3. 冻结后运行时必须先拆最多 3 项 requirements，再完整原问题做首轮 B1；只有实际 evidence insufficiency 才允许一次只面向 missing requirements 的补检。
4. 冻结评分协议并拆分：确定性引用有效性、语义 quote 支持、gold 覆盖审计。结构化 grader 的布尔与理由矛盾记为 `grader_inconsistent`，不得在正式结果后自动修复或改分。
5. 最终 major claim 必须有实际 evidence ID；确定性校验证据存在、index version、quote、页码与 ID 完整性。语义判断由冻结 grader 完成，并以 20% 盲审清单报告 false positive/negative/inconsistent。
6. 保留 2 rounds、4 tool calls、12 evidence、12,000 tokens 上限，以及重复 query+scope、evidence IDs 无变化、coverage 无提升、预算耗尽、取消、模型或检索错误停止。第二轮后仅有限回答或 refuse，不得第三轮。
7. fixed/adaptive 使用相同问题、history、scope、index version、模型和评分口径；保存逐题 evidence、requirements、coverage、stops、rounds、tool calls、token、延迟与评分错误。

质量门槛：
- route 每类 recall ≥ 0.75，macro F1 ≥ 0.80。
- adaptive-eligible 子集至少 5 条 requirement coverage 改善，退化不超过 2 条。
- fixed-eligible 子集报告误触发率，且 citation/support 不得低于 fixed。
- 总体 citation correctness、citation completeness、major fact support rate 不低于 fixed；unsupported major claim count 不高于 fixed。
- successful termination=100%，平均轮数≤1.5，每题 tool calls≤4，exact duplicate query+scope=0；延迟仅记录。

验证：
- 新增 route、预算、claim validation、scoring consistency 和分层评测测试。
- uv run --extra dev python -m pytest tests/test_agent_graph.py tests/test_agent_budget.py tests/test_claim_validation.py tests/test_route_eval.py -q
- uv run python -m evals.runner --config evals/configs/v2_m4_1_2_route.yaml
- uv run python -m evals.runner --config evals/configs/v2_m4_1_2_answer.yaml
- uv run --extra dev ruff check agent core evals tests
- uv run --extra dev python -m pytest -q
- npm --prefix web run lint
- npm --prefix web run build

交付：
- 创建 docs/implementation/m4_1_2_adaptive_eval_acceptance.md，记录冻结 hash、authoring snapshot、逐题报告、分层结果、混淆矩阵、预算、评分误判、盲审清单、坏例、默认策略和回滚方式。
- 只有所有门槛通过才可提出是否更新 M4.1 结论；M4.2 仍需用户单独批准，绝不自动启动。
- 仅在实现与评测可复现后创建一个独立 commit，不推送；完成后停止。
```

## Goal 4B：M4.2 持久 Run 与恢复（终止，不执行）

> 本 Goal 仅保留历史设计，禁止复制、执行或以任何方式恢复为待办。M4.1 系列未通过质量门槛，M4.2 的进入条件永久不满足。

```text
目标：在 M4.1 已证明 Adaptive 有净收益后，把可选 Adaptive 链路接入持久 run、单 worker、LangGraph checkpoint 和可重连事件流。M4.2 解决恢复和幂等，不重新设计或调优 M4.1 策略。

进入条件：
- docs/implementation/m4_1_acceptance.md 存在，m4_1_quality_passed=true、m4_2_entry_ready=true。
- M4.1 的数据集、配置、逐题结果和 commit 可复现。
- 用户明确批准执行 M4.2。任一条件不满足就停止，不得为了建设运行架构放宽 M4.1 质量门槛。

开始前必须读取 AGENTS.md、web/AGENTS.md、docs/research/v2_upgrade_plan.md 第 4、7、8、9、10、11、12、15 节、M1/M3.2/M4.1 验收报告和 docs/implementation/m4_2_durable_run_handoff.md。修改前端前读取当前安装版本的 Next.js 文档。

执行边界：
- 不修改 M4.1 的 route、证据充分性、补检预算、评分器和冻结评测集。
- 保留 /api/chat fixed 链路和 ANSWER_STRATEGY=fixed 回滚路径。
- 不实现完整技术调试工作台、Compare、Workspace、Docker、Redis、Celery、PostgreSQL 或外部任务服务。
- SQLite 方案只承诺单机、单用户、单 run worker 的可恢复演示，不表述为通用高并发生产架构。
- GraphState 只保存 JSON 基础类型和紧凑控制状态，禁止保存 Document、完整候选、完整会话、模型 client 或数据库连接。

必须完成：
1. 新增 chat_messages、runs、run_events、retrieval_candidates 和 evidence_items migration。数据库迁移 forward-only，迁移前保留可恢复备份。
2. run 状态为 queued -> running|cancel_requested -> completed|failed|cancelled。同一 session 同时只允许一个 queued 或 running run，第二个请求返回 409。
3. 创建 run 时事务性写入 history_snapshot_json、index_version、baseline_config_hash、initial_state 和预算。
4. thread_id=run_id。使用独立 CHECKPOINT_DB_PATH、AsyncSqliteSaver 和 JsonPlusSerializer(pickle_fallback=False)，锁定与当前 LangGraph 兼容的 langgraph-checkpoint-sqlite 版本并做最小恢复测试。
5. 复用 index worker 的 lease、heartbeat、启动扫描和有限重试模式，新增单 run worker。lease 为 30 秒，heartbeat 为 10 秒，最大尝试 2 次。
6. checkpoint 只能保证节点边界恢复。所有数据库副作用必须用唯一键和 upsert 保持幂等，事件需要稳定 idempotency key。
7. final answer、evidence、claims、assistant message 和 completed 状态必须在同一事务成功。
8. 实现 POST /api/runs、GET /api/runs/{run_id}、GET /api/runs/{run_id}/events 和 POST /api/runs/{run_id}/cancel。
9. SSE 以持久 run_events.seq 为准，支持 Last-Event-ID；不得把进程内 LangGraph stream 当作重连数据源。
10. SSE 只发送进度、确认后的 evidence、validation.completed 和一次 answer.final，不发送 provisional answer、prompt 或未校验模型 token。
11. 前端只接入创建 run、进度、取消、断线恢复、最终答案和 evidence cards。完整调试 trace 留给 M5。
12. ANSWER_STRATEGY=fixed 时完整绕过 Adaptive run，旧 fixed chat 保持可用。

验证：
- uv run --extra dev python -m pytest tests/test_run_repository.py tests/test_run_recovery.py tests/test_run_streaming.py tests/test_agent_budget.py -q
- uv run --extra dev ruff check agent api core tests
- uv run --extra dev python -m pytest -q
- npm --prefix web run lint
- npm --prefix web run build

必须通过故障注入：
- 第一轮检索完成后终止进程并重启。
- final answer 事务写入前终止进程并重启。
- lease 过期后启动第二 worker 竞争。
- SSE 断线后使用 Last-Event-ID 继续。
- 同一 session 并发创建两个 run。
- cancel_requested 分别发生在排队、检索和生成节点边界。

成功条件：
- 最多一个 worker 持有 lease。
- run 从 checkpoint 恢复，或在没有 checkpoint 时使用保存输入幂等重启。
- assistant message、evidence、claims、事件和工具副作用均不重复。
- 同 session 的第二个并发 run 返回 409。
- 已完成 run 的 checkpoint 可清理，最终答案和 evidence 不受影响。
- fixed 回滚链路不依赖新 worker 或 checkpoint 数据库。

交付：
- 创建 docs/implementation/m4_2_acceptance.md，记录 migration、状态机、GraphState 大小、恢复协议、依赖版本、故障注入、SSE、前端检查、回滚方法和实际修改文件。
- 仅在 M4.2 验收通过后创建独立 commit，不推送。
- 完成后停止，不得自动执行 M5。
```

## Goal 5：M5 证据导向的固定 RAG Web 应用（待授权）

```text
目标：交付一个可日常使用、可在面试中演示的固定 RAG Web 应用。以全站统一的证据阅读体验为主线，完成生产级的视觉重构，并让 Chat 的每条回答保存和展示其结构化证据。

进入条件：
- 用户明确批准执行本 Goal。

开始前读取 AGENTS.md、web/AGENTS.md、docs/research/v2_upgrade_plan.md、当前 Chat API、Chat 页面、引用组件和相关测试。修改 Next.js 前读取当前安装版本的相关官方文档。

执行时以 `docs/research/m5_fixed_product_implementation_plan.md` 为详细手册；本 Goal 只定义授权边界和验收门槛。

执行边界：
- 不改检索策略、graph、索引、数据库 schema、模型调用、SSE 连接模型或普通用户设置。
- 复用现有 evidence schema 和会话 JSON 存储；不能只保存 Markdown，不能丢失原有 assistant 消息。
- 页面设计使用 UI skill，以根目录既有的 `DESIGN.md` 为准。采用中文、桌面优先的“纸刊学术编辑部”方向：暖白纸面、墨黑字、细分隔线、墨蓝证据标记；不得做成后台模板、渐变玻璃卡片或圆角卡片墙。
- 不新增前端依赖；保留基础响应式可用性，并完成桌面和 375px 宽度的检查。

必须完成：
1. 落实既有 `DESIGN.md`，重构全局导航、首页、论文库、搜索、论文阅读和 Chat 的布局、字体、颜色、规则、表单、状态和可访问性；确保页面不是同质化卡片网格。
2. assistant message 增加可选的结构化 evidence，SSE evidence 事件和 `GET /api/chat/{session_id}` 返回相同字段。
3. `POST /api/chat` 后的流式完成路径将 answer 与其 evidence 一起写入会话；历史会话读取后每轮 answer 仍能显示自己的 evidence。
4. 实现 evidence rail：每张证据条目显示论文名、章节、页码、quote 与论文跳转链接。没有 evidence 时显示明确的空状态。
5. 补齐 API、会话持久化、连续两轮回答、刷新回看、无 evidence 与 PDF 跳转链接的测试。

验证：
- uv run --extra dev python -m pytest -q
- uv run --extra dev ruff check .
- npm --prefix web run lint
- npm --prefix web run build
- 手工提问两轮、刷新页面、重开会话并逐个打开证据卡片。检查桌面和 375px 宽度。
- 实现、全量验证完成后，调用独立 review subagent 审查 API 兼容、会话持久化、证据完整性、可访问性、性能和回归风险。首选模型 `gpt-5.6-luna`，reasoning effort 设为 `max`。reviewer 只报告问题，不直接修改代码；实现者必须修复所有可修复问题并重跑受影响验证。仅不成立或超出本 Goal 范围的问题可不修复，且必须在验收报告说明判断依据。

交付：
- 创建 docs/implementation/m5_fixed_product_acceptance.md，记录 API 兼容、会话回看、全站视觉检查、验证、review 模型与结论、坏例、回滚方式和实际修改文件。
- 仅在全部验证通过后创建独立 commit，不推送。完成后停止，等待用户决定是否执行 M6。
```

## Goal 6：M6 评测实验室

```text
目标：把现有可复现评测转成只读实验室展示，让访问者理解为何产品固定使用 B1，而不是让其配置或运行未经证明的检索策略。

进入条件：
- `docs/implementation/m5_fixed_product_acceptance.md` 存在且 M5 验收通过。
- M3.2、M4.1.1、M4.1.2 的报告和验收文档均存在，工作区保留其实际结论。
- 用户明确批准执行本 Goal。

开始前读取 AGENTS.md、web/AGENTS.md、根目录 DESIGN.md、docs/research/v2_upgrade_plan.md、M3.2/M4.1.1/M4.1.2/M5 验收报告。修改 Next.js 前读取当前安装版本的相关官方文档，并使用 UI skill 设计和检查实验室页面。

执行时以 `docs/research/m6_evaluation_lab_implementation_plan.md` 为详细手册；本 Goal 只定义授权边界和验收门槛。

执行边界：
- 实验室只读已提交的报告与经过脱敏的摘要，不触发模型调用、不运行评测、不修改 active index。
- 不将 S1、B2/B3 或 Adaptive 做成面向普通用户的产品配置，也不把实验室结果表述为线上质量承诺。
- 不实现 M4.2、Compare、Workspace、持久 run、checkpoint、队列、Docker 或新外部服务。
- 页面遵循 `DESIGN.md` 的证据导向视觉系统，作为现有 Web 应用的一部分，而不是独立的后台或数据看板模板。

必须完成：
1. 展示固定 B1、S1、M4.1.1、M4.1.2 的目的、数据集版本、关键指标、逐题胜平负或混淆摘要、延迟与代表坏例。
2. 清晰显示晋级决策：S1 因 holdout Context Recall 未通过，Adaptive 因路由误触发和答案质量未证明净收益，产品默认保持 fixed。
3. 检查页面和数据不暴露 API Key、prompt、完整本地路径、未脱敏问题或模型凭据。

验证：
- uv run --extra dev python -m pytest -q
- uv run --extra dev ruff check .
- npm --prefix web run lint
- npm --prefix web run build
- 手工核对实验室指标与原始验收报告一致。
- 实现、全量验证完成后，调用独立 review subagent 审查只读数据边界、指标与原始报告的一致性、脱敏、前端性能、可访问性和回归风险。首选模型 `gpt-5.6-luna`，reasoning effort 设为 `max`。reviewer 只报告问题，不直接修改代码；实现者必须修复所有可修复问题并重跑受影响验证。仅不成立或超出本 Goal 范围的问题可不修复，且必须在验收报告说明判断依据。

交付：
- 创建 docs/implementation/m6_evaluation_lab_acceptance.md，记录数据来源、脱敏规则、页面检查、演示稿、验证、review 模型与结论和回滚方式。
- 仅在全部验证通过后创建独立 commit，不推送。完成后停止，等待用户决定是否执行 M7。
```

## Goal 7：M7 项目设计与面试指南

```text
目标：编写一份深入浅出的中文项目指南，帮助项目作者理解并讲清这个项目的用户问题、架构、检索、证据、评测、失败决策和工程取舍。

进入条件：
- M6 验收报告存在且用户明确批准执行本 Goal。

执行边界：
- 以当前代码、M1 至 M6 验收报告和冻结评测资产为唯一事实来源。
- 清楚区分已实现能力、实验失败结论和未实施计划，不写虚构指标或能力。
- 不修改运行代码、评测数据、默认策略或外部依赖。

必须完成：
1. 在 `docs/` 新增中文项目指南，解释用户问题、模块关系、上传到回答的数据流、B1 固定检索、引用与索引契约。
2. 解释 M3/M4 的实验设计、门槛、结果和为什么保留 B1。
3. 提供至少 15 组基于事实的面试问题与回答框架，包括技术选型、评测、失败决策、可靠性、边界和下一步。
4. 每个关键结论链接到对应代码或验收文档，并提供 10 分钟演示顺序。

验证：
- git diff --check
- 手工按指南完成 10 分钟项目讲解，并随机追问 5 个指标或设计决定。

交付：
- 创建 docs/implementation/m7_project_guide_acceptance.md，记录事实来源、讲解检查、已知限制与修改文件。
- 仅在全部检查通过后创建独立 commit，不推送。完成后停止。
```
