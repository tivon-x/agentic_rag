# M4.1.2 Adaptive 场景对齐复验交接

## 当前结论

- 分支：`codex/v2-core`，收口提交：`cec8918 feat: add bounded adaptive retrieval evaluation`。
- M3.2 的 B1 仍冻结为 `v1_flat_rerank`，config hash 为 `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`。
- M4.1.1 已完成但未通过：route 的 fixed recall=`0.5833`；answer 集没有 coverage 改善；Adaptive 的 unsupported major claims 高于 fixed。
- 安全约束通过：100% 正常停止、最多 2 轮/4 次 retrieval、无重复 query+scope、coverage 不提升会停止。
- 继续保持 `ANSWER_STRATEGY=fixed`，`m4_1_quality_passed=false`、`m4_2_entry_ready=false`。禁止启动 M4.2。

详见 `docs/implementation/m4_1_1_retrieval_quality_acceptance.md` 和正式逐题报告：

- `artifacts/evals/v2_m4_1_1/m4_1_1_route_report.json`
- `artifacts/evals/v2_m4_1_1/m4_1_1_answer_report.json`

## 为什么需要 M4.1.2

M4.1 的价值不在“题目看起来复杂”，而在首轮 B1 只覆盖部分需求、且一次有针对性的补检能补到缺口。M4.1.1 的 route 集混入了首轮证据判断认为不足的 fixed 标签；answer 集也未充分验证“部分覆盖 → 定向补检 → 最终引用新增证据”的完整链路。

这不是对 M4.1.1 结果的重解释或重算。M4.1.2 必须使用新的、此前未执行过的冻结数据集和新的评分协议，M4.1.1 的问题、标签、gold、grader、报告一律不改。

## 新会话开始前

依次读取：

1. `AGENTS.md`
2. `docs/research/v2_upgrade_plan.md` 第 6、7、9、10、11、12、15 节
3. `docs/research/phase2_goal_prompts.md` 的 Goal 4A 和 Goal 4A.1
4. `docs/implementation/m3_2_strategy_acceptance.md`
5. `docs/implementation/m3_2_strategy_per_question.md`
6. `artifacts/evals/v2_m3_2/m4_fixed_baseline.json`
7. `docs/implementation/m4_1_1_retrieval_quality_protocol.md`
8. `docs/implementation/m4_1_1_retrieval_quality_acceptance.md`
9. 本文件
10. `agent/adaptive.py`、`agent/adaptive_graph.py`、`agent/prompts.py`、`agent/schemas.py`、`evals/m4_1_1_runner.py`

先确认：工作区干净、HEAD 含 `cec8918`、active index 未改变、B1 contract 仍匹配。真实评测会调用外部模型，必须在调用前取得当次用户授权。

## M4.1.2 的问题定义

只验证一个狭义命题：在 **首轮部分覆盖且补检有可获得缺口证据** 的事实型问题上，bounded Adaptive 是否优于同一 B1 的一轮 fixed。

同时保留反例集，验证它不会把首轮已经充分的单跳问题过度送进补检。运行时只能依据首轮 evidence sufficiency，不得使用数据集类别、M3 标签、case ID 或 gold 规则。

## 数据集与冻结流程

在改动 prompt、route、planner、assessor、follow-up、answerer 或 grader 前，创建新版本：

- `evals/datasets/m4_1_2_route_v1.json`
- `evals/datasets/m4_1_2_answer_v1.json`
- `evals/datasets/m4_1_2_dataset_manifest.json`
- `evals/configs/v2_m4_1_2_route.yaml`
- `evals/configs/v2_m4_1_2_answer.yaml`

可使用一次**只读 B1 authoring snapshot**挑选候选题，但必须在 manifest 中记录其代码/索引/模型、时间和每题首轮证据摘要；不得运行 Adaptive 来挑题，不得基于 M4.1.2 的正式结果回改。冻结后计算 SHA-256，并只运行正式 answer/route 各一次。

Route 仍为 48 条、四类各 12 条。fixed 的 12 条必须经 authoring snapshot 确认其全部 requirements 有一轮可定位证据；adaptive 的 12 条必须确认至少一个 requirement 首轮缺失，且库中存在可由不同定向 query 发现的页码可定位 evidence。该确认仅用于数据编写，不能输入运行时。

Answer 至少 24 条，建议分层如下：

- 12 条 adaptive-eligible：跨章节、跨论文比较、术语/缩写变体、方法与实验设置组合、正文与表格/附录联合支撑等；每题 2–3 个 requirements。
- 8 条 fixed-eligible：一轮 B1 足够的单跳问题，用于测量过度补检。
- 4 条明确证据不足：检验有限回答或 refuse，不能以第三轮检索掩盖缺口。

新数据集不得与 M3.2 holdout 或 M4.1.1 的原问题重复或近似改写。

## 评分协议（必须先冻结）

将三类判断分开记录，避免把其中任一类冒充另一类：

1. **确定性引用有效性**：claim 的 evidence ID 实际在本 run 返回；当前 index version 一致；quote 非空；paper/section/page 可定位。
2. **语义支持**：冻结的结构化 grader 仅根据 claim 和所引 quote 判断是否直接支持。其输出必须包含明确布尔值和理由；布尔值与理由矛盾时记为 `grader_inconsistent`，不事后自动改分。预先定义其如何进入主分和错误率。
3. **gold 覆盖审计**：gold/acceptable evidence 是要求和作者标注的审计锚点，用于诊断检索缺口与漏证；不可在正式结果后扩充。若主分要求 exact ID，必须在冻结前证明该口径不会把同一事实的等价、可定位 evidence 系统性判零；否则主分以“有效引用 + 语义支持 + requirement 映射”为准，gold ID 仅作为独立审计列。

在正式运行前固定：claim-to-requirement 映射、无 claims 的处理、grader error/inconsistent 的分母、人工盲审抽样比例和所有阈值。至少对 20% case 做盲审清单；盲审结果只报告 grader false positive/negative/inconsistent，不得用来事后更改自动主分。

## 策略改进方向

只允许改变 M4.1 策略质量层，不改 B1：

- 首轮检索始终保留完整原问题，避免 planner 改写丢失限定条件。
- sufficiency 必须逐 requirement 列出 evidence IDs、coverage、missing reason，并要求指出“下一查询为什么能补到缺口”。
- follow-up 只可查询缺失 requirement，且必须与首轮 query+scope 不重复。
- final answer 只保留有实际 evidence ID 的主要事实；第二轮新增 evidence 必须可被追溯到新增/补全的 requirement。
- 保留 3 requirements、2 rounds、4 tool calls、12 evidence、12,000 tokens 和所有既有停止条件。

先在单独 dev/calibration 集分析 M4.1.1 暴露的失败类型；不得用最终冻结集调 prompt 或阈值。每次实现修正都要有边界单测。

## 门槛与交付

质量门槛沿用原 M4.1 的安全限制；在冻结 M4.1.2 协议中另明确：

- route 每类 recall ≥ 0.75，macro F1 ≥ 0.80；
- adaptive-eligible 子集的 requirement coverage 改善至少 5 条，退化不超过 2 条；
- fixed-eligible 子集必须报告误触发率，且不能因补检使 citation/support 指标低于 fixed；
- citation correctness、citation completeness、major-fact support rate 不低于 fixed，unsupported major claims 不高于 fixed；
- termination=100%，平均轮数 ≤ 1.5，每题 tool calls ≤ 4，duplicate query+scope=0；延迟仅记录。

创建 `docs/implementation/m4_1_2_adaptive_eval_acceptance.md`，保存 hashes、authoring snapshot、完整逐题报告、混淆矩阵、分层指标、评分错误、盲审清单、坏例、默认策略和回滚方式。

无论结果如何都只创建一个独立 commit，不推送，并停止。只有用户明确审阅后才可以讨论是否重新定义 M4.1 或启动 M4.2。
