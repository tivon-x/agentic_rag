# M4.1 有界 Adaptive 质量闭环交接

## 1. 当前状态

- 工作分支：`codex/v2-core`
- M3.2 完成提交：`5982305 feat: close M3.2 fixed retrieval strategy`
- `m3_1_core_passed=false`
- `m3_strategy_closed=true`
- `m4_entry_ready=true`
- 默认 fixed pipeline：`v1_flat_rerank`
- M4 fixed baseline：`v1_flat_rerank`
- baseline config hash：`ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`
- formal holdout run count：`1`
- metadata prefix leak：`0`
- active index 未改变

M3.2 的 S1 在 holdout 上 Context Passage Recall 为 `0.854167`，低于 B1 的
`0.875000`，因此未通过。M4.1 不再调整固定检索参数，只验证 B1 之上的证据缺口
判断和一次定向补检。

## 2. 开始前检查

依次执行：

```powershell
git status --short --branch -uall
git rev-parse HEAD
```

读取：

1. `AGENTS.md`
2. `docs/research/v2_upgrade_plan.md`
3. `docs/research/phase2_goal_prompts.md` 的 Goal 4A
4. `docs/implementation/m3_2_strategy_acceptance.md`
5. `docs/implementation/m3_2_strategy_per_question.md`
6. `artifacts/evals/v2_m3_2/m4_fixed_baseline.json`
7. `agent/graph.py`
8. `agent/nodes.py`
9. `agent/edges.py`
10. `agent/states.py`
11. `agent/prompts.py`
12. `agent/schemas.py`
13. `agent/tools.py`
14. `core/settings.py`
15. `evals/` 下现有 runner、schema、grader 和报告实现

必须保留用户已有的未跟踪文件：

- `docs/implementation/m3_1_experiment_handoff.md`
- `docs/implementation/m3_2_strategy_handoff.md`

如果 HEAD、baseline contract、验收字段或工作区状态与本文件不一致，先确认差异。
不得重新运行 M3.2 holdout，也不得覆盖不属于 M4.1 的用户改动。

真实评测会把问题、候选证据和生成上下文发送给现有模型服务。调用外部模型前需要
取得用户授权，不得把 API Key、完整环境变量或未脱敏的本地路径写入数据集和报告。

## 3. 目标和非目标

M4.1 要回答一个问题：冻结 B1 后，系统能否识别第一轮证据缺口，并通过最多一次
定向补检改善回答，同时控制退化、延迟和工具调用。

本 Goal 不做：

- 不修改 B1 的 sparse、dense、fusion、reranker、top-k 或 context packing。
- 不新增数据库 migration。
- 不新增 run worker、lease、checkpoint 或 SSE 重连。
- 不实现技术调试工作台。
- 不修改 M3、M3.1、M3.2 的数据集、结果和验收文档。
- 不把 adaptive 设为默认策略。
- 不安装 Docling，不新增外部服务。

## 4. 冻结基线

每次事实检索都必须加载：

`artifacts/evals/v2_m3_2/m4_fixed_baseline.json`

运行前校验：

- `selected_pipeline_name == "v1_flat_rerank"`
- `pipeline_config_hash == "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17"`
- index manifest、parser artifact、embedding provider、model、dimension、input mode 和
  max input chars 与 contract 一致

任一字段不一致立即失败，不允许用当前默认配置猜测或重建 contract。

## 5. 先冻结评测数据

在实现或调整 adaptive prompt、route、证据充分性判断前，创建：

- `evals/datasets/m4_1_route_v1.json`
- `evals/datasets/m4_1_answer_v1.json`
- `evals/datasets/m4_1_dataset_manifest.json`
- `evals/configs/v2_m4_1_route.yaml`
- `evals/configs/v2_m4_1_answer.yaml`

manifest 至少记录：

- schema version
- 创建时间
- route 和 answer 文件 SHA-256
- baseline contract path 和 config hash
- index version
- parser artifact hash
- embedding contract
- 问题数量和类别数量
- 冻结声明

数据冻结后，不得根据 test 运行结果修改问题、标签、gold evidence、阈值、grader
prompt 或评分权重。发现标注错误时记录为坏例，本 Goal 停止，由用户决定是否建立新
版本。不能直接覆盖 `v1`。

### 5.1 Route test

共 48 条，每类 12 条：

- `direct`：寒暄、确认、格式调整、对已有回答的压缩或重排
- `fixed`：第一轮 B1 足以覆盖的定义、数值、单篇或单章节定位
- `adaptive`：第一轮证据存在明确缺口，且一次定向补检有机会补齐
- `refuse`：论文库外实时事实、库外知识请求、无证据问题

每条至少包含：

- `id`
- `query`
- `history`
- `scope`
- `expected_route`
- `required_facts`
- `authoring_source`
- `notes`

`authoring_source` 只能标记 `m3_difficulty_taxonomy` 或 `independent`。不得把 M3
case ID、gold tag 或困难类别暴露给运行时 route。

### 5.2 Answer test

共 24 条：

- 12 条覆盖 M3 暴露的困难类型，包括跨论文、跨章节、表格数值、缩写和组合条件
- 12 条独立编写，不能改写 M3.2 holdout 问题

每条至少包含：

- `id`
- `query`
- `history`
- `scope`
- `requirements`
- `gold_evidence`
- `allowed_answer_shape`
- `refusal_expected`
- `authoring_source`

固定 B1 与 adaptive 必须使用同一问题、history、scope、index version、生成模型、
grader 和评分口径。

## 6. 实现设计

### 6.1 策略

保留现有 fixed graph，新增独立 `AdaptiveGraphState`。M4.1 的调用仍在现有同步或
异步 chat 边界内完成，不接入持久 run。

策略语义：

- `direct`：不检索，不产生新的论文事实
- `fixed`：第一轮 B1 已覆盖全部需求，直接生成和校验
- `adaptive`：第一轮 B1 后仍有可描述的证据缺口，补检一次
- `refuse`：请求超出论文库，或两轮后仍缺主要证据

事实型问题不能只根据问题表面复杂度直接决定是否补检。先拆解回答需求，再运行第一轮
B1，最后根据实际 evidence 判断。

### 6.2 需求和预算

- plan items 最多 3 个
- 第一轮最多 3 次 B1 检索
- 第二轮最多 1 次定向补检
- 总 retrieval tool calls 不超过 4
- 总 evidence 不超过 12
- 总上下文不超过 12,000 tokens
- 总检索轮数不超过 2

第二轮 query 只能覆盖第一轮的 missing requirements。禁止再次搜索已经覆盖的需求。

### 6.3 证据充分性

结构化输出至少包含：

- requirement ID
- covered
- evidence IDs
- coverage
- missing reason
- recommended follow-up query

确定性校验负责：

- evidence ID 存在
- evidence 属于当前 index version
- quote 非空
- paper、section 和 page 可定位
- claim 引用的 evidence ID 完整
- follow-up query 与已有 query 和 scope 不完全重复

模型判断负责 quote 与 requirement、claim 的语义支持关系。验收必须单独记录
false positive 和 false negative，不能把这部分描述成确定性验证。

### 6.4 停止和回答

出现任一条件立即停止补检：

- 已覆盖全部 requirements
- 已完成第二轮
- evidence IDs 与上一轮相同
- coverage 没有提升
- query 和 scope 完全重复
- tool call、evidence 或 context 预算耗尽
- 用户取消
- 模型或检索错误

第二轮仍缺证据时：

- 次要缺口：输出有限回答，并在 `limitations` 明确缺失项
- 主要缺口：`refuse`
- 不允许第三轮检索

每个主要事实 claim 必须声明 evidence IDs。不支持的 claim 删除、降低确定性措辞或
写入 limitations。

## 7. 配置和回滚

新增或确认：

```text
ANSWER_STRATEGY=fixed
```

允许值只有 `fixed` 和 `adaptive`。未知值启动失败，不能静默降级。

M4.1 完成后默认仍为 `fixed`。回滚只需设置 `ANSWER_STRATEGY=fixed`，现有 fixed
chat 不依赖 AdaptiveGraphState。

## 8. 自动验证

至少补齐：

- direct 不检索且不产生论文事实
- fixed 第一轮充分时不进入第二轮
- adaptive 只补检 missing requirements
- refuse 不检索库外实时事实
- plan items、轮数、tool calls、evidence、context 预算
- duplicate query + scope 停止
- evidence IDs 无变化停止
- coverage 无提升停止
- claim evidence ID 完整性
- evidence index version 和页码校验
- 第二轮不足时有限回答或拒答
- 模型错误和检索错误能够终止
- `ANSWER_STRATEGY=fixed` 完整绕过 adaptive
- baseline contract 不匹配时立即失败

执行：

```powershell
uv run --extra dev python -m pytest tests/test_agent_graph.py tests/test_agent_budget.py tests/test_claim_validation.py tests/test_route_eval.py -q
uv run python -m evals.runner --config evals/configs/v2_m4_1_route.yaml
uv run python -m evals.runner --config evals/configs/v2_m4_1_answer.yaml
uv run --extra dev ruff check agent core evals tests
uv run --extra dev python -m pytest -q
npm --prefix web run lint
npm --prefix web run build
```

## 9. 发布门槛

全部满足才算通过：

- route 每类 recall ≥ 0.75
- route macro F1 ≥ 0.80
- adaptive 相对 fixed B1 至少 5 条 requirement coverage 改善
- adaptive 相对 fixed B1 退化不超过 2 条
- citation correctness 不低于 fixed B1
- citation completeness 不低于 fixed B1
- 主要事实支持率不低于 fixed B1
- unsupported major claim count 不高于 fixed B1
- successful termination rate = 100%
- 平均检索轮数 ≤ 1.5
- 每题 tool calls ≤ 4
- exact duplicate query + scope = 0
- adaptive p95 总延迟 ≤ fixed B1 的 2.5 倍

任一条件失败：

- `m4_1_quality_passed=false`
- `m4_2_entry_ready=false`
- `ANSWER_STRATEGY=fixed`
- 不执行 M4.2

## 10. 交付

创建：

- `docs/implementation/m4_1_acceptance.md`
- `artifacts/evals/v2_m4_1/` 下的逐题结果、混淆矩阵、指标、延迟、token 和坏例

验收文件必须记录：

- 数据集路径和 SHA-256
- baseline contract 和 config hash
- fixed/adaptive 的逐题结果
- route confusion matrix
- requirement coverage 胜平负
- citation 和 claim 指标
- 平均轮数、tool calls、延迟和 token
- 证据充分性误判
- 默认策略
- 回滚方式
- 实际修改文件
- `m4_1_quality_passed`
- `m4_2_entry_ready`

只有实现、评测、完整测试、Ruff 和前端验证都通过后才创建一个独立 commit，不推送。
完成后停止，等待用户决定是否执行 M4.2。
