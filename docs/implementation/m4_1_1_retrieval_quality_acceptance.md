# M4.1.1 检索质量复验验收

## 结论

- `m4_1_1_retrieval_quality_passed=false`
- `m4_1_quality_passed=false`（原 M4.1 失败结论不变）
- `m4_2_entry_ready=false`
- 默认仍为 `ANSWER_STRATEGY=fixed`；回滚方式为不设置该变量或显式设置为 `fixed`。

本复验没有证明“证据缺口判断 + 一次补检”优于冻结 B1。安全预算和停止行为合格，但 route 的 fixed 类召回以及回答质量门槛未通过。因此不进入 M4.2。

## 冻结输入与可复现命令

- route：`evals/datasets/m4_1_1_route_v1.json`，SHA-256 `01372eff9bea8d7a96b8bafb5ac13402d5342a4cc093632ee1f63d6143cbdd7d`，48 条（四类各 12 条）。
- answer：`evals/datasets/m4_1_1_answer_v1.json`，SHA-256 `e7eadb3d63a199b87d2f7c1fe3ea16ba0bb9bbf1a8bfe9df98126589b07abbaa`，24 条（M3 困难类型 12 条、独立问题 12 条）。
- 冻结 B1：`v1_flat_rerank`，hash `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`；评测 index manifest 为 `artifacts/evals/v2_m3_2/old_dev/manifests/b1.json`。
- 模型：规划、充分性、补检、回答和 claim 判断均为 `qwen3.7-max-2026-06-08`。仅替换 RAG LLM，不改变检索、embedding 或 active index。
- 命令：`uv run python -m evals.runner --config evals/configs/v2_m4_1_1_route.yaml` 和 `uv run python -m evals.runner --config evals/configs/v2_m4_1_1_answer.yaml`。
- 正式原始结果：`artifacts/evals/v2_m4_1_1/m4_1_1_route_report.json`、`artifacts/evals/v2_m4_1_1/m4_1_1_answer_report.json`。M3.2 holdout 未重跑、未复用；问题、标签、gold、阈值和评分器未在正式结果后更改。

## Route 结果

| 类别 | Recall |
| --- | ---: |
| direct | 1.0000 |
| fixed | 0.5833 |
| adaptive | 0.8333 |
| refuse | 1.0000 |

- macro F1：`0.8519`。
- 混淆矩阵（expected → predicted）：direct `12/0/0/0`、fixed `0/7/5/0`、adaptive `0/2/10/0`、refuse `0/0/0/12`，列序为 direct/fixed/adaptive/refuse。
- 失败：fixed recall 低于 `0.75`。误判为 adaptive 的 fixed case 是 `route-m411-fixed-03/06/08/09/10`；误判为 fixed 的 adaptive case 是 `route-m411-adaptive-01/02`。
- 这说明首轮“充分性”判断仍会把部分单跳问题送入补检；不能用 M3 困难标签在运行时硬编码修正。

## Answer 结果

| 指标 | fixed B1 | adaptive |
| --- | ---: | ---: |
| requirement coverage | 0.0000 | 0.0000 |
| citation correctness | 0.0000 | 0.0000 |
| citation completeness | 0.0000 | 0.0000 |
| major fact support rate | 0.0000 | 0.0000 |
| unsupported major claims（均值） | 2.3750 | 2.7083 |
| 平均检索轮数 | 1.0000 | 1.7500 |
| 平均 tool calls | 0.9583 | 1.7500 |
| 平均上下文 tokens | 826.7083 | 1164.9583 |
| p95 总延迟 ms（仅记录） | 115498.6888 | 339452.6559 |

- coverage 改善 `0`，退化 `0`：未达到至少改善 5 条。
- adaptive unsupported major claims 高于 fixed：不通过。
- successful termination rate=`1.0`；最大轮数为 2；每题 tool calls 不超过 4；exact duplicate query + scope=`0`；`coverage_not_improved` 停止 11 次。这些预算和停止门槛通过。
- 延迟按用户授权仅记录，不作为晋级门槛。

## 评分误判与坏例

- `answer-m411-10` 的 fixed 检索出现 `retrieval_error`，而 adaptive 的第二轮完成；这是唯一展示补检可能价值的坏例，但不足以形成质量提升。
- `answer-m411-03/08/11/18` 等 case 中，adaptive 的最终可支持 claim 少于 fixed 或未形成可接受引用；补检没有转化成最终主要事实支持。
- 结构化 claim 评分的布尔字段与其正向文字理由存在自相矛盾：fixed 14 处、adaptive 6 处。正式成绩严格保留冻结评分器输出，未以事后规则修复这些布尔值；它们作为模型误判记录，而非确定性语义证明。
- 某些有效 evidence 与数据集 `acceptable_evidence_ids` 不一致。由于冻结评分口径要求 claim 引用落在该集合，正式结果不将它们计为支持；未在结果后放宽 gold 或评分器。

## 交付边界

- Adaptive 路径保持至多 3 个 requirements、2 轮、4 次 retrieval、12 条 evidence、12,000 tokens，并保留重复查询、证据不变、coverage 不升、预算、取消和模型错误停止。
- 没有新增 database migration、worker、checkpoint、SSE 或前端模式。
- 本文只结束 M4.1.1 复验；不得自动执行 M4.2。
