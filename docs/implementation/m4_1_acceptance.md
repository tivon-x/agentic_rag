# M4.1 有界 Adaptive 质量闭环验收

## 结论

- `m4_1_quality_passed=false`
- `m4_2_entry_ready=false`
- 默认策略：`ANSWER_STRATEGY=fixed`
- 回滚：显式设置或保持 `ANSWER_STRATEGY=fixed`；fixed graph 和 fixed chat 不依赖 `AdaptiveGraphState`。

本次未进入 M4.2。冻结 B1 上的补检在 answer coverage 上有改善，但 route、引用正确性、平均检索轮数和 p95 延迟未通过质量门槛。

## 冻结输入与可复现性

| 项目 | 值 |
| --- | --- |
| route dataset | `evals/datasets/m4_1_route_v1.json`，48 条 |
| route SHA-256 | `5107ec4bbb2f2d5bd0db0079db979b57c2e4ffc3dc369bc56eeaaf151f8c0a4a` |
| answer dataset | `evals/datasets/m4_1_answer_v1.json`，24 条 |
| answer SHA-256 | `00219523b7178713b242a966b7b72df1cf88f5794eef6674984b5657a7db0fd2` |
| dataset manifest | `evals/datasets/m4_1_dataset_manifest.json` |
| baseline contract | `artifacts/evals/v2_m3_2/m4_fixed_baseline.json` |
| frozen pipeline | `v1_flat_rerank` |
| config hash | `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17` |
| evaluation index | M3.2 `old_dev` 的只读 B1 索引；未运行或修改 M3.2 holdout，active index 未改变 |
| model / embedding | `qwen3.7-flash` / `qwen3.7-text-embedding` |

完整、逐题可复现产物：

- `artifacts/evals/v2_m4_1/m4_1_route_report.json`
- `artifacts/evals/v2_m4_1/m4_1_answer_report.json`

## Route 结果

| 真实 \ 预测 | direct | fixed | adaptive | refuse |
| --- | ---: | ---: | ---: | ---: |
| direct | 12 | 0 | 0 | 0 |
| fixed | 0 | 6 | 4 | 2 |
| adaptive | 0 | 2 | 3 | 7 |
| refuse | 0 | 0 | 0 | 12 |

| 指标 | 结果 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| direct recall | 1.0000 | >= 0.75 | pass |
| fixed recall | 0.5000 | >= 0.75 | fail |
| adaptive recall | 0.2500 | >= 0.75 | fail |
| refuse recall | 1.0000 | >= 0.75 | pass |
| macro F1 | 0.6608 | >= 0.80 | fail |
| successful termination | 1.0000 | = 1.00 | pass |
| average rounds | 0.8125 | <= 1.5 | pass |
| average tool calls | 1.6250 | <= 4 | pass |
| route p95 latency | 160948.3844 ms | record | recorded |

主要坏例：`route-fixed-03`、`route-fixed-04` 被拒答；`route-adaptive-01`、`03`、`04`、`09`、`10` 被拒答；`route-adaptive-05`、`06` 被第一轮误判为 fixed。完整停止原因见 route JSON。

## Answer 对照结果

| 指标 | fixed B1 | adaptive | 门槛 | 结论 |
| --- | ---: | ---: | --- | --- |
| requirement coverage | 0.5625 | 0.7917 | adaptive 改善 >= 5、退化 <= 2 | 8 改善、1 退化，pass |
| citation correctness | 0.0990 | 0.0923 | adaptive 不低于 fixed | fail |
| citation completeness | 0.5625 | 0.7917 | adaptive 不低于 fixed | pass |
| major-fact support-rate | 0.0990 | 0.0923 | adaptive 不低于 fixed | fail |
| unsupported major claim count | 0 | 0 | adaptive 不高于 fixed | pass |
| average retrieval rounds | 1.0000 | 1.6250 | adaptive <= 1.5 | fail |
| average tool calls | 1.0000 | 3.0833 | 每题 <= 4 | pass（最大 4） |
| p95 total latency | 1099.3602 ms | 146489.7099 ms | adaptive <= 2.5x fixed | fail（133.25x） |
| context tokens | 798.3333 | 1341.7917 | record | recorded |
| duplicate query + scope | — | 0 | = 0 | pass |
| successful termination | — | 1.0000 | = 1.00 | pass |

`answer-m4-09` 是 requirement coverage 的唯一退化；citation correctness 退化的逐题项为 `answer-m4-03`、`05`、`07`、`08`、`09`、`12`、`13`、`14`、`15`、`17`、`18`、`20`、`22`、`24`。每题 fixed/adaptive 的 evidence IDs、coverage、rounds、tool calls、latency、tokens 和停止原因均在 answer JSON 中保存。

## 评分口径与误判边界

- requirement coverage、citation completeness：gold evidence ID 的覆盖比例。
- citation correctness、major-fact support-rate：citation evidence ID 与 gold evidence ID 的交集比例。
- unsupported major claim count：结构化 answer 中没有通过 evidence-ID 完整性校验的 major claim 数量。
- quote 对 requirement/claim 的语义支持是模型判断；确定性检查只验证 ID 存在、同一 index version、quote 非空、页码及 claim evidence ID 完整性。
- 本次模型语义判断的可复核 proxy：没有发现“判断 covered 但没有 gold evidence 命中”的 case 级 false positive，也没有发现“判断未覆盖但已命中 gold evidence”的 case 级 false negative。该 proxy 不等同于人工语义证明；因为引用正确性和 major-fact support-rate 已低于 fixed，本 gate 仍判失败。

## 预算与实现边界

- 独立 `AdaptiveGraphState` 只保存控制信息、ID、覆盖、预算和最终小结果。
- 每次事实检索校验冻结 B1 contract；没有修改 dense、BM25、fusion、reranker、top-k 或 context packing。
- 上限：3 个 requirement、2 轮、4 个 retrieval tool calls、12 条 evidence、12,000 context tokens。
- 停止原因覆盖：重复 query+scope、evidence IDs 不变、coverage 无提升、预算耗尽、取消、模型错误、检索错误和第二轮结束。
- 未新增 migration、run worker、checkpoint saver、SSE 协议或前端技术模式。

## 验证

- 定向 M4.1 pytest：10 passed。
- 全量 pytest：264 passed（最终工作区）。
- `ruff check agent core evals tests`：passed。
- `npm --prefix web run lint`：passed。
- `npm --prefix web run build`：passed。

本里程碑因上述质量门槛失败而结束，等待用户决定后续方向；不得自动开始 M4.2。
