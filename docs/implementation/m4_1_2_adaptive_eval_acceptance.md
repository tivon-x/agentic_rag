# M4.1.2 Adaptive 场景对齐复验验收

## 结论

- `m4_1_2_retrieval_quality_passed=false`。
- 原 `m4_1_quality_passed=false` 和 `m4_2_entry_ready=false` 不变。
- 默认仍为 `ANSWER_STRATEGY=fixed`；回滚方式是不设置该变量或显式设为 `fixed`。
- M4.2 未启动，且不能因本次复验启动。

新冻结集没有达到 M4.1.2 的 route 或 answer 质量门槛。结果不覆盖、不重算也不修改 M4.1.1。

## 冻结输入与快照

- route：`evals/datasets/m4_1_2_route_v1.json`，SHA-256 `16f39e49af1284088e757f6fac859e07f55cee8f1ca80c897aa9a42be6dbd0ba`，48 条、四类各 12。
- answer：`evals/datasets/m4_1_2_answer_v1.json`，SHA-256 `d7afeffce1f567fc595ad577df9114d08c0a2d556fb8b0d516fcee025625fb03`，24 条：12 adaptive-eligible、8 fixed-eligible、4 evidence-insufficient。
- authoring snapshot：`artifacts/evals/v2_m4_1_2/m4_1_2_authoring_snapshot.json`，SHA-256 `faf3ea1585af9c7f31f3c6e3b788cb39a53b8f999e0251680531995ee3ee74c1`；只读 B1、未调用 Adaptive。
- B1：`v1_flat_rerank`，config hash `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`；评测索引为 `artifacts/evals/v2_m3_2/old_dev/manifests/b1.json`。
- 协议：`docs/implementation/m4_1_2_retrieval_quality_protocol.md`。

## 正式报告

- route report：`artifacts/evals/v2_m4_1_2/m4_1_2_route_report.json`，SHA-256 `cc7b3199cb324058a08205c657c3443acb05092fb43cb2a85cb19b70041d8c06`。
- answer report：`artifacts/evals/v2_m4_1_2/m4_1_2_answer_report.json`，SHA-256 `b21e6894e685f1a8a3713ee77cea42429fc4c1ea634b5d6571380525675229a1`。
- 两份报告保存每题 query、requirements、evidence、coverage、stop、rounds、tool calls、tokens、latency、grader 输出和盲审清单。

## Route

| 类别 | Recall |
| --- | ---: |
| direct | 0.7500 |
| fixed | 0.4167 |
| adaptive | 0.7500 |
| refuse | 0.8333 |

- macro F1=`0.7029`，未达 `0.80`；fixed recall 未达 `0.75`。
- 混淆矩阵（expected → direct/fixed/adaptive/refuse）：direct `9/0/3/0`、fixed `0/5/7/0`、adaptive `0/3/9/0`、refuse `0/0/2/10`。
- fixed 的 7 个误触发为 adaptive，说明当前 evidence sufficiency 仍过于激进；不能用 case ID、gold 或 route 标签修复。

## Answer

| 指标 | fixed | adaptive |
| --- | ---: | ---: |
| requirement coverage | 0.4375 | 0.3125 |
| citation correctness | 0.2326 | 0.1687 |
| citation completeness | 0.4375 | 0.3125 |
| major fact support rate | 0.2326 | 0.1687 |
| unsupported major claims | 1.7917 | 2.1667 |
| 平均 rounds | 1.0000 | 1.7917 |
| 平均 tool calls | 1.0000 | 1.7917 |

- adaptive-eligible：只改善 1 条（`answer-m412-07`），退化 4 条（`01/04/09/12`）；未达至少改善 5、退化最多 2。
- fixed-eligible 误触发率=`0.75`（6/8）。
- 质量非退化指标全部失败；平均 rounds 大于 `1.5`。
- 安全预算合格：termination=`1.0`、每题 rounds≤2、tool calls≤4、重复 query+scope=`0`；延迟仅记录，adaptive p95=`219214.6212ms`。

## 评分与盲审

主分严格分开确定性引用有效性、语义 quote 支持和 gold 覆盖审计。结构化 grader 的布尔/理由冲突以 `grader_inconsistent` 原样记录，未自动修复或改分。盲审清单为固定 hash 抽样的 5 条：`answer-m412-15/09/22/03/21`，载于 answer report；本次自动标记的 false positive、false negative、inconsistent 均为 0。盲审仅报告，不改自动主分。

## 坏例与后续边界

- `answer-m412-03` 能检到两项相关 Scaling Laws evidence，但 adaptive 最终引用未完全映射到冻结 requirement。
- `answer-m412-09` 受长论文目录/背景 passage 干扰，不能把检索到的文本转为可支持 claim。
- 不得根据这些失败项改写本冻结集或重跑正式结果。若用户要继续，必须单独批准新 Goal 和全新数据冻结。
