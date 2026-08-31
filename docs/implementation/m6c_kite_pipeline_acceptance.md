# M6C KITE 固定 Pipeline 比较与生产决策验收

状态：正式验收通过（2026-08-31）。B0、B1、B2、B3 已在同一 clean evaluation commit 和冻结 KITE snapshot 上完成正式运行，并完成逐题比较、证据审计和生产决策冻结。

## 已完成的正式尝试

| Pipeline | Mean KITE-protocol score | Valid | p95 latency | Mean context tokens | formal |
|---|---:|---:|---:|---:|---|
| B0 | 4.2667 | 15/15 | 173270.2967 ms | 14450.6 | true |
| B1 | 6 | 15/15 | 207923.746 ms | 15810.1333 | true |
| B2 | 6.4 | 15/15 | 187091.5157 ms | 18256 | true |
| B3 | 6.3333 | 15/15 | 176588.074 ms | 17182.7333 | true |

B0 至 B3 使用相同 KITE snapshot、parser artifact、embedding/generation/judge 配置和 clean evaluation commit `787d75580e3950f15981c2b138dc9942dff75014`；四份报告均为 `formal_run=true`、15/15 有效、judge 错误和重试为 0。parser artifact SHA-256 为 `c6477f7f2044140739d1786ab06aa72295ef47e7fb991ce51c4c10fec0c4c7bc`，manifest SHA-256 为 `716539e41b03e94b6a98546e20ff7283a7ee4a421ae1d57a515df2b4c40ca415`。

正式报告：B0 `artifacts/evals/kite/b0/report.json`（SHA-256 `03731c5f8726208c1cca28f7fbe29eb1d73e8fba14b00cc734b526f9a1d6435f`）；B1 `artifacts/evals/kite/b1/report.json`（SHA-256 `01a1e50b47b0619097df4b89d9a0d564cb7ac42ec335c11822ac801cc72091cd`）；B2 `artifacts/evals/kite/b2/report.json`（SHA-256 `51df9abba3d912f50de4372d211c8b699d3e736e0c43b08f4c33316fb655a855`）；B3 `artifacts/evals/kite/b3/report.json`（SHA-256 `5378718b85aa4d1824c88718f45ce29cd0768ffe3bfe1d7f18b30bff2f8b1f26`）。

四份报告的 index manifest SHA-256 分别为：B0 `aeee67ed00e3e415d9626256c874eed66ec0d2e7ec4f631d70d07d2041297467`、B1 `15a01e9f92aaa1f8b2c80e0ff4aa49eb3d9b44bbefeff38a4c9f4da3e627e860`、B2 `f247ac8be0315909bdb6925a133f599b3231f8edc59cd465fde62cb4222f3253`、B3 `e89018e9d09973474aac857f598f6c7851762dfff9a2148ed8c224089a5b34a6`。

## 非正式诊断归档

先前脏工作区产生的 B0-B3 报告和 summary 已移至 `artifacts/evals/kite/nonformal/`，保留用于历史对照，不与本轮正式结果混用，也不作为 M6C 生产决策证据。

## 决策

四条 Pipeline 的逐题结果已写入 `artifacts/evals/kite/summary.json`，summary SHA-256 为 `128a948580ce85f1b497bd692ea7ee3d20eacdd5acb64dae953921cb9e135ae`。相对 B1：B2 为 5 胜 / 7 平 / 3 负，平均分只提升 `0.4`；B3 为 5 胜 / 6 平 / 4 负，平均分只提升 `0.3333`。两者都未同时满足分数提升 `≥0.5` 和 loss `≤2`，因此 promotion candidates 为空。

生产默认继续 `b1 / v1_flat_rerank`，不自动切换；M3.2 的固定 B1 结论和 M4.1 Adaptive 未证明净收益的内部诊断保持有效。M6C 已冻结，M6D 已完成，M7 已获授权且可以开始。
