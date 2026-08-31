# M6C KITE 固定 Pipeline 比较与生产决策验收

状态：未通过正式验收（2026-08-31）。本记录保留为非正式诊断结果，不能作为 M6C 正式生产决策；没有重跑或修改历史 holdout。

## 聚合结果

| Pipeline | Mean KITE-protocol score | Valid | p95 latency | Mean context tokens | 相对 B1 |
|---|---:|---:|---:|---:|---|
| B0 | 3.8667 | 15/15 | 402185.6011 ms | 16474.2667 | 1 win / 3 ties / 11 losses |
| B1 | 6.1333 | 15/15 | 166207.5666 ms | 16599.6000 | baseline |
| B2 | 6.8667 | 15/15 | 170017.9374 ms | 16935.5333 | 5 wins / 8 ties / 2 losses |
| B3 | 6.5333 | 15/15 | 177816.5361 ms | 17810.4000 | 7 wins / 5 ties / 3 losses |

所有 case 都取得有效整数 judge score，judge retry 为 0。B2 相对 B1 提升 `0.7334`，满足分数、至少 4 胜、最多 2 负、p95 不超过 1.5 倍、context 不超过 1.5 倍；B3 的 3 个 loss 不满足最多 2 负，且分数提升只有 `0.4000`。B0 明显低于 B1。

## 逐题审计

- B2 wins：`004, 008, 009, 011, 012`；losses：`003, 007`。
- B3 wins：`002, 004, 009, 011, 012, 013, 014`；losses：`003, 007, 010`。
- B0 wins：`004`；losses：`003, 005, 006, 007, 008, 009, 010, 011, 012, 014, 015`。
- 已检查每个候选的全部 win/loss。每题 evidence 通过 passage ID 从 parser artifact 归一化，包含 paper、section、page 和 source-faithful quote；公开 JSON 没有 `retrieval_text`。
- 主要退化：B2 在 `ai-papers-003`（Levels of AGI 六项原则）和 `ai-papers-007`（AlphaCodium 双重验证）低于 B1；B3 还在 `ai-papers-010`（MiniLMv2 层选择）降为 0 分。分数收益没有掩盖这些错误回答。

## 决策

`b2` 仅记录为非正式诊断候选，不构成 promotion candidate 或生产批准；当前默认继续 `b1 / v1_flat_rerank`，runner 不自动切换。M3.2 的固定 B1 结论和 M4.1 Adaptive 未证明净收益的内部诊断保持有效。

summary：`artifacts/evals/kite/summary.json`，SHA-256 `2ca78af3de2e1900e4f6dc88528797a71637bc3b5bbe96c18b8f1c98f8720cb1`。

报告 SHA-256：B0 `34e73910aaab608c81ddd00716a7f76cab17ae1ee4212c4c72f18db26eda2e9d`，B1 `b567e82293fc1deaa27e55eb113cc4dbc27a7a6e5f4f856f38a3301c52c3830b`，B2 `b4e8d78b3e5bd965da329ea56362bb28ee89fdf563154338ca26ce49a6ecf74b`，B3 `c185204ff63957f545a819d2a7cbb8fd10ce62aba6b39834e7475ac0c9d07af2`。

本轮四份报告因工作区有未提交改动而标记 `formal_run=false`，并保留代码 commit 与 patch hash；其中 B1 与其他报告的 patch hash 不一致。因此结果只用于非正式横向诊断，不宣称为 clean checkout 正式基线，M6C 尚未完成正式验收。
