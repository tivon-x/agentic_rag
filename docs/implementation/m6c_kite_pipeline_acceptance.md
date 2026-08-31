# M6C KITE 固定 Pipeline 比较与生产决策验收

状态：未通过正式验收（2026-08-31）。B2、B3 在建索引的 embedding dimension probe 均收到供应商 `403 AccessDenied.Unpurchased`，因此没有生成正式报告，也不能据此冻结 M6C 或推进 M6D。

## 已完成的正式尝试

| Pipeline | Mean KITE-protocol score | Valid | p95 latency | Mean context tokens | formal |
|---|---:|---:|---:|---:|---|
| B0 | 3.8 | 15/15 | 230453.0793 ms | 16068.8 | true |
| B1 | 5.7333 | 15/15 | 179770.5466 ms | 16863.2667 | true |
| B2 | — | — | — | — | blocked: embedding 403 |
| B3 | — | — | — | — | blocked: embedding 403 |

B0 和 B1 使用相同 KITE snapshot、parser artifact、embedding/generation/judge 配置和 clean commit `787d75580e3950f15981c2b138dc9942dff75014`；两份报告均为 `formal_run=true`、15/15 有效、judge 错误和重试为 0。

正式报告：B0 `artifacts/evals/kite/b0/report.json`（SHA-256 `f04c0ad36a64ca751ce7acbb40c99dbfa022afe0a52525d4bfbcdcd85279ae69`）；B1 `artifacts/evals/kite/b1/report.json`（SHA-256 `becdd4acbf0b816a29d5ba7b639ebd464db1adedf71c751411f9901837276608`）。

## 非正式诊断归档

先前脏工作区产生的 B0-B3 报告和 summary 已移至 `artifacts/evals/kite/nonformal/`，保留用于历史对照，不与本轮正式结果混用，也不作为 M6C 生产决策证据。

## 决策

M6C 暂不验收，当前不能生成可审计的四 Pipeline summary 或 promotion candidate。生产默认继续 `b1 / v1_flat_rerank`，不自动切换；M3.2 的固定 B1 结论和 M4.1 Adaptive 未证明净收益的内部诊断保持有效。恢复 `qwen3.7-text-embedding` 访问后，应从同一 clean commit 重新运行 B2、B3，再聚合并冻结 M6C。
