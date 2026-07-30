# M3.2 策略收口验收

- Strategy candidate passed: `False`
- Default fixed pipeline: `v1_flat_rerank`
- M4 fixed baseline: `v1_flat_rerank`
- M3.1 core passed: `false`（历史失败结论保持不变）
- M3 strategy closed: `True`
- M4 entry ready: `True`
- Formal holdout runs: `1`
- Metadata prefix leaks: `0`
- Active index changed: `False`

## 冻结 gate

### holdout

- Passed: `False`
- W/T/L: `18/23/7`
- Checks: `{"recall_at_10_not_lower": true, "mrr_at_10_not_lower": true, "ndcg_at_10_not_lower": true, "wins_at_least_10": true, "losses_at_most_8": true, "each_subset_declines_at_most_1": true, "p95_latency_not_higher": true, "context_passage_recall_not_lower": false}`
- Answer smoke: `{"candidate_metadata_prefix_leaks_zero": true, "baseline_metadata_prefix_leaks_zero": true, "context_packing_not_lower": true, "candidate_citations_and_pages_present": true, "baseline_citations_and_pages_present": true}`

### old_dev

- Passed: `True`
- W/T/L: `24/16/8`
- Checks: `{"recall_at_10_not_lower": true, "mrr_at_10_not_lower": true, "ndcg_at_10_not_lower": true, "wins_at_least_10": true, "losses_at_most_8": true, "each_subset_declines_at_most_1": true, "p95_latency_not_higher": true, "context_passage_recall_not_lower": true}`
- Answer smoke: `{"candidate_metadata_prefix_leaks_zero": true, "baseline_metadata_prefix_leaks_zero": true, "context_packing_not_lower": true, "candidate_citations_and_pages_present": true, "baseline_citations_and_pages_present": true}`

## 决策

S1 failed at least one frozen M3.2 gate; B1 remains the fixed baseline.

M4 只可使用冻结 baseline contract；本里程碑不实现 M4。

## 指标与人工核查

| Dataset | Pipeline | Recall@10 | MRR@10 | nDCG@10 | Context Recall | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| holdout | B1 | 0.875000 | 0.680258 | 0.710130 | 0.875000 | 389.0657 |
| holdout | S1 | 0.885417 | 0.766667 | 0.776056 | 0.854167 | 231.1416 |
| old dev | B1 | 0.614583 | 0.356622 | 0.405581 | 0.583333 | 424.5483 |
| old dev | S1 | 0.729167 | 0.435499 | 0.483281 | 0.697917 | 228.8369 |

- 已核查逐题表中的 win、tie、loss，及表格、缩写、跨章节、跨论文案例。
- 已抽查 `term-01` 至 `term-05` 的 S1 trace：统一配置均为 `use_rerank=false`，最终候选没有 reranker score；trace 中保留的 `rerank` 字段是兼容的最终排序载体，不表示调用 reranker。
- answer preview 的引用、页码、context packing 均通过冻结 smoke gate；metadata prefix leak 为 `0`。
- holdout 只运行 `1` 次；没有在结果后修改候选、gold、阈值或 active index。
