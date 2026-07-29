# M3.1 验收

- Core passed: `False`
- Default pipeline: `v1_flat_rerank`
- M4 entry ready: `False`
- Dev candidates: `24`
- Formal holdout runs: `0`
- Metadata prefix leaks: `0`
- Active index changed: `False`
- Parser artifact SHA-256: `98e8adf680c578c21d2fffe5b97f3f85d24b768b827fe81aa8ddfc280af242d9`
- Old dev SHA-256: `e1da7d23d352cd17a1601f56280a5c9820ff81002a36dc5ad786cb3a8f90c936`
- New holdout SHA-256: `47e2a70de438468150e22ca07c5f57aaf8630d601b3b645ffcf7a2d3f0dfea78`

## 停止原因

No candidate passed the frozen dev promotion gate.

## Dev promotion gate

- Passed candidates: `0`
- Pareto frontier: `r1_01_quote_mixed_minmax, r1_06_best_boost_off, r3_07_title_section_quote, r3_08_retrieval_input, r3_09_dense_1_25_sparse_0_75`
- Diagnostic candidate: `r3_07_title_section_quote`（仅用于失败分析，不是 finalist）
- Latency protocol: `{"warmup_count": 1, "repeat_count": 5, "random_seed": 31}`

| Candidate | Recall@10 | MRR@10 | nDCG@10 | W/T/L | p95/B1 | Failed checks |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| r1_01_quote_mixed_minmax | 0.729167 | 0.435499 | 0.483281 | 24/16/8 | 0.680618 | losses_at_most_3 |
| r1_02_quote_mixed_rrf | 0.708333 | 0.36727 | 0.434439 | 18/19/11 | 0.698072 | losses_at_most_3 |
| r1_03_section_mixed_rrf | 0.552083 | 0.277687 | 0.325352 | 12/16/20 | 0.684947 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1 |
| r1_04_title_section_mixed_rrf | 0.53125 | 0.25687 | 0.310348 | 11/15/22 | 0.700644 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1 |
| r1_05_full_metadata_mixed_rrf | 0.572917 | 0.297999 | 0.34969 | 13/16/19 | 0.736655 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1 |
| r1_06_best_boost_off | 0.729167 | 0.434135 | 0.480764 | 22/18/8 | 0.654621 | losses_at_most_3 |
| r2_1_1 | 0.625 | 0.368279 | 0.402468 | 16/19/13 | 1.013157 | recall_delta_at_least_0_02, losses_at_most_3, ndcg_not_lower |
| r2_1_2 | 0.645833 | 0.385119 | 0.42802 | 17/18/13 | 9.705411 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r2_1_3 | 0.53125 | 0.330837 | 0.352631 | 15/17/16 | 19.651428 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r2_1_4 | 0.25 | 0.070279 | 0.106267 | 8/11/29 | 18.590312 | recall_delta_at_least_0_02, wins_at_least_10, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r2_2_1 | 0.625 | 0.355928 | 0.396522 | 16/19/13 | 1.127032 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower |
| r2_2_2 | 0.645833 | 0.381448 | 0.425052 | 17/18/13 | 12.255411 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r2_2_3 | 0.552083 | 0.32705 | 0.354762 | 13/17/18 | 21.33433 | recall_delta_at_least_0_02, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r2_2_4 | 0.302083 | 0.069056 | 0.119667 | 6/12/30 | 22.463458 | recall_delta_at_least_0_02, wins_at_least_10, losses_at_most_3, mrr_not_lower, ndcg_not_lower, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r3_1_1 | 0.71875 | 0.421106 | 0.467835 | 25/15/8 | 11.31798 | losses_at_most_3, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r3_1_2 | 0.71875 | 0.385417 | 0.451522 | 19/20/9 | 11.124291 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_1_3 | 0.697917 | 0.396032 | 0.453939 | 18/19/11 | 10.34724 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_2_1 | 0.71875 | 0.426339 | 0.471407 | 23/16/9 | 10.449308 | losses_at_most_3, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r3_2_2 | 0.71875 | 0.415708 | 0.470047 | 20/19/9 | 11.083417 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_2_3 | 0.697917 | 0.404712 | 0.460052 | 19/18/11 | 9.347191 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_07_title_section_quote | 0.739583 | 0.477885 | 0.51184 | 27/14/7 | 9.677832 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_08_retrieval_input | 0.739583 | 0.468915 | 0.509842 | 25/16/7 | 9.591266 | losses_at_most_3, p95_ratio_at_most_1_35 |
| r3_09_dense_1_25_sparse_0_75 | 0.71875 | 0.421106 | 0.467835 | 25/15/8 | 9.447723 | losses_at_most_3, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |
| r3_10_dense_0_75_sparse_1_25 | 0.71875 | 0.421106 | 0.467835 | 25/15/8 | 10.218088 | losses_at_most_3, each_subset_declines_at_most_1, p95_ratio_at_most_1_35 |

## 可复现性与安全检查

- Config SHA-256: `eb5e823510def8e64eb91896c3d93bca54f88aee9b9b087b511802d9f5389557`
- Code commit: `199492dd5adcdeb0e8abce76509e4a4001fbdd0d`
- Working-tree patch SHA-256: `55738f75819bc366816f5d1a4c652bf023ac738d244a439954b953fab5b22bad`
- Dev answer preview metadata prefix leaks: `0`
- Active index before: `{"active_json": null, "sqlite_app_state_exists": false, "sqlite_active_index_version": null}`
- Active index after: `{"active_json": null, "sqlite_app_state_exists": false, "sqlite_active_index_version": null}`
- Holdout quality evaluation was not run; formal holdout run count is `0`.

## 决策

候选选择严格按预先冻结的字典序规则执行，不使用综合分。

## 最终验证

- Parser quality gate：通过，16 篇、48 个重点页，10 个 gate 全部通过；
  p95 latency ratio 为 `7.0623`。
- 后端完整测试：`249 passed`。
- Ruff：通过。
- 前端 lint：通过。
- Next.js production build：通过。
- 未运行 final，未访问 holdout 质量结果，未修改 active index，未执行 M4。
