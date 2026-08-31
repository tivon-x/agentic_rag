# M6B KITE B1 公开 E2E 基线验收

状态：正式验收通过（2026-08-31）。本报告由 clean commit `787d75580e3950f15981c2b138dc9942dff75014` 生成。

## 冻结运行条件

- Pipeline：registry 中的 `b1`，产品名 `v1_flat_rerank`，未改生产默认。
- Embedding：`openai / qwen3.7-text-embedding / 1024`，batch 20，`raw`，最大输入 6000 字符。
- Generation：`qwen3.7-plus`，fixed answer path。
- Judge：task type `kite_judge`，`qwen3.7-plus`，`kite-official-compatible-v1`，temperature 0。
- Judge 只接受单个 `0` 至 `10` 整数；非法输出、异常和超时最多重试一次，最终失败保存 null 和错误。

## 运行结果

Smoke 固定使用 `ai-papers-001/005/010/015`，4/4 有效，平均分 `7.75`。

B1 全量 15/15 有效，KITE-protocol 平均分 `6`，p50 `134755.3933 ms`，p95 `207923.746 ms`，平均 context `15810.1333` tokens，平均 evidence `3.8`，judge retry 0。

报告：`artifacts/evals/kite/b1/report.json`，SHA-256 `01a1e50b47b0619097df4b89d9a0d564cb7ac42ec335c11822ac801cc72091cd`。

本轮正式报告统一绑定 parser artifact SHA-256 `c6477f7f2044140739d1786ab06aa72295ef47e7fb991ce51c4c10fec0c4c7bc` 和 KITE manifest SHA-256 `716539e41b03e94b6a98546e20ff7283a7ee4a421ae1d57a515df2b4c40ca415`。

Smoke 报告归档于 `artifacts/evals/kite/nonformal/b1_smoke_report.json`，SHA-256 `4522e62d9f49910e73da01c39ec4ea065e6ee9559ed44deaa5644a17a0fbbdc4`。

报告中的每题记录包括 source index、query、reference、rubric、answer、retrieval-owned evidence、context tokens、score、judge model/prompt、latency 和错误字段。`input_tokens`、`output_tokens`、`llm_calls` 在 provider 没有给出真实值时保持 null，不估算。

## 验收

```text
uv run --extra dev python -m pytest tests/test_kite_eval.py tests/test_retrieval_pipeline.py tests/test_agent_grounded_answer.py -q
通过（KITE contract、fixed pipeline 和 grounded answer 回归）

uv run ruff check evals agent core indexing tests
All checks passed!

uv run python -m evals.kite_runner run --config evals/configs/kite_b1_smoke.yaml
4/4 valid

uv run python -m evals.kite_runner run --config evals/configs/kite_b1.yaml
15/15 valid, mean_score=6
```

报告明确标记 `formal_run=true`、`dirty=false`，并绑定 manifest、parser artifact、index manifest、配置和 clean commit。证据输出不包含 `retrieval_text`，没有向产品默认值写入任何变更。真实 embedding、generation 和 judge 调用均成功，judge 错误和重试为 0。
