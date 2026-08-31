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

B1 全量 15/15 有效，KITE-protocol 平均分 `5.7333`，p50 `134311.8399 ms`，p95 `179770.5466 ms`，平均 context `16863.2667` tokens，平均 evidence `3.7333`，judge retry 0。

报告：`artifacts/evals/kite/b1/report.json`，SHA-256 `becdd4acbf0b816a29d5ba7b639ebd464db1adedf71c751411f9901837276608`。

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
15/15 valid, mean_score=5.7333
```

报告明确标记 `formal_run=true`、`dirty=false`，并绑定 manifest、parser artifact、index manifest、配置和 clean commit。证据输出不包含 `retrieval_text`，没有向产品默认值写入任何变更。真实 embedding、generation 和 judge 调用均成功，judge 错误和重试为 0。
