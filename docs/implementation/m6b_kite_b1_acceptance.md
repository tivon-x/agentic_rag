# M6B KITE B1 公开 E2E 基线验收

状态：未通过正式验收（2026-08-30）。本记录保留为非正式诊断结果，不能作为 M6B 正式基线。

## 冻结运行条件

- Pipeline：registry 中的 `b1`，产品名 `v1_flat_rerank`，未改生产默认。
- Embedding：`openai / qwen3.7-text-embedding / 1024`，batch 20，`raw`，最大输入 6000 字符。
- Generation：`qwen3.7-plus`，fixed answer path。
- Judge：task type `kite_judge`，`qwen3.7-plus`，`kite-official-compatible-v1`，temperature 0。
- Judge 只接受单个 `0` 至 `10` 整数；非法输出、异常和超时最多重试一次，最终失败保存 null 和错误。

## 运行结果

Smoke 固定使用 `ai-papers-001/005/010/015`，4/4 有效，平均分 7.0。

B1 全量 15/15 有效，KITE-protocol 平均分 `6.1333`，p50 `130716.1334 ms`，p95 `166207.5666 ms`，平均 context `16599.6` tokens，平均 evidence `3.8667`，judge retry 0。

报告：`artifacts/evals/kite/b1/report.json`，SHA-256 `b567e82293fc1deaa27e55eb113cc4dbc27a7a6e5f4f856f38a3301c52c3830b`。

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
15/15 valid, mean_score=6.1333
```

本次工作区包含未提交改动，报告明确标记 `formal_run=false` 并保存 working-tree patch hash；结果只可用于本轮诊断，不能冒充 clean checkout 的正式可复现基线。证据输出不包含 `retrieval_text`，没有向产品默认值写入任何变更。M6B 尚未完成正式验收。
