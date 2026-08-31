# M6D README 与只读评测展示验收

状态：正式验收通过（2026-08-31）。页面只读展示冻结的 M6C 正式结果，M7 已可开始但本轮不实施 M7。

## 交付

- README 增加 M6 KITE 正式结果表、协议分数边界和默认值不自动切换说明。
- 新增 `/evaluation` 静态 Server Component，复用现有 editorial layout 和 warm-paper design tokens，展示 snapshot、模型、四条 Pipeline、p95/context、逐题 win/tie/loss、坏例和 M3/M4 历史结论。
- 页面数据是冻结 summary 的清洗副本，未接入 API、index、模型、参数编辑或 Pipeline 切换；不展示 prompt 全文、密钥、本机路径或 KITE PDF。
- 评测页加入主导航，Library、Search、Paper、Chat 路径保持原样。

## 验收

```text
uv run --extra dev python -m pytest tests/test_project_status.py tests/test_kite_eval.py tests/test_retrieval_pipeline.py tests/test_agent_grounded_answer.py -q
45 passed, 3 warnings

uv run --extra dev ruff check evals/kite_runner.py evals/kite.py evals/v2_runner.py indexing tests/test_project_status.py tests/test_kite_eval.py tests/test_retrieval_pipeline.py tests/test_agent_grounded_answer.py
All checks passed!

npm --prefix web run test:contracts
通过

npm --prefix web run lint
通过

npm --prefix web run build
通过（Next.js 16.2.0，`/evaluation` 生成静态路由）

本轮只更新正式 summary 的清洗数据和状态文案，未改变评测页布局、交互或主路径。
```

页面数值来自 `artifacts/evals/kite/summary.json` 和 `docs/kite_benchmark_report.md`，四条 Pipeline 的有效题数均为 15/15，运行均为 `formal_run=true`。页面不触发模型或索引操作，不公开 PDF、LFS object、派生全文、索引、本地路径或密钥；清洗后的 manifest、summary 和四份 report 可复核。M6D 已完成正式验收，M7 已获单独授权但尚未实施。
