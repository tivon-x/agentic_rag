# M6D README 与只读评测展示验收

状态：未通过正式验收（2026-08-31）。本记录保留为非正式诊断展示，不能作为 M6D 已交付的证明。

## 交付

- README 增加 M6 KITE 非正式诊断结果表、协议分数边界和默认值不自动切换说明。
- 新增 `/evaluation` 静态 Server Component，复用现有 editorial layout 和 warm-paper design tokens，展示 snapshot、模型、四条 Pipeline、p95/context、逐题 win/tie/loss、坏例和 M3/M4 历史结论。
- 页面数据是冻结 summary 的清洗副本，未接入 API、index、模型、参数编辑或 Pipeline 切换；不展示 prompt 全文、密钥、本机路径或 KITE PDF。
- 评测页加入主导航，Library、Search、Paper、Chat 路径保持原样。

## 验收

```text
uv run --extra dev python -m pytest -q
313 passed, 3 warnings

uv run --extra dev ruff check .
All checks passed!

npm --prefix web run test:contracts
通过

npm --prefix web run lint
通过

npm --prefix web run build
通过（Next.js 16.2.0，`/evaluation` 生成静态路由）

通过：生产构建在桌面与 375px 视口打开 `/evaluation`；移动端表格横向滚动提示可见，B2 逐题明细可展开。
```

页面数值来自 `artifacts/evals/kite/summary.json` 和 `docs/kite_benchmark_report.md`，四条 Pipeline 的有效题数均为 15/15；这些运行均为 `formal_run=false` 的非正式诊断。清洗后的 manifest、summary 和四份 report 可复核，PDF、LFS object、派生全文和索引仍不提交。M6D 尚未完成正式验收。
