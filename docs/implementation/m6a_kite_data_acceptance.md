# M6A KITE 数据与可复现准备验收

状态：已完成（2026-08-30）。

## 范围

- 固定 KITE 仓库 `https://github.com/D-Star-AI/KITE`，commit `85e71ad63db9ea410eccbb0158f94e7d72462b99`。
- 读取 `queries/ai_papers.json`，固定 SHA-256 `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3`。
- 校验 15 个 case、6 个空 rubric 和 `knowledge-base-content/ai-papers` 下 134 个真实 PDF。
- manifest 只保存逻辑路径、来源、hash、文件大小和 parser 版本；PDF、LFS object、派生全文和索引留在仓库外或 ignored artifacts。

## 实现

- `evals/kite.py`：固定 case adapter、严格字段/hash 校验、PDF/LFS/空文件校验、排序 corpus manifest 和 parser artifact。
- `evals/v2_corpus.py`：复用现有 parser artifact 逻辑，增加递归语料支持，不改变原有非递归行为。
- `evals/configs/kite_b1.yaml`：冻结 parser、embedding、reranker、generation 和 judge 配置。
- `artifacts/evals/kite/manifest.json`：manifest SHA-256 `716539e41b03e94b6a98546e20ff7283a7ee4a421ae1d57a515df2b4c40ca415`，corpus manifest SHA-256 `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8`。
- parser artifact SHA-256 `c6477f7f2044140739d1786ab06aa72295ef47e7fb991ce51c4c10fec0c4c7bc`，仅作为本机运行缓存，不提交完整内容。

## 验收

```text
uv run --extra dev python -m pytest tests/test_kite_eval.py -q
14 passed, 3 warnings

uv run ruff check evals/kite.py evals/kite_runner.py tests/test_kite_eval.py
All checks passed!

uv run python -m evals.kite_runner prepare --config evals/configs/kite_b1.yaml
case_count=15, corpus_file_count=134

git diff --check
passed
```

没有调用 embedding、generation 或 judge 服务；没有修改生产 Pipeline、active index 或数据库。
