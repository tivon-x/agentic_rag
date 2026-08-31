# Production Pipeline Decision 诊断记录

状态：非正式诊断记录。B0 至 B3 报告均为 `formal_run=false`，工作区 dirty 且代码 patch hash 不一致；本文件不构成正式 M6C 生产决策或 M6 已交付证明。

- Benchmark: `kite-ai-papers`, upstream commit `85e71ad63db9ea410eccbb0158f94e7d72462b99`.
- Query SHA-256: `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3`; corpus: `134` PDFs; corpus manifest SHA-256: `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8`.
- KITE-protocol diagnostic scores: B0 `3.8667`, B1 `6.1333`, B2 `6.8667`, B3 `6.5333`.
- Pairwise: B2 `5/8/2` wins/ties/losses; B3 `7/5/3`.
- Latency/context: B1 p95 `166207.5666 ms`, `16599.6 tokens`; B2 p95 `170017.9374 ms`, `16935.5333 tokens`; B3 p95 `177816.5361 ms`, `17810.4 tokens`.
- Existing internal diagnostic result: M3.2 kept fixed B1 as the frozen default; M4.1 adaptive rechecks did not prove net benefit.
- Known failure modes: B2 regresses on `ai-papers-003` and `ai-papers-007`; B3 additionally regresses on `ai-papers-010`; B0 is below baseline on 11 cases.
- Evidence audit: all 15 cases per pipeline have valid scores; evidence was normalized from retrieval-owned parser records and excludes `retrieval_text`.
- Default: `b1` (`v1_flat_rerank`).
- Automatic switch: `False`.
- Promotion candidates: none approved; b2 is a non-formal diagnostic candidate only.
- Decision: no production switch is approved; keep B1 active。M6C 尚未通过正式验收。
- Reason: this non-formal KITE run is evidence for a future decision, not a production change.
