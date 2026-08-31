# Production Pipeline Decision

- Benchmark: `kite-ai-papers`, upstream commit `85e71ad63db9ea410eccbb0158f94e7d72462b99`.
- Query SHA-256: `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3`; corpus: `134` PDFs; corpus manifest SHA-256: `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8`.
- KITE-protocol scores: B0 `4.2667`, B1 `6`, B2 `6.4`, B3 `6.3333`.
- Pairwise wins/ties/losses vs B1: B0 `2/6/7`; B2 `5/7/3`; B3 `5/6/4`.
- Latency/context: B1 p95 `207923.746 ms`, `15810.1333 tokens`; B2 p95 `187091.5157 ms`, `18256.0 tokens`; B3 p95 `176588.074 ms`, `17182.7333 tokens`.
- Existing internal diagnostic result: M3.2 kept fixed B1 as the frozen default; M4.1 adaptive rechecks did not prove net benefit.
- Known case regressions: B0 losses: ai-papers-003, ai-papers-005, ai-papers-006, ai-papers-008, ai-papers-009, ai-papers-014, ai-papers-015; B2 losses: ai-papers-003, ai-papers-006, ai-papers-010; B3 losses: ai-papers-003, ai-papers-006, ai-papers-010, ai-papers-015.
- Evidence audit: all 15 cases per pipeline have valid scores; evidence was normalized from retrieval-owned parser records and excludes `retrieval_text`.
- Default: `b1` (`v1_flat_rerank`).
- Automatic switch: `False`.
- Promotion candidates: none.
- Decision: keep B1 active; any candidate (none) requires separate production approval.
- Reason: KITE is evidence for a decision; it does not mutate the product default.
