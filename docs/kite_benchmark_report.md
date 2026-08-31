# KITE AI Papers Benchmark

Frozen upstream repository: `https://github.com/D-Star-AI/KITE`.

Frozen upstream commit: `85e71ad63db9ea410eccbb0158f94e7d72462b99`.

Frozen query SHA-256: `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3`; corpus: `134` PDFs; corpus manifest SHA-256: `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8`.

Generation model: `qwen3.7-plus`; judge model: `qwen3.7-plus`; prompt: `kite-official-compatible-v1`.

| Pipeline | Mean score | Valid cases | p95 latency (ms) | Mean context tokens |
|---|---:|---:|---:|---:|
| b0 | 4.2667 | 15/15 | 173270.2967 | 14450.6 |
| b1 | 6 | 15/15 | 207923.746 | 15810.1333 |
| b2 | 6.4 | 15/15 | 187091.5157 | 18256.0 |
| b3 | 6.3333 | 15/15 | 176588.074 | 17182.7333 |

Scores are judge outputs on the frozen KITE protocol; per-case evidence and diagnostics remain in the JSON reports.

## Pairwise results vs B1

- b0: 2 wins, 6 ties, 7 losses; wins=`ai-papers-002, ai-papers-013`, losses=`ai-papers-003, ai-papers-005, ai-papers-006, ai-papers-008, ai-papers-009, ai-papers-014, ai-papers-015`
- b2: 5 wins, 7 ties, 3 losses; wins=`ai-papers-002, ai-papers-008, ai-papers-009, ai-papers-011, ai-papers-013`, losses=`ai-papers-003, ai-papers-006, ai-papers-010`
- b3: 5 wins, 6 ties, 4 losses; wins=`ai-papers-002, ai-papers-005, ai-papers-009, ai-papers-011, ai-papers-013`, losses=`ai-papers-003, ai-papers-006, ai-papers-010, ai-papers-015`

## Evidence audit

Every reported case has an integer score and every public evidence item was canonicalized from the parser artifact by passage ID. Reports contain source, paper, section, page and source-faithful quote fields; `retrieval_text` is not public.

## Decision gate

Promotion candidates before a separate production approval: none.

Known candidate regressions remain visible in the pairwise case lists. This score uses the KITE protocol with the configured local models and is not comparable to the upstream absolute score.
