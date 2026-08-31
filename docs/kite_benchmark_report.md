# KITE AI Papers Benchmark 诊断报告

状态：非正式诊断。B0 至 B3 报告均为 `formal_run=false`，工作区 dirty 且代码 patch hash 不一致；本文件不构成正式冻结基线或 M6 交付证明。

Upstream repository snapshot: `https://github.com/D-Star-AI/KITE`.

Snapshot commit: `85e71ad63db9ea410eccbb0158f94e7d72462b99`.

Query SHA-256: `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3`; corpus: `134` PDFs; corpus manifest SHA-256: `f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8`.

Generation model: `qwen3.7-plus`; judge model: `qwen3.7-plus`; prompt: `kite-official-compatible-v1`.

| Pipeline | Mean score | Valid cases | p95 latency (ms) | Mean context tokens |
|---|---:|---:|---:|---:|
| b0 | 3.8667 | 15/15 | 402185.6011 | 16474.2667 |
| b1 | 6.1333 | 15/15 | 166207.5666 | 16599.6 |
| b2 | 6.8667 | 15/15 | 170017.9374 | 16935.5333 |
| b3 | 6.5333 | 15/15 | 177816.5361 | 17810.4 |

Scores are judge outputs on the KITE protocol for this diagnostic run; per-case evidence and diagnostics remain in the JSON reports. They are not formal frozen results.

## Pairwise results vs B1

- b0: 1 wins, 3 ties, 11 losses; wins=`ai-papers-004`, losses=`ai-papers-003, ai-papers-005, ai-papers-006, ai-papers-007, ai-papers-008, ai-papers-009, ai-papers-010, ai-papers-011, ai-papers-012, ai-papers-014, ai-papers-015`
- b2: 5 wins, 8 ties, 2 losses; wins=`ai-papers-004, ai-papers-008, ai-papers-009, ai-papers-011, ai-papers-012`, losses=`ai-papers-003, ai-papers-007`
- b3: 7 wins, 5 ties, 3 losses; wins=`ai-papers-002, ai-papers-004, ai-papers-009, ai-papers-011, ai-papers-012, ai-papers-013, ai-papers-014`, losses=`ai-papers-003, ai-papers-007, ai-papers-010`

## Evidence audit

Every reported case has an integer score and every public evidence item was canonicalized from the parser artifact by passage ID. Reports contain source, paper, section, page and source-faithful quote fields; `retrieval_text` is not public.

## Diagnostic gate

No formal promotion candidate was approved. `b2` is retained as a non-formal diagnostic candidate only; production remains B1.

Known candidate regressions remain visible in the pairwise case lists. This score uses the KITE protocol with the configured local models and is not comparable to the upstream absolute score. M6B–M6D remain pending formal acceptance（正式验收）。
