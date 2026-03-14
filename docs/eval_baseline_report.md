# Eval Baseline Report

## Run Context

- Date: 2026-03-14
- Command: `uv run python main.py eval --suite all --offline --output-format both`
- Corpus: `evals/`
- Dataset: `evals/datasets/`
- Embeddings: `FakeEmbeddings`
- LLM answer generation: disabled

Generated artifacts:

- `data/eval_reports/eval_report_all_20260314_233920.json`
- `data/eval_reports/eval_report_all_20260314_233920.md`

## Variant Summary

### baseline_flat

- Routing: `route_accuracy=0.5`
- Retrieval: `recall_at_k=1.0`, `mrr=0.85`, `ndcg=0.8681`
- Answer: `groundedness=1.0`, `citation_precision=0.2833`, `answer_completeness=0.2842`
- Answer mode: `offline_extractive`

### flat_rerank

- Routing: `route_accuracy=0.5`
- Retrieval: `recall_at_k=1.0`, `mrr=1.0`, `ndcg=1.0`
- Answer: `groundedness=1.0`, `citation_precision=0.4333`, `answer_completeness=0.1882`
- Answer mode: `offline_extractive`

### hierarchical

- Routing: `route_accuracy=0.5`
- Retrieval: `recall_at_k=0.8333`, `mrr=0.8333`, `ndcg=0.8084`
- Answer: `groundedness=1.0`, `citation_precision=0.3`, `answer_completeness=0.1351`
- Answer mode: `offline_extractive`

## Takeaways

1. In this offline run, `flat_rerank` is the strongest retrieval configuration.
2. `hierarchical` underperforms the flat rerank baseline on the current offline corpus and dataset, so it should be treated as the next optimization target.
3. All answer metrics were produced in `offline_extractive_fallback` mode, so they are useful for regression tracking but not for claiming generative answer quality wins.
4. Routing stayed flat at `0.5` because this run did not use LLM-based routing and relied on heuristic fallback mode.

## Interpretation Notes

- Do not compare the answer leaderboard from this run with future online runs that use grounded generation.
- Use this report as the repository's first offline regression baseline.
- For milestone-quality comparison of answer generation, rerun the same suite with LLM routing and grounded aggregation enabled.
