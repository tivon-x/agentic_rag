# Eval Guide

## Overview

Milestone 7 adds a minimal, repeatable evaluation loop for:

- `routing`
- `retrieval`
- `answer`

The evaluation corpus lives in `evals/`, and the datasets live in `evals/datasets/`.

## Dataset Layout

```text
evals/
├── 01-Attention Is All You Need.pdf
├── 02-The Annotated Transformer.md
├── ...
└── datasets/
    ├── routing_cases.jsonl
    ├── retrieval_cases.jsonl
    └── answer_cases.jsonl
```

Each JSONL row includes at least:

- `question`
- `expected_route`
- `gold_doc_ids`
- `gold_node_ids`
- `reference_answer`
- `difficulty`
- `notes`

## Running Evaluations

Use `uv run` in this repository:

```bash
uv run python main.py eval --suite retrieval --offline
uv run python main.py eval --suite answer --output-format markdown
uv run python main.py eval --suite all --variant hierarchical
```

## Experiment Variants

The runner compares three configurations:

1. `baseline_flat`
2. `flat_rerank`
3. `hierarchical`

Indexes are cached under `data/eval_reports/indexes/` unless `--force-reindex` is set.

## Metrics

### Routing

- `route_accuracy`

### Retrieval

- `recall_at_k`
- `mrr`
- `ndcg`
- `redundancy_rate`

### Answer

- `groundedness`
- `citation_precision`
- `answer_completeness`
- `hallucination_rate_rule`
- `hallucination_rate_llm_judge` when LLM config is available

Answer reports also include:

- `answer_mode`
- `evaluation_mode`

`evaluation_mode=generative_grounded` means the answer came from the normal grounded generation path and is suitable for cross-variant comparison.

`evaluation_mode=offline_extractive_fallback` means the answer was produced by a deterministic extractive fallback. These scores are still useful for smoke testing, but they should not be interpreted as equivalent to full generative answer quality.

## Offline vs Online

- `--offline` forces FakeEmbeddings and disables LLM judge metrics.
- Without `--offline`, the runner uses your configured embeddings and LLMs when available.
- If LLM config is missing, routing falls back to a heuristic router and answer evaluation falls back to offline extractive synthesis.
- Markdown and JSON reports mark offline answer results as `fallback-only` in the leaderboard.

## Report Output

Reports are written to `data/eval_reports/` by default:

- `eval_report_<suite>_<timestamp>.json`
- `eval_report_<suite>_<timestamp>.md`

The JSON report contains per-case rows and aggregate metrics. The Markdown report provides a concise comparison view for quick inspection.
