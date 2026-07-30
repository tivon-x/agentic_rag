# PROJECT KNOWLEDGE BASE

**Updated:** 2026-07-30
**Branch:** `codex/v2-core`

## OVERVIEW

Python 3.12+ Agentic RAG system for a local-first paper library. The backend uses
FastAPI, SQLite, LangGraph, FAISS, BM25 and an OpenAI-compatible model provider.
The product frontend is Next.js; the older Gradio UI remains available through
`main.py ui`.

The repository has completed M1 through M3.2. The current fixed retrieval baseline
is `v1_flat_rerank`. M4 adaptive behavior, persistent runs and checkpoints are
planned but not implemented.

## CURRENT MILESTONE STATE

- M1: runtime, migrations, persistent indexing jobs and immutable index versions
  completed.
- M2: paper catalog, parser pipeline, page-addressable evidence, Library, Paper and
  Search completed.
- M3/M3.1: fixed retrieval evaluation completed; the original promotion gate did
  not pass.
- M3.2: strategy closure completed. S1 failed the holdout Context Passage Recall
  gate, so B1 remains the fixed baseline.
- Next authorized milestone: M4.1, adaptive quality loop. M4.2 persistent runs can
  start only if M4.1 passes its frozen quality gate.

Frozen M4 baseline:

- pipeline: `v1_flat_rerank`
- config hash:
  `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`
- contract:
  `artifacts/evals/v2_m3_2/m4_fixed_baseline.json`
- acceptance:
  `docs/implementation/m3_2_strategy_acceptance.md`

The M3.2 acceptance run reported 253 pytest tests, Ruff, parser gate and frontend
lint/build passing. Treat that as a milestone snapshot, not a substitute for
running current verification.

## STRUCTURE

```text
agentic_rag/
├── agent/          # LangGraph wiring, node package, edges, prompts, tools, schemas
├── api/            # FastAPI routes, SQLite repositories, migrations, index worker
├── core/           # settings, factory, persistence, corpus profile, RAG answer
├── indexing/       # parser, passages, embeddings, immutable indexes, retrieval
├── evals/          # parser and retrieval datasets, runners, gates and reports
├── llms/           # ChatOpenAI adapter and task-type model router
├── ui/             # legacy Gradio UI
├── web/            # Next.js 16 frontend
├── tests/          # pytest suite
├── artifacts/      # committed evaluation contracts and reports where applicable
├── main.py         # CLI entrypoint
└── pyproject.toml  # Python dependencies and test config
```

Subdirectory instructions:

- `indexing/AGENTS.md` applies inside `indexing/`.
- `web/AGENTS.md` applies inside `web/`.
- Before changing Next.js code, read the relevant installed documentation under
  `web/node_modules/next/dist/docs/`.

## WHERE TO LOOK

| Task | File | Notes |
|---|---|---|
| Graph wiring | `agent/graph.py` | node and edge assembly; currently uses `InMemorySaver` |
| Agent nodes | `agent/nodes/` | one module per node, exported from `agent/nodes/__init__.py` |
| Routing edges | `agent/edges.py` | conditional routing belongs here |
| LangChain research agent | `agent/research_search_agent.py` | `create_agent` and fallback middleware |
| Prompts | `agent/prompts.py` | all system prompts live here |
| Tool definitions | `agent/tools.py` | `ToolFactory.create_tools()` and retrieval tool |
| Structured outputs | `agent/schemas.py` | Pydantic routing and query schemas |
| Graph state | `agent/states.py` | current fixed graph state; M4 state is not implemented |
| LLM router | `llms/llm.py` | cached `get_llm_by_type(task_type)` |
| Settings | `core/settings.py` | frozen `AppSettings` and environment loading |
| Dependency wiring | `core/factory.py` | settings to model, indexer and retriever |
| Database migrations | `api/db/migrations.py` | forward-only migrations with backup |
| Database and jobs | `api/db/database.py` | app state, jobs, idempotency and leases |
| Paper repository | `api/db/papers.py` | papers, versions, sections, passages and metadata |
| Index worker | `api/services/index_worker.py` | single leased worker with recovery |
| Upload API | `api/routers/indexing.py` | safe staging, validation and idempotency |
| Paper API | `api/routers/papers.py` | catalog, metadata correction and PDF delivery |
| Search API | `api/routers/search.py` | page-addressable evidence search |
| Chat API | `api/routers/chat.py` | current fixed chat and final-answer streaming |
| Index versions | `indexing/index_versions.py` | immutable build, activation and rollback |
| Ingestion pipeline | `indexing/indexer.py` | end-to-end indexing orchestration |
| Parser protocol | `indexing/parsers/paper_parser.py` | project-owned parser schema and stable IDs |
| PDF ingestion | `indexing/paper_ingestion.py` | PyMuPDF4LLM, quality gate and legacy fallback |
| Structure normalizer | `indexing/parsers/structure_normalizer.py` | deterministic sections, blocks, tables and formulas |
| Passage materialization | `indexing/passages.py` | quote/retrieval split and embedding limit |
| Fixed pipeline registry | `indexing/retrieval_pipeline.py` | B0/B1/B2/B3/S1 contracts and aliases |
| Sparse retrieval | `indexing/bm25_index.py` | BM25 build and persistence |
| Fusion retrieval | `indexing/retriever.py` | dense, sparse, fusion and rerank execution |
| Vector store | `indexing/vectorstore.py` | FAISS-backed store |
| M3 evaluation | `evals/v2_runner.py` | fixed retrieval evaluation |
| M3.1 experiments | `evals/m3_1_runner.py`, `evals/m3_1_experiments.py` | frozen candidate search |
| M3.2 closure | `evals/m3_2_strategy.py` | strategy gate and baseline selection |
| M4.1 handoff | `docs/implementation/m4_1_adaptive_handoff.md` | next execution contract |
| M4.2 handoff | `docs/implementation/m4_2_durable_run_handoff.md` | persistence work after M4.1 passes |

## COMMANDS

```bash
# Install
uv sync
uv sync --extra dev

# Backend and legacy UI
python main.py api
python main.py ui
python main.py index <path>
python main.py activate-index <version>
python main.py ask "your question"

# Next.js frontend
npm --prefix web run dev
npm --prefix web run lint
npm --prefix web run build

# Backend verification
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check .

# Focused tests
uv run pytest tests/test_retriever.py -v
uv run pytest tests/test_retriever.py::test_bm25_retriever_basic_query -v
uv run pytest -k "bm25" -v

# Current parser and retrieval evaluation entrypoints
uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
uv run python -m evals.runner --config <config-path>
```

No type checker is configured in `pyproject.toml`. Run pyright or mypy only as an
explicit ad hoc check.

## CODE STYLE

### Python

- Minimum Python version is 3.12.
- Use built-in generics: `list[str]`, `dict[str, Any]`, `tuple[int, ...]`.
- Prefer `X | None` over `Optional[X]`.
- Use `from __future__ import annotations` when deferred annotations help.
- Use absolute imports from the project root. Do not add relative imports.
- Annotate public function parameters and return values.
- Ruff is the only configured linter. Do not add Black or isort behavior.

Import order:

```python
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel

from agent.prompts import get_research_search_prompt
from core.settings import AppSettings
```

### Naming

| Construct | Convention | Example |
|---|---|---|
| Functions and variables | `snake_case` | `get_llm_by_type` |
| Classes | `PascalCase` | `RetrievalPipelineConfig` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_TOOL_CALLS` |
| Private helpers | leading underscore | `_build_chat_model` |
| LangGraph state keys | `camelCase` | `routingDecision` |

### Errors and logging

- Catch specific exceptions unless a graph node or middleware must provide a safe
  fallback.
- A fallback must return or log a human-readable reason.
- Do not add empty `except` blocks.
- Use `ValueError` for invalid settings and input contracts unless an existing API
  error type already applies.

## ARCHITECTURAL CONVENTIONS

### Agent graph

- Nodes are plain functions and live under `agent/nodes/`.
- Conditional routing stays in `agent/edges.py`.
- Prompts stay in `agent/prompts.py`; do not inline system prompts elsewhere.
- Structured model outputs use Pydantic schemas from `agent/schemas.py`.
- Access models only through `get_llm_by_type(task_type)`.
- M4.1 must add a separate bounded adaptive path without breaking the current fixed
  graph. Do not add M4.2 persistence while implementing M4.1.

### Settings

- All runtime configuration flows through the frozen `AppSettings`.
- Do not call `os.getenv()` outside `core/settings.py`.
- `load_settings()` runs at startup and the result is passed down.
- The current defaults include:
  - `RETRIEVAL_PIPELINE=v1_flat_rerank`
  - `INDEX_WRITE_MODE=versioned`
  - `PAPER_PARSER=pymupdf4llm`
  - `EMBEDDING_INPUT_MODE=raw`
  - `EMBEDDING_MAX_INPUT_CHARS=6000`

### Fixed retrieval

- `indexing/retrieval_pipeline.py` is the source of truth for fixed pipeline
  definitions and index contracts.
- B1, `v1_flat_rerank`, is the current default and frozen M4 baseline.
- Do not switch the default to B2, B3 or S1 without a new frozen evaluation and
  explicit promotion decision.
- M4 retrieval calls must validate the baseline contract instead of reconstructing
  B1 from remembered defaults.
- `retrieval_text` may contain retrieval metadata. User-visible context and
  citations must use source-faithful `quote_text`.
- Query-time embedding settings must match the active index manifest. Fail on
  incompatibility; do not silently rebuild or downgrade.

### Evaluation integrity

- Frozen test questions, labels, gold evidence, thresholds and graders cannot be
  changed after seeing formal results.
- M3.2 holdout has already been opened once. Do not rerun it or reuse it as M4.1
  final test data.
- Record dataset, config, parser artifact, index manifest and code hashes for formal
  runs.
- Preserve per-question wins, ties, losses, subset regressions, latency and bad
  cases. Do not replace them with one aggregate score.
- External model or embedding calls require authorization in the current session;
  do not inherit approval from an earlier session.

### Index reliability

- API indexing runs only in `INDEX_WRITE_MODE=versioned`.
- Legacy mode is read-only and exists for explicit rollback.
- Versioned mode must not silently read or seed a legacy index without an embedding
  manifest.
- SQLite `app_state` is authoritative for the active index. `active.json` is an
  atomic, startup-reconciled mirror.
- Index jobs are persisted before execution and claimed through SQLite leases.
- Migrations are forward-only and create a recovery backup before modifying an
  existing database.
- Upload paths must remain below `UPLOAD_ROOT` after resolution.

### Paper catalog

- `papers.id` is the SHA-256 of uploaded file bytes. Different bytes remain
  different papers.
- Paper versions, sections and passages use deterministic IDs.
- Metadata correction may rebuild `retrieval_text`; it must not change
  `quote_text`.
- Enforce `EMBEDDING_MAX_INPUT_CHARS` on the complete embedding input.
- PyMuPDF4LLM plus the deterministic normalizer is the default parser.
- `PAPER_PARSER=legacy` is the rollback path.
- Parser failure, fallback and `needs_ocr` are visible product states. Do not promise
  OCR.
- A failed reparse must retain the last successful catalog version.
- Metadata updates, passage-prefix refresh and the reindex job are one transaction.

### Testing

- Tests live in `tests/` and use `test_<module>.py`.
- Reuse fixtures from `tests/conftest.py`.
- Use `FakeEmbeddings` for unit tests. Do not make real API calls.
- Use `tmp_path` for filesystem tests and `monkeypatch` for environment isolation.
- Clear inherited environment variables before setting values in settings tests.
- Every bug fix needs a regression test; every new budget or state transition needs
  boundary tests.

## MILESTONE BOUNDARIES

- `docs/research/v2_upgrade_plan.md` is the implementation source of truth.
- `docs/research/phase2_goal_prompts.md` defines executable Goal prompts.
- M4.1 implements and evaluates only the adaptive quality loop.
- M4.2 starts only when `m4_1_quality_passed=true` and
  `m4_2_entry_ready=true`.
- M4.2 owns run tables, worker, checkpoint, SSE recovery and basic run UI.
- M5 owns full trace and technical debugging UI.
- Complete one Goal, create its acceptance report and stop. Do not continue to the
  next Goal without user authorization.

## ANTI-PATTERNS

- Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
  local indexes, model caches or logs.
- Do not commit `.env`, API keys, prompts containing secrets or full environment
  dumps.
- Do not inline prompts outside `agent/prompts.py`.
- Do not instantiate `ChatOpenAI` directly.
- Do not bypass `VectorStore` to access FAISS internals.
- Do not add a second retrieval pipeline implementation outside the registry.
- Do not tune against a formal holdout after opening it.
- Do not describe planned M4 functionality as implemented.
- Do not introduce Redis, Celery, PostgreSQL, a vector database, GraphRAG, RAPTOR or
  multi-agent orchestration without a new approved plan.
