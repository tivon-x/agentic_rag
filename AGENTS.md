# PROJECT KNOWLEDGE BASE

**Updated:** 2026-07-26
**Branch:** codex/v2-core

## OVERVIEW

Python 3.12+ Agentic RAG system: LangGraph-based agent graph, hybrid BM25+FAISS retrieval, and Gradio UI.

## STRUCTURE

```
agentic_rag/
├── agent/          # graph wiring, nodes, edges, prompts, tools, schemas, states
├── api/            # FastAPI routes, SQLite migrations, persistent index worker
├── core/           # AppSettings, factory, persistence, corpus_profile, rag_answer
├── indexing/       # data_processor, chunker, embeddings, vectorstore, bm25_index, retriever, indexer
├── llms/           # ChatOpenAI adapter + task-type LLM router
├── ui/             # Gradio UI (gradio.py)
├── web/            # Next.js frontend
├── tests/          # pytest suite
├── main.py         # CLI entrypoint (index / ask / ui sub-commands)
└── pyproject.toml  # deps + tool config
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Graph wiring | agent/graph.py | node/edge assembly, compile with InMemorySaver |
| Node logic | agent/nodes.py | summarize_history → decide_retrieval → rewrite/direct/oos/aggregate |
| Routing edges | agent/edges.py | conditional edge functions |
| LangChain agent | agent/research_search_agent.py | create_agent + FallbackMiddleware |
| Prompts (ALL) | agent/prompts.py | every system prompt lives here; never inline elsewhere |
| Tool definitions | agent/tools.py | ToolFactory.create_tools() → search_relevant_chunks |
| Pydantic schemas | agent/schemas.py | RetrievalDecision, QueryAnalysis |
| Graph states | agent/states.py | GraphState, ResearchSearchState |
| LLM router | llms/llm.py | get_llm_by_type(task_type) — cached per model+task |
| Settings | core/settings.py | AppSettings dataclass + load_settings() |
| Factory | core/factory.py | wire up settings → LLM router + indexer |
| Database migrations | api/db/migrations.py | forward-only versions + pre-migration backup |
| Persistent index jobs | api/db/database.py | job state, idempotency, leases, recovery |
| Index worker | api/services/index_worker.py | single leased worker + heartbeat/retry |
| Upload API | api/routers/indexing.py | safe staging, validation, idempotency |
| Index versions | indexing/index_versions.py | immutable builds, manifest, activation/rollback |
| Corpus profile | core/corpus_profile.py | read/write corpus_profile.json |
| Indexing pipeline | indexing/indexer.py | end-to-end ingest flow |
| Chunkers | indexing/chunker.py | Recursive / Token / SemanticNLTK |
| Embeddings | indexing/embeddings.py | OpenAI-compatible; FakeEmbeddings for tests |
| Vector store | indexing/vectorstore.py | FAISS-backed, local persist |
| BM25 index | indexing/bm25_index.py | rank-bm25, serialised to bm25.pkl |
| Fusion retriever | indexing/retriever.py | BM25Retriever + FusionRetriever (hybrid) |
| Paper parser protocol | indexing/parsers/paper_parser.py | Project-owned parser schema + stable IDs |
| PDF parser pipeline | indexing/paper_ingestion.py | Timeout, quality gate, PyMuPDF4LLM + legacy fallback |
| Structure normalizer | indexing/parsers/structure_normalizer.py | Deterministic sections, blocks, tables, formulas |
| Passage materialization | indexing/passages.py | quote/retrieval split + embedding hard limit |
| Paper catalog repository | api/db/papers.py | papers, versions, sections, passages, metadata correction |
| Paper/Search APIs | api/routers/papers.py, api/routers/search.py | Range PDF delivery + page-addressable evidence |

---

## COMMANDS

```bash
# Install (use uv)
uv sync            # production deps
uv sync --extra dev  # + dev deps (pytest, ruff)

# Run
python main.py ui                         # launch Gradio
python main.py api                        # launch FastAPI
python main.py index <path>               # build and activate an index
python main.py activate-index <version>   # roll back to a ready version
python main.py ask "your question"        # CLI query

# Lint
uv run ruff check .
uv run ruff check . --fix

# Tests — full suite
uv run --extra dev python -m pytest -q

# Tests — single file
uv run pytest tests/test_retriever.py -v

# Tests — single test by name
uv run pytest tests/test_retriever.py::test_bm25_retriever_basic_query -v

# Tests — keyword match
uv run pytest -k "bm25" -v
```

No type-checker is configured in pyproject.toml; pyright/mypy can be run ad-hoc.

---

## CODE STYLE

### Formatting / Lint
- **Ruff** is the only linter/formatter (`ruff>=0.9.0`). No black, no isort.
- Ruff handles import ordering — do not reorder imports manually.
- No explicit `[tool.ruff]` section in pyproject.toml; defaults apply.

### Python Version
- Minimum **3.12**. Use modern syntax freely: `list[str]`, `dict[str, Any]`, `X | Y`, `match`.
- Use `from __future__ import annotations` at the top of files where forward references or deferred evaluation are needed (see `core/settings.py`, `llms/llm.py`).

### Imports
Follow this order (Ruff enforces it):
```python
# 1. stdlib
from __future__ import annotations
import os
from pathlib import Path

# 2. third-party
from langchain_core.documents import Document
from pydantic import BaseModel

# 3. local (absolute, project-root-relative)
from agent.prompts import get_research_search_prompt
from core.settings import AppSettings
```
No relative imports (`.module`) — use absolute imports from project root.

### Type Annotations
- Use built-in generics: `list[str]`, `dict[str, Any]`, `tuple[int, ...]` — NOT `List`, `Dict`, `Tuple` from `typing`.
- Prefer `X | None` over `Optional[X]`.
- Use `TypeAlias` for complex aliases: `ChatModel: TypeAlias = ChatOpenAI`.
- Annotate all public function parameters and return types.
- `Annotated[T, reducer]` is used in LangGraph states for custom reducers.

### Naming
| Construct | Convention | Example |
|-----------|------------|---------|
| Functions / variables | `snake_case` | `get_llm_by_type`, `corpus_profile` |
| Classes | `PascalCase` | `FusionRetriever`, `AppSettings` |
| Constants / module-level config | `UPPER_SNAKE_CASE` | `MAX_TOOL_CALLS`, `_LLM_CACHE` |
| Private helpers | leading underscore | `_build_chat_model`, `_get_env` |
| Graph state keys | `camelCase` | `routingDecision`, `corpusProfile` |

LangGraph `GraphState` fields use **camelCase** (matching LangGraph conventions). All other identifiers use `snake_case`.

### Docstrings
- Module-level docstrings: plain prose description (see `indexing/chunker.py`).
- Class docstrings: one-line summary (see `BM25Retriever`, `FusionRetriever`).
- Function docstrings: Google-style Args/Returns when non-obvious (see `indexing/retriever.py`).
- Simple internal helpers: no docstring required.

### Error Handling
- Catch only specific exceptions; bare `except Exception` only in node/middleware fallback paths where the system must not crash (e.g., `decide_retrieval`, `rewrite_query`).
- On fallback, assign a safe default and log or return a human-readable reason.
- Do **not** use empty `except` blocks.
- `ValueError` for invalid config/input (e.g., missing API key); no custom exception hierarchy exists yet.

---

## ARCHITECTURAL CONVENTIONS

### Agent Graph
- All nodes are plain functions `(state: GraphState) -> dict`. No class-based nodes.
- Conditional edges are separate functions in `agent/edges.py` — keep routing logic out of nodes.
- Every prompt string must live in `agent/prompts.py` and be retrieved via a `get_*_prompt()` function. Never inline system prompts in nodes or agent code.
- Structured outputs (`with_structured_output`) use Pydantic models from `agent/schemas.py`.

### LLM Router
- Access LLMs exclusively via `get_llm_by_type(task_type)`. Never instantiate `ChatOpenAI` directly in nodes.
- New task types: add the env var name to `_TASK_MODEL_ENV_NAMES` in `core/settings.py`.

### Settings
- All configuration flows through `AppSettings` (frozen dataclass). No naked `os.getenv()` calls outside `core/settings.py`.
- `load_settings()` is called once at startup (FastAPI lifespan or `main.py`); the result is passed down.

### Index Reliability
- API indexing runs only in `INDEX_WRITE_MODE=versioned`; legacy mode is API read-only, does not reconcile versioned pointers or consume queued jobs, and exists solely for explicit rollback.
- Versioned mode must never silently read or seed a legacy index without an embedding manifest.
- SQLite `app_state` is the authoritative active pointer; `active.json` is an atomic, startup-reconciled mirror.
- Versioned CLI index/activation commands initialize and migrate SQLite before activation; API startup imports a validated file-only pointer only when SQLite has no active state.
- Index jobs are persisted before execution and claimed only through the SQLite global/job leases.
- Schema migrations are forward-only and create a SQLite recovery backup before changing an existing database.
- User upload paths must remain under `UPLOAD_ROOT` after resolution; never concatenate an unchecked filename into a destination path.

### Paper Catalog
- `papers.id` is the SHA-256 of the uploaded file bytes; different bytes are distinct papers and are not auto-merged.
- `paper_versions`, `sections`, and `passages` use deterministic IDs derived from the paper, parser/normalizer versions, hierarchy, page, and quote.
- Keep `quote_text` source-faithful. Metadata corrections may rebuild `retrieval_text` prefixes but must not change quotes.
- Enforce `EMBEDDING_MAX_INPUT_CHARS` on the complete metadata prefix + passage before calling the embedding provider.
- PDF parsing defaults to PyMuPDF4LLM and the deterministic normalizer. `PAPER_PARSER=legacy` remains the rollback path.
- Parser failure, fallback, and `needs_ocr` are product states; do not hide them or promise OCR.
- `ParsedPage.source_text` and `source_fingerprint` are deterministic page-evidence fields; they do not replace PyMuPDF4LLM Markdown used for sections, passages, and quotes.
- A failed reparse must not downgrade a paper with a successful `latest_version_id`; retain the last usable catalog and record the latest attempt error separately.
- Catalog materialization must read current metadata while holding the SQLite write transaction. Metadata updates, passage-prefix refresh, and their reindex job are one atomic transaction.

### Testing
- Test files live in `tests/`, named `test_<module>.py`.
- Shared fixtures in `tests/conftest.py` (sample_documents, sample_text, tmp_env_file).
- Use `FakeEmbeddings` from `indexing/embeddings.py` for tests that need a vector store — no real API calls.
- Use `tmp_path` (pytest built-in) for file system operations.
- Use `monkeypatch` to isolate environment variables; always `delenv` before setting to avoid cross-test pollution.
- Test names: `test_<what>_<condition>` (e.g., `test_bm25_retriever_respects_k_parameter`).

---

## ANTI-PATTERNS

- Do **not** commit `.venv/`, `__pycache__/`, `.ruff_cache/`, `data/index/`, or `logs/`.
- Do **not** commit `.env` with real secrets; use `.env.example`.
- Do **not** inline prompts outside `agent/prompts.py`.
- Do **not** call `os.getenv()` outside `core/settings.py`.
- Do **not** instantiate `ChatOpenAI` directly — always use `get_llm_by_type()`.
- Do **not** bypass `VectorStore` to access FAISS internals directly.
- Do **not** use `List`, `Dict`, `Optional` from `typing` — use built-in generics instead.
