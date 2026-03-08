# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-28
**Commit:** 6f80dab
**Branch:** master

## OVERVIEW
Python 3.12+ RAG prototype with a LangGraph-based agent, indexing pipeline, and LLM wrapper.

## STRUCTURE
```
agentic_rag/
├── agent/        # agent graph, nodes/edges, prompts, tools
├── indexing/     # ingestion, chunking, embeddings, retrieval, vector store
├── llm/          # LLM adapter (OpenAI-compatible)
├── config.py     # runtime config constants
├── main.py       # CLI entrypoint
└── utils.py      # PDF conversion + token estimation helpers
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Agent orchestration | agent/research_search_agent.py | LangChain agent + middleware limits |
| Graph wiring | agent/graph.py | Node/edge graph assembly |
| Node logic | agent/nodes.py | summarize/rewrite/aggregate flow |
| Retrieval tools | agent/tools.py | ToolFactory exposes search tool |
| Indexing pipeline | indexing/indexer.py | end-to-end indexing flow |
| Retrieval logic | indexing/retriever.py | BM25 + vector fusion |
| Embeddings | indexing/embeddings.py | OpenAI-compatible embeddings only |
| Vector store | indexing/vectorstore.py | FAISS-backed store |
| LLM adapter | llm/llm.py | ChatOpenAI wrapper |
| Config knobs | config.py | chunk sizes, model names, paths |
| PDF → markdown | utils.py | PyMuPDF/PyMuPDF4LLM conversion |

## CONVENTIONS
- Python >= 3.12 (see `.python-version`, `pyproject.toml`).
- Dependencies are declared in `pyproject.toml` with minimum versions.
- `pyproject.toml` sets a default uv index to Tsinghua PyPI mirror; installs may pull from that mirror.

## ANTI-PATTERNS (THIS PROJECT)
- Do not commit runtime artifacts like `.venv/`, `__pycache__/`, or `.ruff_cache/`.
- Do not commit `.env` with secrets; use `.env.example` and environment variables.

## UNIQUE STYLES
- Agent uses a LangGraph state graph with explicit nodes/edges and middleware limits.
- Prompts are centralized in `agent/prompts.py` and should not be inlined elsewhere.

## COMMANDS
```bash
# Run the demo entrypoint
python main.py
```

## NOTES
- No CI workflows or test suites are present; add pytest/CI if needed.
- `llm/llm.py` requires API key/base URL; see `.env.example`.

