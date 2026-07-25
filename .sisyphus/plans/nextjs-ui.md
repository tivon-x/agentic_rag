# Next.js Production UI for Agentic RAG

## TL;DR

> **Quick Summary**: Build a production-ready Next.js frontend with FastAPI backend layer for the existing Python RAG system. Keep Gradio for development/debugging. Features: chat with SSE streaming, knowledge base builder, citation display.
> 
> **Deliverables**:
> - FastAPI backend (`api/`) with SSE streaming, corpus profile CRUD, background indexing
> - Next.js 14 frontend (`web/`) with chat interface, KB builder, citation accordion
> - SQLite persistence for chat sessions
> - Vercel AI SDK Elements for chat UI
> 
> **Estimated Effort**: Large (15-20 tasks across 4 waves)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: T1 → T4 → T8 → T12 → T15 → F1-F4 → user okay

---

## Context

### Original Request
User wants a Next.js UI for their existing Python RAG system (agentic_rag). The current Gradio UI will be kept for development/debugging while Next.js serves as the production frontend.

### Interview Summary
**Key Discussions**:
- Primary goal: Separate Production Frontend (Next.js alongside Gradio)
- Architecture: FastAPI + SSE + Next.js 14 App Router
- Features IN: KB Builder, Chat, Citations | OUT: Debug Panel, Tests, Auth
- Styling: shadcn/ui + Tailwind CSS
- Language: Chinese (Simplified)
- Package manager: pnpm
- Ports: FastAPI 8000, Next.js 3000

**Research Findings**:
- Use `sse-starlette.EventSourceResponse` for SSE streaming
- Use Vercel AI SDK Elements for chat UI (shadcn-chat is deprecated)
- Follow `ui/gradio.py:809-823` pattern for `graph.astream_events`
- Implement graph caching with fingerprint-based invalidation

### Metis Review
**Identified Gaps** (addressed):
- shadcn-chat deprecated → Switched to Vercel AI SDK Elements
- Indexing timeout risk → Background tasks with polling
- Session persistence → SQLite added
- Graph caching → Will follow Gradio's fingerprint pattern

---

## Work Objectives

### Core Objective
Add a FastAPI backend layer exposing the existing RAG system via REST/SSE endpoints, and build a Next.js production frontend with chat streaming, knowledge base builder, and citation display.

### Concrete Deliverables
- `api/` directory with FastAPI app (main.py, routers/, models/)
- `web/` directory with Next.js 14 app (App Router, shadcn/ui, Tailwind)
- SQLite database for session persistence
- Working chat with SSE streaming
- Working knowledge base builder with file upload
- Citation/evidence display in chat responses

### Definition of Done
- [ ] `curl http://localhost:8000/api/health` returns `{"status": "ok"}`
- [ ] `curl http://localhost:8000/api/chat -d '{"message":"你好"}'` returns session_id
- [ ] `curl http://localhost:8000/api/chat/stream?session_id=<id>` streams SSE tokens
- [ ] `curl http://localhost:8000/api/corpus-profile` returns corpus profile JSON
- [ ] `pnpm --prefix web dev` starts Next.js on localhost:3000
- [ ] Chat page sends message and receives streaming response
- [ ] KB Builder saves corpus profile and uploads files for indexing

### Must Have
- SSE streaming for chat responses with LangGraph integration
- Corpus profile CRUD (all fields from current Gradio UI)
- File upload for .pdf, .md, .txt with background indexing
- Citation accordion showing evidence sources
- Chinese UI text throughout
- Graph caching with fingerprint invalidation

### Must NOT Have (Guardrails)
- Debug panel (keep in Gradio only)
- Index statistics display
- Authentication/login system
- User management
- Multiple knowledge bases
- File management/deletion UI
- Export/import corpus profiles
- Theme customization
- Mobile-specific layouts
- GZipMiddleware on SSE endpoints (breaks streaming)
- Exposed LLM API keys to frontend
- build_graph() called per-request without caching
- Importing from ui/gradio.py in FastAPI code

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: NO (new project directories)
- **Automated tests**: Tests-after (add after MVP functional)
- **Framework**: pytest (FastAPI) + vitest (Next.js)
- **If TDD**: N/A for MVP

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Frontend/UI**: Use Playwright (playwright skill) — Navigate, interact, assert DOM, screenshot
- **API/Backend**: Use Bash (curl) — Send requests, assert status + response fields
- **Library/Module**: Use Bash (python REPL) — Import, call functions, compare output

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 5-8 tasks per wave.

```
Wave 1 (Foundation — scaffolding, no dependencies):
├── Task 1: FastAPI app skeleton + health endpoint + CORS [quick]
├── Task 2: Next.js 14 init with pnpm + Tailwind + shadcn/ui [quick]
├── Task 3: SQLite session persistence setup [quick]
├── Task 4: Pydantic models for API contracts [quick]
└── Task 5: Graph caching module (follow Gradio pattern) [unspecified-high]

Wave 2 (Core Backend — depends on Wave 1):
├── Task 6: Corpus profile GET/PUT endpoints (depends: 1, 4) [unspecified-high]
├── Task 7: Background job queue for indexing (depends: 1, 3) [deep]
├── Task 8: Chat session creation endpoint (depends: 1, 3, 4, 5) [unspecified-high]
├── Task 9: SSE streaming endpoint with LangGraph (depends: 1, 4, 5) [deep]
└── Task 10: File upload + indexing endpoint (depends: 1, 4, 7) [unspecified-high]

Wave 3 (Frontend — depends on Wave 1-2):
├── Task 11: useSSEStream hook with reconnection (depends: 2) [unspecified-high]
├── Task 12: Chat page with streaming messages (depends: 2, 11) [visual-engineering]
├── Task 13: Citation accordion component (depends: 2) [visual-engineering]
├── Task 14: Knowledge base builder page (depends: 2) [visual-engineering]
└── Task 15: File upload component with drag-drop (depends: 2, 14) [visual-engineering]

Wave 4 (Integration — connects frontend to backend):
├── Task 16: Connect chat page to FastAPI SSE stream (depends: 9, 12) [deep]
├── Task 17: Connect KB builder to FastAPI endpoints (depends: 6, 10, 14, 15) [unspecified-high]
└── Task 18: Add Chinese UI text + error messages (depends: 12, 14) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T5 → T8 → T9 → T12 → T16 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 5 (Waves 1, 2, 3)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|------------|--------|------|
| 1 | — | 6,7,8,9,10 | 1 |
| 2 | — | 11,12,13,14,15 | 1 |
| 3 | — | 7,8 | 1 |
| 4 | — | 6,8,9,10 | 1 |
| 5 | — | 8,9 | 1 |
| 6 | 1,4 | 17 | 2 |
| 7 | 1,3 | 10 | 2 |
| 8 | 1,3,4,5 | 16 | 2 |
| 9 | 1,4,5 | 11,16 | 2 |
| 10 | 1,4,7 | 17 | 2 |
| 11 | 2,9 | 12 | 3 |
| 12 | 2,11 | 16,18 | 3 |
| 13 | 2 | 16 | 3 |
| 14 | 2 | 15,17,18 | 3 |
| 15 | 2,14 | 17 | 3 |
| 16 | 9,12,13 | F1-F4 | 4 |
| 17 | 6,10,14,15 | F1-F4 | 4 |
| 18 | 12,14 | F1-F4 | 4 |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks — T1-T4 → `quick`, T5 → `unspecified-high`
- **Wave 2**: 5 tasks — T6,T8,T10 → `unspecified-high`, T7,T9 → `deep`
- **Wave 3**: 5 tasks — T11 → `unspecified-high`, T12-T15 → `visual-engineering`
- **Wave 4**: 3 tasks — T16 → `deep`, T17 → `unspecified-high`, T18 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [ ] 1. FastAPI app skeleton + health endpoint + CORS

  **What to do**:
  - Add Python dependencies to `pyproject.toml`:
    - `fastapi>=0.110.0`
    - `uvicorn[standard]>=0.27.0`
    - `sse-starlette>=2.0.0`
    - `aiosqlite>=0.20.0`
    - `python-multipart>=0.0.9`
  - Run `uv sync` to install new dependencies
  - Create `api/` directory with `main.py`, `__init__.py`
  - Set up FastAPI app with `/api/health` GET endpoint returning `{"status": "ok"}`
  - Configure CORS middleware allowing `http://localhost:3000` origin
  - Add `api/routers/__init__.py` for future router modules
  - Do NOT add GZipMiddleware (breaks SSE streaming)

  **Must NOT do**:
  - No authentication middleware
  - No GZipMiddleware on the app
  - No importing from `ui/gradio.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple scaffolding, single file creation, no complex logic
  - **Skills**: `[]`
    - No special skills needed for basic FastAPI setup

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Tasks 6, 7, 8, 9, 10
  - **Blocked By**: None (can start immediately)

  **References**:
  - `ui/gradio.py:1-50` - See how existing app imports core modules
  - `core/settings.py` - AppSettings for configuration pattern
  - FastAPI docs: https://fastapi.tiangolo.com/tutorial/cors/

  **Acceptance Criteria**:
  - [ ] `api/main.py` exists with FastAPI app instance
  - [ ] `api/routers/__init__.py` exists (empty file for now)
  - [ ] CORS configured for localhost:3000
  - [ ] `pyproject.toml` includes fastapi, uvicorn, sse-starlette, aiosqlite, python-multipart
  - [ ] `uv sync` succeeds without errors

  **QA Scenarios**:

  ```
  Scenario: Health endpoint returns OK
    Tool: Bash (python + curl)
    Preconditions: Dependencies installed via uv sync
    Steps:
      1. Start server via Python subprocess (cross-platform):
         `python -c "import subprocess, sys; p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app', '--port', '8000']); import time; time.sleep(3); print('SERVER_PID:', p.pid)"`
      2. `curl -s http://localhost:8000/api/health`
      3. Parse JSON response
      4. Stop server via Python: `python -c "import os, signal; os.kill(<PID>, signal.SIGTERM)"`
    Expected Result: Response body is exactly `{"status":"ok"}`, HTTP 200
    Failure Indicators: Non-200 status, missing "status" key, value not "ok"
    Evidence: .sisyphus/evidence/task-1-health-endpoint.json

  Scenario: CORS allows localhost:3000 origin
    Tool: Bash (curl)
    Preconditions: FastAPI server running on port 8000 (use same subprocess approach)
    Steps:
      1. Start server via Python subprocess as above
      2. `curl -s -I -X OPTIONS http://localhost:8000/api/health -H "Origin: http://localhost:3000" -H "Access-Control-Request-Method: GET"`
      3. Check response headers
      4. Stop server
    Expected Result: `Access-Control-Allow-Origin: http://localhost:3000` header present
    Failure Indicators: Missing CORS header, wrong origin value, 4xx response
    Evidence: .sisyphus/evidence/task-1-cors-headers.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add FastAPI skeleton with health endpoint and CORS`
  - Files: `api/main.py`, `api/__init__.py`, `api/routers/__init__.py`

---

- [ ] 2. Next.js 14 init with pnpm + Tailwind + shadcn/ui

  **What to do**:
  - Run `pnpm create next-app@latest web --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"`
  - Initialize shadcn/ui: `cd web && pnpm dlx shadcn@latest init`
  - Add shadcn components: `pnpm dlx shadcn@latest add button card input textarea scroll-area accordion`
  - Create `web/src/components/ui/` directory structure
  - Configure `next.config.js` for API proxy to localhost:8000 (optional, can use direct fetch)

  **Must NOT do**:
  - No authentication setup
  - No theme customization beyond defaults
  - No mobile-specific layouts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard project scaffolding with CLI commands
  - **Skills**: `[]`
    - No special skills needed, just CLI execution

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Tasks 11, 12, 13, 14, 15
  - **Blocked By**: None (can start immediately)

  **References**:
  - Next.js docs: https://nextjs.org/docs/getting-started/installation
  - shadcn/ui docs: https://ui.shadcn.com/docs/installation/next

  **Acceptance Criteria**:
  - [ ] `web/` directory exists with Next.js 14 App Router structure
  - [ ] `web/package.json` has Next.js, React, Tailwind dependencies
  - [ ] `web/src/components/ui/button.tsx` exists (from shadcn)
  - [ ] `pnpm --prefix web dev` starts without errors

  **QA Scenarios**:

  ```
  Scenario: Next.js dev server starts successfully
    Tool: Bash
    Preconditions: web/ directory exists with dependencies installed
    Steps:
      1. `cd web && pnpm install` (if not already)
      2. `pnpm dev &` (background)
      3. Wait 5 seconds for compilation
      4. `curl -s http://localhost:3000`
    Expected Result: HTTP 200, HTML response containing "<!DOCTYPE html>"
    Failure Indicators: Connection refused, compilation errors, non-200 status
    Evidence: .sisyphus/evidence/task-2-nextjs-start.txt

  Scenario: shadcn/ui components installed
    Tool: Bash
    Preconditions: web/ directory exists
    Steps:
      1. `ls web/src/components/ui/`
      2. Check for button.tsx, card.tsx, input.tsx
    Expected Result: All specified component files exist
    Failure Indicators: Missing files, empty directory
    Evidence: .sisyphus/evidence/task-2-shadcn-components.txt
  ```

  **Commit**: YES
  - Message: `feat(web): init Next.js 14 with pnpm, Tailwind, shadcn/ui`
  - Files: `web/`

---

- [ ] 3. SQLite session persistence setup

  **What to do**:
  - Create `api/db/` directory with `__init__.py`, `database.py`, `models.py`
  - Use SQLite with `aiosqlite` for async operations
  - Define `ChatSession` table: `id (UUID PK)`, `created_at`, `updated_at`, `messages (JSON)`
  - Define `IndexingJob` table: `id (UUID PK)`, `status (pending/running/completed/failed)`, `created_at`, `updated_at`, `error_message`
  - Create `get_db()` async context manager for session management
  - Auto-create tables on app startup

  **Must NOT do**:
  - No user authentication tables
  - No complex migrations (simple create-if-not-exists)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard SQLite setup, no complex logic
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: None (can start immediately)

  **References**:
  - `aiosqlite` docs: https://aiosqlite.omnilib.dev/en/stable/
  - SQLite JSON functions: https://www.sqlite.org/json1.html

  **Acceptance Criteria**:
  - [ ] `api/db/database.py` exists with async SQLite connection
  - [ ] `api/db/models.py` defines ChatSession and IndexingJob schemas
  - [ ] Tables auto-create on first connection

  **QA Scenarios**:

  ```
  Scenario: Database tables created successfully
    Tool: Bash (python)
    Preconditions: api/db/ module exists
    Steps:
      1. `python -c "import asyncio; from api.db.database import init_db; asyncio.run(init_db())"`
      2. `python -c "import sqlite3; conn = sqlite3.connect('data/sessions.db'); cursor = conn.execute('SELECT name FROM sqlite_master WHERE type=\\'table\\''); tables = [r[0] for r in cursor.fetchall()]; print('TABLES:', tables); conn.close()"`
    Expected Result: Output contains "chat_sessions" and "indexing_jobs"
    Failure Indicators: Import error, missing tables, SQLite error
    Evidence: .sisyphus/evidence/task-3-db-tables.txt

  Scenario: Session CRUD operations work
    Tool: Bash (python)
    Preconditions: Database initialized
    Steps:
      1. Run Python script that creates, reads, updates a session
      2. Verify session data roundtrip
    Expected Result: Session created with UUID, retrieved with same data
    Failure Indicators: Insert/select failure, data corruption
    Evidence: .sisyphus/evidence/task-3-session-crud.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add SQLite session persistence module`
  - Files: `api/db/`

---

- [ ] 4. Pydantic models for API contracts

  **What to do**:
  - Create `api/models/` directory with `__init__.py`
  - Create `api/models/chat.py`: `ChatRequest`, `ChatResponse`, `StreamToken`, `ChatMessage`
  - Create `api/models/corpus.py`: `CorpusProfile` with ALL fields from `core/corpus_profile.py`:
    - `name: str`
    - `summary: str`
    - `coverage: str`
    - `non_coverage: str`
    - `usage_notes: str`
    - `source_examples: list[str]`
    - `recommended_questions: list[str]`
    - `forbidden_questions: list[str]`
    - `domain_keywords: list[str]`
    - `preferred_answer_style: str`
    - `primary_entities: list[str]`
  - Create `api/models/indexing.py`: `IndexingJobStatus`, `IndexingJobResponse`, `FileUploadResponse`
  - All models use Pydantic v2 syntax (`model_config`, `ConfigDict`)

  **Must NOT do**:
  - No user/auth models
  - No importing from core/corpus_profile.py directly (define own Pydantic model)
  - No inventing fields not in core/corpus_profile.py (e.g., no "domain", "tone", "language")

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure Pydantic model definitions, no complex logic
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Tasks 6, 8, 9, 10
  - **Blocked By**: None (can start immediately)

  **References**:
  - `core/corpus_profile.py` - CorpusProfile fields to mirror
  - `agent/schemas.py` - Pydantic model patterns in this codebase
  - `agent/states.py:GraphState` - See what fields are in graph output

  **Acceptance Criteria**:
  - [ ] `api/models/chat.py` defines ChatRequest, ChatResponse, StreamToken
  - [ ] `api/models/corpus.py` defines CorpusProfile with all fields
  - [ ] `api/models/indexing.py` defines IndexingJobStatus, IndexingJobResponse
  - [ ] All models importable without errors

  **QA Scenarios**:

  ```
  Scenario: Models import and validate correctly
    Tool: Bash (python)
    Preconditions: api/models/ exists
    Steps:
      1. `cd api && python -c "from models.chat import ChatRequest, ChatResponse; from models.corpus import CorpusProfile; from models.indexing import IndexingJobStatus; print('OK')"`
    Expected Result: Prints "OK" without import errors
    Failure Indicators: ImportError, ModuleNotFoundError
    Evidence: .sisyphus/evidence/task-4-models-import.txt

  Scenario: ChatRequest validates message field
    Tool: Bash (python)
    Preconditions: Models exist
    Steps:
      1. `python -c "from api.models.chat import ChatRequest; r = ChatRequest(message='test'); print(r.message)"`
      2. `python -c "from api.models.chat import ChatRequest; ChatRequest()"` (should fail)
    Expected Result: First prints "test", second raises ValidationError
    Failure Indicators: Missing required field not caught
    Evidence: .sisyphus/evidence/task-4-validation.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add Pydantic models for API contracts`
  - Files: `api/models/`

---

- [ ] 5. Graph caching module (follow Gradio pattern)

  **What to do**:
  - Create `api/services/` directory with `__init__.py`
  - Create `api/services/graph_cache.py` following pattern from `ui/gradio.py:75-102`
  - Implement `_compute_cache_fingerprint()` using settings hash
  - Implement `get_cached_graph(settings: AppSettings)` returning compiled graph
  - Cache invalidation when fingerprint changes
  - Use module-level cache dict (same pattern as Gradio)

  **Must NOT do**:
  - No importing from ui/gradio.py (copy pattern, don't import)
  - No per-request graph compilation (defeats caching purpose)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding existing pattern and adapting it correctly
  - **Skills**: `[]`
    - No special skills, but requires careful pattern matching

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: None (can start immediately)

  **References**:
  - `ui/gradio.py:75-102` - CRITICAL: Follow this exact caching pattern
  - `core/factory.py:build_graph()` - The function being cached
  - `core/settings.py:AppSettings` - Settings used for fingerprint

  **Acceptance Criteria**:
  - [ ] `api/services/graph_cache.py` exists with `get_cached_graph()` function
  - [ ] Fingerprint computed from settings hash
  - [ ] Same settings → same graph instance (cache hit)
  - [ ] Changed settings → new graph (cache invalidation)

  **QA Scenarios**:

  ```
  Scenario: Graph caching returns same instance for same settings
    Tool: Bash (python)
    Preconditions: graph_cache module exists, .env configured
    Steps:
      1. `python -c "
         from api.services.graph_cache import get_cached_graph
         from core.settings import load_settings
         settings = load_settings()
         g1 = get_cached_graph(settings)
         g2 = get_cached_graph(settings)
         print('SAME' if g1 is g2 else 'DIFFERENT')
         "`
    Expected Result: Prints "SAME" (identical object reference)
    Failure Indicators: Prints "DIFFERENT", import errors
    Evidence: .sisyphus/evidence/task-5-cache-hit.txt

  Scenario: Fingerprint changes invalidate cache
    Tool: Bash (python)
    Preconditions: graph_cache module exists, .env configured
    Steps:
      1. `python -c "
         import os
         from api.services.graph_cache import get_cached_graph
         from core.settings import load_settings
         
         # Get graph with original settings
         settings_a = load_settings()
         g1 = get_cached_graph(settings_a)
         
         # Change an env var that affects settings fingerprint (LLM_MODEL is used in settings)
         original_val = os.environ.get('LLM_MODEL', '')
         os.environ['LLM_MODEL'] = 'different-model-for-cache-test'
         
         # Reload settings with changed env
         settings_b = load_settings()
         g2 = get_cached_graph(settings_b)
         
         # Restore original
         if original_val:
             os.environ['LLM_MODEL'] = original_val
         else:
             os.environ.pop('LLM_MODEL', None)
         
         print('DIFFERENT' if g1 is not g2 else 'SAME (BUG)')
         "`
    Expected Result: Prints "DIFFERENT" (different instances for different settings)
    Failure Indicators: Prints "SAME (BUG)" - cache not invalidated
    Evidence: .sisyphus/evidence/task-5-cache-invalidation.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add graph caching with fingerprint invalidation`
  - Files: `api/services/`

---

- [ ] 6. Corpus profile GET/PUT endpoints

  **What to do**:
  - Create `api/routers/corpus.py` with APIRouter
  - `GET /api/corpus-profile`:
    - First check if file exists: `corpus_profile_path(settings.index_dir).exists()`
    - If NOT exists: return HTTP 404 with `{"error": "Corpus profile not found. Index documents first."}`
    - If exists: read via `core.corpus_profile.load_corpus_profile()` and return
  - `PUT /api/corpus-profile` - Write via `core.corpus_profile.save_corpus_profile()`
  - Convert between internal format and API Pydantic models
  - Register router in `api/main.py`

  **Must NOT do**:
  - No DELETE endpoint
  - No export/import functionality
  - No multiple corpus profiles
  - Do NOT call load_corpus_profile() for 404 check (it returns defaults even when missing)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integrates with existing core module, needs proper error handling
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10)
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 1, 4

  **References**:
  - `core/corpus_profile.py` - load_corpus_profile(), save_corpus_profile() functions
  - `api/models/corpus.py` - CorpusProfile Pydantic model (from Task 4)
  - `ui/gradio.py:150-200` - How Gradio handles corpus profile

  **Acceptance Criteria**:
  - [ ] `api/routers/corpus.py` exists with GET and PUT endpoints
  - [ ] Router registered in `api/main.py`
  - [ ] GET returns 200 with profile or 404 if missing
  - [ ] PUT saves profile and returns 200

  **QA Scenarios**:

  ```
  Scenario: GET corpus-profile returns 404 when missing
    Tool: Bash (python + curl)
    Preconditions: No index exists (clean state), server running
    Steps:
      1. Delete any existing corpus_profile.json using settings-aware path:
         `python -c "from core.settings import load_settings; from core.corpus_profile import corpus_profile_path; p = corpus_profile_path(load_settings().index_dir); p.unlink(missing_ok=True); print('Deleted:', p)"`
      2. Start server via Python subprocess
      3. `curl -s -w "\n%{http_code}" http://localhost:8000/api/corpus-profile`
      4. Stop server
    Expected Result: HTTP 404, body contains error message
    Failure Indicators: Returns 200 with empty body, 500 error
    Evidence: .sisyphus/evidence/task-6-corpus-404.txt

  Scenario: PUT corpus-profile saves and GET retrieves
    Tool: Bash (python + curl)
    Preconditions: Server running
    Steps:
      1. Start server via Python subprocess
      2. `curl -X PUT http://localhost:8000/api/corpus-profile -H "Content-Type: application/json" -d '{"name":"测试知识库","summary":"测试摘要","coverage":"覆盖范围","non_coverage":"","usage_notes":"使用说明","source_examples":[],"recommended_questions":[],"forbidden_questions":[],"domain_keywords":["测试"],"preferred_answer_style":"简洁","primary_entities":[]}'`
      3. `curl http://localhost:8000/api/corpus-profile`
      4. Parse response JSON
      5. Stop server
    Expected Result: PUT returns 200, GET returns same data with name="测试知识库"
    Failure Indicators: PUT fails, GET returns different data, missing fields
    Evidence: .sisyphus/evidence/task-6-corpus-crud.json
  ```

  **Commit**: YES
  - Message: `feat(api): add corpus profile GET/PUT endpoints`
  - Files: `api/routers/corpus.py`, `api/main.py` (router registration)

---

- [ ] 7. Background job queue for indexing

  **What to do**:
  - Create `api/jobs/` directory with `__init__.py`, `queue.py`, `worker.py`
  - Implement simple in-memory job queue with asyncio
  - `JobQueue.submit(job_type, payload)` → returns job_id (UUID)
  - `JobQueue.get_status(job_id)` → returns status, progress, error
  - Background worker processes jobs sequentially (single worker, no concurrency)
  - Integrate with SQLite `IndexingJob` table for persistence
  - Job statuses: `pending`, `running`, `completed`, `failed`

  **Must NOT do**:
  - No Redis/Celery (keep simple, in-process)
  - No parallel job execution (indexing is single-threaded)
  - No job cancellation (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful async design, background task coordination
  - **Skills**: `[]`
    - No special skills, but needs solid async Python understanding

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8, 9, 10)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `indexing/indexer.py` - The Indexer class that jobs will call
  - `api/db/models.py` - IndexingJob table schema (from Task 3)
  - Python asyncio.create_task docs

  **Acceptance Criteria**:
  - [ ] `api/jobs/queue.py` defines JobQueue class
  - [ ] `api/jobs/worker.py` defines background worker
  - [ ] Jobs persist to SQLite and survive restarts (pending jobs)
  - [ ] Status polling works correctly

  **QA Scenarios**:

  ```
  Scenario: Job submission returns job_id immediately
    Tool: Bash (python)
    Preconditions: JobQueue module exists
    Steps:
      1. `python -c "
         import asyncio
         from api.jobs.queue import JobQueue
         async def test():
             q = JobQueue()
             job_id = await q.submit('index', {'path': '/tmp/test'})
             print(f'JOB_ID:{job_id}')
         asyncio.run(test())
         "`
    Expected Result: Prints "JOB_ID:<uuid>" immediately without blocking
    Failure Indicators: Blocks, no job_id returned, exception
    Evidence: .sisyphus/evidence/task-7-job-submit.txt

  Scenario: Job status transitions correctly
    Tool: Bash (python)
    Preconditions: JobQueue with worker running
    Steps:
      1. Submit a test job
      2. Poll status immediately (should be pending or running)
      3. Wait for completion
      4. Check final status
    Expected Result: Status transitions pending→running→completed
    Failure Indicators: Stuck in pending, wrong final status
    Evidence: .sisyphus/evidence/task-7-job-status.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add background job queue for indexing`
  - Files: `api/jobs/`

---

- [ ] 8. Chat session creation endpoint

  **What to do**:
  - Create `api/routers/chat.py` with APIRouter
  - `POST /api/chat` - Accept `ChatRequest`, create session in SQLite, return `ChatResponse` with session_id
  - Store initial message in session's messages JSON array
  - Session ID is UUID v4
  - Register router in `api/main.py`

  **Must NOT do**:
  - No streaming in this endpoint (separate endpoint for SSE)
  - No actual LLM calls (just session creation)
  - No authentication

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integrates DB, models, routing - moderate complexity
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 9, 10)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 1, 3, 4, 5

  **References**:
  - `api/db/models.py` - ChatSession schema (from Task 3)
  - `api/models/chat.py` - ChatRequest, ChatResponse (from Task 4)
  - `ui/gradio.py:809-823` - How Gradio initiates a chat

  **Acceptance Criteria**:
  - [ ] `api/routers/chat.py` exists with POST endpoint
  - [ ] Session created in SQLite with UUID
  - [ ] Returns `{"session_id": "<uuid>"}`
  - [ ] Router registered in `api/main.py`

  **QA Scenarios**:

  ```
  Scenario: POST /api/chat creates session and returns ID
    Tool: Bash (curl)
    Preconditions: Server running, database initialized
    Steps:
      1. `curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message":"你好"}'`
      2. Parse JSON response
      3. Extract session_id
    Expected Result: HTTP 200, response contains "session_id" with UUID format
    Failure Indicators: Missing session_id, non-UUID format, 4xx/5xx error
    Evidence: .sisyphus/evidence/task-8-chat-create.json

  Scenario: Session persists in database
    Tool: Bash (python + sqlite3)
    Preconditions: Session created via API
    Steps:
      1. Create session via POST
      2. Extract session_id
      3. Query SQLite: `SELECT * FROM chat_sessions WHERE id = '<session_id>'`
    Expected Result: Row exists with matching ID and message in JSON
    Failure Indicators: No row found, corrupted data
    Evidence: .sisyphus/evidence/task-8-session-db.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add chat session creation endpoint`
  - Files: `api/routers/chat.py`, `api/main.py`

---

- [ ] 9. SSE streaming endpoint with LangGraph

  **What to do**:
  - Add `GET /api/chat/stream` endpoint to `api/routers/chat.py`
  - Accept `session_id` query parameter
  - Use `sse-starlette.EventSourceResponse` for SSE
  - Retrieve session from SQLite, get user message
  - Call `graph.astream_events()` following `ui/gradio.py:809-850` pattern
  - Emit events: `token` (content delta), `citation` (evidence), `done` (completion)
  - Handle `asyncio.CancelledError` properly (re-raise, don't swallow)
  - Check `await request.is_disconnected()` before expensive operations
  - Update session messages JSON with assistant response when complete

  **Must NOT do**:
  - No GZipMiddleware (breaks SSE)
  - No authentication
  - No concurrent graph invocations per session

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex async streaming, LangGraph integration, error handling
  - **Skills**: `[]`
    - Requires understanding of SSE, asyncio, LangGraph events

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 10)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 1, 4, 5

  **References**:
  - `ui/gradio.py:809-850` - CRITICAL: Follow this exact streaming pattern
  - `agent/graph.py` - How graph is compiled and invoked
  - `agent/states.py:GraphState` - Output state structure
  - SSE research: use `sse-starlette.EventSourceResponse`

  **Acceptance Criteria**:
  - [ ] `GET /api/chat/stream?session_id=<id>` returns SSE stream
  - [ ] Emits `event: token` with data containing content delta
  - [ ] Emits `event: citation` with evidence data
  - [ ] Emits `event: done` when complete
  - [ ] Properly handles client disconnect

  **QA Scenarios**:

  ```
  Scenario: SSE stream emits tokens for valid session
    Tool: Bash (python + curl)
    Preconditions: 
      - .env file exists with valid LLM_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_API_KEY
      - Index exists (run indexing first via Task 10 or existing index in data/index/)
      - Server running
    Steps:
      1. Ensure index exists. If not, create minimal index:
         `python -c "from pathlib import Path; assert Path('data/index/faiss').exists(), 'Run indexing first or create test index'"`
         OR create test corpus: save a test.txt file and call indexer
      2. Start server via Python subprocess
      3. Create session: `curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message":"什么是RAG?"}'`
      4. Extract session_id from response
      5. `curl -N "http://localhost:8000/api/chat/stream?session_id=<id>" -H "Accept: text/event-stream" --max-time 30`
      6. Parse SSE events
      7. Stop server
    Expected Result: Receives multiple "event: token" lines with data, ends with "event: done"
    Failure Indicators: Connection closed immediately, no events, only error event
    Note: If no index exists, expect "event: error" with message about missing index
    Evidence: .sisyphus/evidence/task-9-sse-stream.txt

  Scenario: Invalid session_id returns error event
    Tool: Bash (curl)
    Preconditions: Server running (no index required for this test)
    Steps:
      1. Start server via Python subprocess
      2. `curl -N "http://localhost:8000/api/chat/stream?session_id=invalid-uuid-12345" -H "Accept: text/event-stream" --max-time 5`
      3. Stop server
    Expected Result: Receives "event: error" with message about invalid/missing session
    Failure Indicators: 500 error, connection hang, no error event
    Evidence: .sisyphus/evidence/task-9-sse-error.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add SSE streaming endpoint with LangGraph`
  - Files: `api/routers/chat.py`

---

- [ ] 10. File upload + indexing endpoint

  **What to do**:
  - Create `api/routers/indexing.py` with APIRouter
  - `POST /api/index` - Accept multipart file upload (.pdf, .md, .txt)
  - Save uploaded file to temporary location
  - Submit indexing job to JobQueue (from Task 7)
  - Return `{"job_id": "<uuid>"}` immediately (non-blocking)
  - `GET /api/jobs/{job_id}` - Poll job status
  - Register router in `api/main.py`
  - Call `indexing.indexer.Indexer(cfg).index(file_path)` in job worker

  **Must NOT do**:
  - No file management/deletion UI
  - No multiple file upload in single request
  - No direct indexing (must go through job queue)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: File handling, async job coordination, multiple endpoints
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 9)
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 1, 4, 7

  **References**:
  - `indexing/indexer.py` - Indexer class and index() method
  - `api/jobs/queue.py` - JobQueue for async execution (from Task 7)
  - `main.py:cmd_index` (line 31) - CLI indexing for reference

  **Acceptance Criteria**:
  - [ ] `POST /api/index` accepts file upload
  - [ ] Returns job_id immediately (< 100ms response)
  - [ ] `GET /api/jobs/{job_id}` returns status
  - [ ] Indexing actually runs in background

  **QA Scenarios**:

  ```
  Scenario: File upload creates indexing job
    Tool: Bash (python + curl)
    Preconditions: Server running
    Steps:
      1. Create test file via Python:
         `python -c "import tempfile; f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False); f.write('Test content for indexing'); f.close(); print(f.name)"`
         (capture output as TEMP_FILE)
      2. Start server via Python subprocess
      3. `curl -s -X POST http://localhost:8000/api/index -F "file=@<TEMP_FILE>"`
      4. Parse response for job_id
      5. Cleanup temp file: `python -c "import os; os.unlink('<TEMP_FILE>')"`
      6. Stop server
    Expected Result: HTTP 200, response contains "job_id" with UUID format
    Failure Indicators: Slow response (> 1s), missing job_id, 4xx error
    Evidence: .sisyphus/evidence/task-10-upload-job.json

  Scenario: Job status polling works
    Tool: Bash (python + curl)
    Preconditions: Job submitted, server running
    Steps:
      1. Submit file and get job_id (use same approach as above)
      2. `curl http://localhost:8000/api/jobs/<job_id>`
      3. Poll every 2 seconds until status is "completed" or "failed"
      4. Stop server
    Expected Result: Status transitions from pending→running→completed
    Failure Indicators: 404, stuck status, never completes within 60s
    Evidence: .sisyphus/evidence/task-10-job-poll.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add file upload and indexing endpoint`
  - Files: `api/routers/indexing.py`, `api/main.py`

---

- [ ] 11. useSSEStream hook with reconnection

  **What to do**:
  - Create `web/src/hooks/useSSEStream.ts`
  - Custom React hook for SSE consumption
  - Parameters: `url`, `onToken`, `onCitation`, `onDone`, `onError`
  - Implement exponential backoff reconnection (1s, 2s, 4s, 8s, 16s max)
  - Use native EventSource API
  - Handle connection states: connecting, connected, disconnected, error
  - Cleanup on unmount
  - Create test page `web/src/app/test-sse/page.tsx` for QA:
    - Import and use useSSEStream hook
    - Display connection state and received tokens
    - Accept session_id from URL query param
    - This route is ONLY for QA testing, not production UI

  **Must NOT do**:
  - No POST body (EventSource limitation - use session pattern)
  - No WebSocket (use SSE as designed)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Custom hook with reconnection logic, async state management
  - **Skills**: `[]`
    - No special skills needed for React hook development

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 13, 14, 15)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 2, 9 (needs real SSE endpoint from Task 9 for QA)

  **References**:
  - MDN EventSource: https://developer.mozilla.org/en-US/docs/Web/API/EventSource
  - Research findings: SSE must be in "use client" components
  - `api/routers/chat.py` - Real SSE endpoint from Task 9 (required for QA)

  **Acceptance Criteria**:
  - [ ] `web/src/hooks/useSSEStream.ts` exists
  - [ ] Hook is "use client" compatible
  - [ ] Reconnection with exponential backoff works
  - [ ] Cleanup on unmount (no memory leaks)
  - [ ] Test page `web/src/app/test-sse/page.tsx` exists for QA

  **QA Scenarios**:

  ```
  Scenario: Hook connects and receives events
    Tool: Playwright + Python subprocess
    Preconditions: 
      - .env configured with valid LLM_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_API_KEY
      - Index exists in data/index/ (required for chat to return tokens)
      - Next.js and FastAPI servers ready to start
    Steps:
      1. Verify index exists: `python -c "from pathlib import Path; assert Path('data/index/faiss').exists(), 'Need index for token streaming'"`
      2. Create test page at web/src/app/test-sse/page.tsx using useSSEStream hook
      3. Start FastAPI via Python subprocess:
         `python -c "import subprocess, sys; p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app', '--port', '8000']); print('PID:', p.pid)"`
      4. Start Next.js: `pnpm --prefix web dev` (in separate process)
      5. Create a chat session via POST /api/chat to get session_id
      6. Use Playwright to navigate to http://localhost:3000/test-sse?session_id=<id>
      7. Verify onToken callback fires and tokens appear on page
      8. Stop both servers via Python: `python -c "import os, signal; os.kill(<PID>, signal.SIGTERM)"`
    Expected Result: Events received and state updates visible on page
    Failure Indicators: No events, connection error, callback not firing
    Note: If no index/LLM, expect error event which should also be handled
    Evidence: .sisyphus/evidence/task-11-sse-hook.png

  Scenario: Hook reconnects on server restart
    Tool: Playwright + Python subprocess
    Preconditions: 
      - FastAPI and Next.js running
      - Hook connected to SSE (index not strictly required for reconnection test)
    Steps:
      1. Start FastAPI via Python subprocess, capture PID
      2. Establish SSE connection via test page
      3. Stop server via Python:
         `python -c "import os, signal; os.kill(<PID>, signal.SIGTERM)"`
      4. Verify hook enters "reconnecting" state (check UI indicator)
      5. Wait 2-3 seconds for backoff
      6. Restart server via Python subprocess (new PID)
      7. Wait up to 20 seconds, verify hook reconnects (status changes to "connected")
      8. Stop server
    Expected Result: Hook reconnects after server restart, backoff timing visible in UI/logs
    Failure Indicators: No reconnection attempt, immediate give-up, infinite reconnect loop
    Evidence: .sisyphus/evidence/task-11-reconnect.txt
  ```

  **Commit**: YES
  - Message: `feat(web): add useSSEStream hook with reconnection`
  - Files: `web/src/hooks/useSSEStream.ts`, `web/src/app/test-sse/page.tsx`

---

- [ ] 12. Chat page with streaming messages

  **What to do**:
  - Create `web/src/app/chat/page.tsx` (App Router)
  - Use Vercel AI SDK Elements for chat UI (NOT shadcn-chat)
  - Message list with scroll-area component
  - Input field with send button
  - Streaming message display (tokens appear as received)
  - Loading state while waiting for response
  - Error state display
  - Chinese UI text: "发送", "正在思考...", etc.

  **Must NOT do**:
  - No authentication UI
  - No message history persistence UI (just current session)
  - No export chat functionality

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: UI component layout, styling, visual feedback states
  - **Skills**: `["frontend-ui-ux"]`
    - For polished chat interface design

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 13, 14, 15)
  - **Blocks**: Tasks 16, 18
  - **Blocked By**: Tasks 2, 11

  **References**:
  - Vercel AI SDK: https://sdk.vercel.ai/docs/ai-sdk-ui/overview
  - `web/src/components/ui/scroll-area.tsx` - From shadcn (Task 2)
  - `web/src/hooks/useSSEStream.ts` - SSE hook (Task 11)

  **Acceptance Criteria**:
  - [ ] `/chat` route renders chat interface
  - [ ] Input field accepts text and has send button
  - [ ] Messages display in scrollable area
  - [ ] Streaming tokens appear progressively
  - [ ] UI text is in Chinese

  **QA Scenarios**:

  ```
  Scenario: Chat page renders with input field
    Tool: Playwright
    Preconditions: Next.js dev server running
    Steps:
      1. Navigate to http://localhost:3000/chat
      2. Wait for page load
      3. Find input element with placeholder
      4. Find send button
    Expected Result: Input visible with Chinese placeholder, button visible with "发送"
    Failure Indicators: 404, missing input, English text
    Evidence: .sisyphus/evidence/task-12-chat-render.png

  Scenario: Message appears in list after typing
    Tool: Playwright
    Preconditions: Chat page loaded
    Steps:
      1. Type "测试消息" in input
      2. Click send button
      3. Wait for message to appear in message list
    Expected Result: User message "测试消息" visible in chat history
    Failure Indicators: Message not added, input not cleared
    Evidence: .sisyphus/evidence/task-12-message-send.png
  ```

  **Commit**: YES
  - Message: `feat(web): add chat page with streaming messages`
  - Files: `web/src/app/chat/page.tsx`

---

- [ ] 13. Citation accordion component

  **What to do**:
  - Create `web/src/components/CitationAccordion.tsx`
  - Use shadcn/ui Accordion component as base
  - Display evidence sources with title, snippet, relevance score
  - Collapsible sections for each citation
  - Visual indicator for citation confidence (color/icon)
  - Chinese labels: "引用来源", "相关度", etc.
  - Create test route `web/src/app/test-citations/page.tsx` for isolated QA:
    - Import and render CitationAccordion with mock data
    - Mock data: 3 citations with varying relevance scores
    - This route is ONLY for QA testing, not production UI

  **Must NOT do**:
  - No linking to original files (just display)
  - No editing citations
  - No citation export

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Component design, visual hierarchy, styling
  - **Skills**: `["frontend-ui-ux"]`
    - For polished component design

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 14, 15)
  - **Blocks**: Task 16
  - **Blocked By**: Task 2

  **References**:
  - `web/src/components/ui/accordion.tsx` - Base shadcn component (Task 2)
  - `core/rag_answer.py` - Evidence/citation structure
  - `agent/states.py:GraphState` - How citations come from graph

  **Acceptance Criteria**:
  - [ ] `CitationAccordion.tsx` exists and exports component
  - [ ] Renders list of citations with accordion behavior
  - [ ] Shows title, snippet, relevance for each
  - [ ] Chinese labels throughout
  - [ ] Test route `web/src/app/test-citations/page.tsx` exists for QA

  **QA Scenarios**:

  ```
  Scenario: Accordion renders citations correctly
    Tool: Playwright
    Preconditions: Next.js dev server running
    Steps:
      1. Start Next.js via Python subprocess: `pnpm --prefix web dev`
      2. Navigate to http://localhost:3000/test-citations
      3. Wait for page load (selector: `[data-testid="citation-accordion"]`)
      4. Find all accordion items: `page.locator('[data-testid="citation-item"]')`
      5. Assert at least 3 items visible (mock data has 3)
      6. Click first accordion trigger: `page.locator('[data-testid="citation-item"]:first-child button').click()`
      7. Wait for content to expand
      8. Assert snippet text visible: `page.locator('[data-testid="citation-snippet"]').first().isVisible()`
      9. Assert Chinese label "引用来源" present in page
      10. Take screenshot
      11. Stop server
    Expected Result: Accordion expands, shows citation title + snippet + relevance score after click
    Failure Indicators: 404 on /test-citations, no accordion items, click doesn't expand, missing Chinese labels
    Evidence: .sisyphus/evidence/task-13-accordion.png

  Scenario: Multiple citations collapse/expand independently
    Tool: Playwright
    Preconditions: Next.js dev server running
    Steps:
      1. Navigate to http://localhost:3000/test-citations
      2. Click first accordion trigger to expand
      3. Verify first content visible
      4. Click second accordion trigger to expand
      5. Verify second content visible
      6. Verify first content STILL visible (both expanded)
      7. Click first accordion trigger again to collapse
      8. Verify first content hidden
      9. Verify second content still visible
      10. Take screenshot
    Expected Result: Independent expand/collapse behavior, multiple can be open
    Failure Indicators: Expanding one collapses others, only one expands at a time
    Evidence: .sisyphus/evidence/task-13-multi-accordion.png
  ```

  **Commit**: YES
  - Message: `feat(web): add citation accordion component`
  - Files: `web/src/components/CitationAccordion.tsx`, `web/src/app/test-citations/page.tsx`

---

- [ ] 14. Knowledge base builder page

  **What to do**:
  - Create `web/src/app/kb/page.tsx` (KB = Knowledge Base)
  - Form with ALL CorpusProfile fields (matching `core/corpus_profile.py`):
    - `name` (text input) - 知识库名称
    - `summary` (textarea) - 内容摘要
    - `coverage` (textarea) - 覆盖范围
    - `non_coverage` (textarea) - 不覆盖范围
    - `usage_notes` (textarea) - 使用说明
    - `source_examples` (tag input or textarea) - 代表性文件
    - `recommended_questions` (tag input or textarea) - 推荐提问
    - `forbidden_questions` (tag input or textarea) - 禁止/不建议问题
    - `domain_keywords` (tag input or textarea) - 领域关键词
    - `preferred_answer_style` (text input) - 偏好回答风格
    - `primary_entities` (tag input or textarea) - 核心实体
  - Save button that calls PUT /api/corpus-profile
  - Load existing profile on page load (GET /api/corpus-profile)
  - Success/error toast notifications
  - Chinese labels throughout

  **Must NOT do**:
  - No corpus profile deletion
  - No multiple knowledge bases
  - No import/export
  - No inventing fields not in core/corpus_profile.py (e.g., no "domain", "tone", "language")

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Form layout, input styling, toast notifications
  - **Skills**: `["frontend-ui-ux"]`
    - For polished form design

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 15)
  - **Blocks**: Tasks 15, 17, 18
  - **Blocked By**: Task 2

  **References**:
  - `api/models/corpus.py` - CorpusProfile fields (Task 4)
  - `core/corpus_profile.py` - All profile fields
  - `web/src/components/ui/input.tsx`, `textarea.tsx` - Form components

  **Acceptance Criteria**:
  - [ ] `/kb` route renders KB builder form
  - [ ] All CorpusProfile fields have inputs
  - [ ] Save button triggers API call
  - [ ] Existing profile loads on mount
  - [ ] Chinese labels throughout

  **QA Scenarios**:

  ```
  Scenario: KB page loads with form fields
    Tool: Playwright
    Preconditions: Next.js dev server running
    Steps:
      1. Navigate to http://localhost:3000/kb
      2. Find input for "知识库名称"
      3. Find textarea for "内容摘要"
      4. Find textarea for "覆盖范围"
      5. Find input for "偏好回答风格"
      6. Find save button
    Expected Result: All form fields visible with Chinese labels (11 fields total)
    Failure Indicators: Missing fields, English labels, wrong field types, 404
    Evidence: .sisyphus/evidence/task-14-kb-form.png

  Scenario: Form pre-fills with existing profile
    Tool: Playwright
    Preconditions: Corpus profile exists via API
    Steps:
      1. Create profile via PUT API with all fields
      2. Navigate to /kb
      3. Wait for load
      4. Check input values match saved profile
    Expected Result: Form fields contain existing profile data including lists
    Failure Indicators: Empty fields, wrong data, loading spinner stuck
    Evidence: .sisyphus/evidence/task-14-prefill.png
  ```

  **Commit**: YES
  - Message: `feat(web): add knowledge base builder page`
  - Files: `web/src/app/kb/page.tsx`

---

- [ ] 15. File upload component with drag-drop

  **What to do**:
  - Create `web/src/components/FileUpload.tsx`
  - Drag-and-drop zone for file upload
  - Click to browse files
  - Accept only .pdf, .md, .txt files
  - Show file name and size after selection
  - Upload button that calls POST /api/index
  - Progress indicator while uploading
  - Display job status (polling GET /api/jobs/{id})
  - Chinese text: "拖放文件到此处", "支持格式: PDF, Markdown, TXT", etc.
  - Create test route `web/src/app/test-upload/page.tsx` for isolated QA:
    - Import and render FileUpload component standalone
    - Mock API endpoint if needed, or use real backend
    - This route is ONLY for QA testing, not production UI

  **Must NOT do**:
  - No multiple file upload
  - No file preview
  - No file management/deletion

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Drag-drop interaction, progress UI, status display
  - **Skills**: `["frontend-ui-ux"]`
    - For polished upload UX

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12, 13, 14)
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 2, 14

  **References**:
  - HTML5 drag-and-drop API
  - `api/routers/indexing.py` - Upload endpoint (Task 10)
  - `api/models/indexing.py` - Job status response

  **Acceptance Criteria**:
  - [ ] `FileUpload.tsx` exists and exports component
  - [ ] Drag-drop zone visually indicates drop target
  - [ ] File type validation (.pdf, .md, .txt only)
  - [ ] Upload triggers API call and shows progress
  - [ ] Job status displayed after upload
  - [ ] Test route `web/src/app/test-upload/page.tsx` exists for QA

  **QA Scenarios**:

  ```
  Scenario: Drag-drop zone accepts valid file
    Tool: Playwright
    Preconditions: Next.js dev server running, FastAPI running
    Steps:
      1. Start both servers via Python subprocess
      2. Navigate to http://localhost:3000/test-upload
      3. Wait for drop zone: `page.locator('[data-testid="drop-zone"]')`
      4. Create test file: `const buffer = Buffer.from('Test content for upload')`
      5. Simulate file drop using Playwright's setInputFiles on hidden input
         or use DataTransfer API simulation
      6. Assert file name displayed: `page.locator('[data-testid="file-name"]').textContent()`
      7. Assert file size displayed: `page.locator('[data-testid="file-size"]')`
      8. Click upload button: `page.locator('[data-testid="upload-button"]').click()`
      9. Wait for job status: `page.locator('[data-testid="job-status"]')`
      10. Assert job status shows progress or completion
      11. Take screenshot
      12. Stop servers
    Expected Result: File accepted, name/size displayed, upload starts, job status visible
    Failure Indicators: Drop rejected, no file info shown, upload button disabled, no job status
    Evidence: .sisyphus/evidence/task-15-dragdrop.png

  Scenario: Invalid file type rejected
    Tool: Playwright
    Preconditions: Next.js dev server running
    Steps:
      1. Navigate to http://localhost:3000/test-upload
      2. Wait for drop zone
      3. Attempt to upload .exe file via hidden input
      4. Check for error message element: `page.locator('[data-testid="file-error"]')`
      5. Assert error contains Chinese text like "不支持的文件格式"
      6. Assert upload button NOT enabled or file not shown in file-name area
      7. Take screenshot
    Expected Result: File rejected with Chinese error "不支持的文件格式" or similar
    Failure Indicators: File accepted, no error message, English error text
    Evidence: .sisyphus/evidence/task-15-invalid-file.png
  ```

  **Commit**: YES
  - Message: `feat(web): add file upload component with drag-drop`
  - Files: `web/src/components/FileUpload.tsx`, `web/src/app/test-upload/page.tsx`

---

- [ ] 16. Connect chat page to FastAPI SSE stream

  **What to do**:
  - Update `web/src/app/chat/page.tsx` to integrate with backend
  - On send: POST to /api/chat, get session_id
  - Then connect useSSEStream to /api/chat/stream?session_id=<id>
  - Handle streaming tokens: append to current message
  - Handle citations: pass to CitationAccordion
  - Handle done: finalize message, enable input
  - Handle errors: show error toast, enable retry
  - Update CORS if needed for cross-origin SSE

  **Must NOT do**:
  - No WebSocket fallback
  - No message retry (just show error)
  - No offline support

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex integration, multiple async flows, error handling
  - **Skills**: `[]`
    - Requires careful state management and API coordination

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after all backend + frontend done)
  - **Blocks**: Final verification (F1-F4)
  - **Blocked By**: Tasks 9, 12, 13

  **References**:
  - `web/src/hooks/useSSEStream.ts` - SSE hook (Task 11)
  - `api/routers/chat.py` - Backend endpoints (Tasks 8, 9)
  - `web/src/components/CitationAccordion.tsx` - Citation display (Task 13)

  **Acceptance Criteria**:
  - [ ] Sending message creates session and starts stream
  - [ ] Tokens appear progressively in message bubble
  - [ ] Citations appear in accordion after message
  - [ ] Error states handled gracefully
  - [ ] Input re-enabled after response complete

  **QA Scenarios**:

  ```
  Scenario: End-to-end chat with streaming response
    Tool: Playwright
    Preconditions: 
      - .env file configured with valid LLM_MODEL, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_API_KEY
      - Index exists in data/index/ (created via earlier indexing or test fixture)
      - Both FastAPI (port 8000) and Next.js (port 3000) running
    Steps:
      1. Verify index exists: `python -c "from pathlib import Path; assert Path('data/index/faiss').exists()"`
      2. Start FastAPI and Next.js via Python subprocess
      3. Navigate to http://localhost:3000/chat
      4. Type "什么是RAG?" in input field
      5. Click send button
      6. Wait for streaming response (observe characters appearing progressively)
      7. Verify message appears in chat history
      8. Verify citations accordion appears (if response includes evidence)
      9. Take screenshot
      10. Stop servers
    Expected Result: Full response with Chinese content, citations visible if evidence found
    Failure Indicators: No response, connection error, missing citations when expected
    Note: If index is empty or LLM unavailable, expect graceful error message
    Evidence: .sisyphus/evidence/task-16-e2e-chat.png

  Scenario: Error handling on API failure
    Tool: Playwright
    Preconditions: FastAPI NOT running (simulate failure), Next.js running
    Steps:
      1. Start only Next.js (no FastAPI)
      2. Navigate to http://localhost:3000/chat
      3. Type "测试消息" and click send
      4. Wait for error state
      5. Verify user-friendly error message displayed
      6. Assert error text contains Chinese (e.g., "无法连接" or "服务器")
      7. Take screenshot
    Expected Result: User-friendly error in Chinese, not raw exception
    Failure Indicators: Cryptic error, unhandled exception, blank screen
    Evidence: .sisyphus/evidence/task-16-error-handling.png
  ```

  **Commit**: YES
  - Message: `feat(web): connect chat page to FastAPI SSE stream`
  - Files: `web/src/app/chat/page.tsx`

---

- [ ] 17. Connect KB builder to FastAPI endpoints

  **What to do**:
  - Update `web/src/app/kb/page.tsx` to integrate with backend
  - On mount: GET /api/corpus-profile, pre-fill form (handle 404)
  - On save: PUT /api/corpus-profile, show success/error toast
  - Integrate FileUpload component on same page
  - File upload triggers POST /api/index
  - Poll job status and display progress
  - Show success when indexing completes
  - Chinese toast messages: "保存成功", "索引完成", etc.

  **Must NOT do**:
  - No corpus profile deletion
  - No index deletion/rebuild
  - No file list management

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple API integrations, polling logic, form handling
  - **Skills**: `[]`
    - Standard React patterns, no special skills

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 18)
  - **Blocks**: Final verification (F1-F4)
  - **Blocked By**: Tasks 6, 10, 14, 15

  **References**:
  - `api/routers/corpus.py` - Profile endpoints (Task 6)
  - `api/routers/indexing.py` - Upload/job endpoints (Task 10)
  - `web/src/components/FileUpload.tsx` - Upload component (Task 15)

  **Acceptance Criteria**:
  - [ ] Form loads existing profile (or empty if none)
  - [ ] Save button calls PUT and shows toast
  - [ ] File upload section on same page
  - [ ] Indexing progress displayed
  - [ ] All UI in Chinese

  **QA Scenarios**:

  ```
  Scenario: Save corpus profile end-to-end
    Tool: Playwright
    Preconditions: 
      - FastAPI running (port 8000), Next.js running (port 3000)
      - .env configured (LLM keys not strictly required for profile save)
    Steps:
      1. Start both servers via Python subprocess
      2. Navigate to http://localhost:3000/kb
      3. Fill 知识库名称: "测试知识库"
      4. Fill 内容摘要: "这是测试摘要"
      5. Fill 覆盖范围: "测试覆盖范围"
      6. Click save button
      7. Verify success toast "保存成功" appears
      8. Refresh page (F5 or navigate away and back)
      9. Verify form pre-filled with saved values
      10. Take screenshot
      11. Stop servers
    Expected Result: Profile saved, toast shown, data persists across refresh
    Failure Indicators: Save fails, no toast, data lost on refresh
    Evidence: .sisyphus/evidence/task-17-save-profile.png

  Scenario: File indexing shows progress
    Tool: Playwright
    Preconditions: 
      - FastAPI running, Next.js running
      - .env configured with valid EMBEDDING_MODEL, EMBEDDING_API_KEY (required for indexing)
    Steps:
      1. Start both servers via Python subprocess
      2. Navigate to http://localhost:3000/kb
      3. Create a small test .txt file with content
      4. Upload via drag-drop or file picker
      5. Verify progress indicator displayed (e.g., "正在索引..." or spinner)
      6. Poll/wait for completion (up to 60s)
      7. Verify completion notification (e.g., "索引完成" toast)
      8. Take screenshot
      9. Stop servers
    Expected Result: Progress updates visible, completion notification shown
    Failure Indicators: No progress shown, stuck indefinitely, no completion toast
    Note: If embedding API unavailable, expect graceful error with Chinese message
    Evidence: .sisyphus/evidence/task-17-indexing.png
  ```

  **Commit**: YES
  - Message: `feat(web): connect KB builder to FastAPI endpoints`
  - Files: `web/src/app/kb/page.tsx`

---

- [ ] 18. Add Chinese UI text + error messages

  **What to do**:
  - Create `web/src/lib/i18n.ts` with all Chinese text constants
  - Audit all components for hardcoded English text
  - Replace with Chinese equivalents
  - Error messages: "无法连接到服务器", "请输入消息", "文件格式不支持"
  - UI labels: "发送", "保存", "知识库", "聊天", "引用来源"
  - Loading states: "正在加载...", "正在思考...", "正在索引..."
  - Success messages: "保存成功", "索引完成"
  - Add navigation header with Chinese text

  **Must NOT do**:
  - No i18n framework (just static Chinese)
  - No language switcher
  - No English fallback

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Text replacement, no complex logic
  - **Skills**: `[]`
    - Simple string updates

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 17)
  - **Blocks**: Final verification (F1-F4)
  - **Blocked By**: Tasks 12, 14

  **References**:
  - All component files in `web/src/`
  - `ui/gradio.py` - Chinese text examples for reference

  **Acceptance Criteria**:
  - [ ] `web/src/lib/i18n.ts` exports all text constants
  - [ ] No English text visible in UI (except technical terms)
  - [ ] All error messages in Chinese
  - [ ] Navigation shows Chinese labels

  **QA Scenarios**:

  ```
  Scenario: No unexpected English text visible in UI
    Tool: Playwright
    Preconditions: Full app running (FastAPI + Next.js)
    Steps:
      1. Start both servers via Python subprocess
      2. Use Playwright to navigate to http://localhost:3000/chat
      3. Extract all visible text: `const text = await page.evaluate(() => document.body.innerText)`
      4. Split into words and check each word
      5. Define allowlist: ["RAG", "API", "OK", "PDF", "Markdown", "TXT", "URL", "ID", "SSE", "KB"]
      6. Assert: all English words (ASCII a-z only) are in allowlist OR are technical terms
      7. Take screenshot for evidence
      8. Repeat for /kb page
      9. Stop servers
    Expected Result: All extracted text is Chinese or in allowlist, no unexpected English labels
    Failure Indicators: Found English words like "Send", "Save", "Loading" not in allowlist
    Evidence: .sisyphus/evidence/task-18-chinese-chat.png, task-18-chinese-kb.png, task-18-text-check.json

  Scenario: Error messages in Chinese
    Tool: Playwright
    Preconditions: FastAPI NOT running (to trigger connection error)
    Steps:
      1. Start only Next.js (no FastAPI)
      2. Navigate to /kb page
      3. Wait for error state
      4. Extract error message text
      5. Assert contains Chinese characters (e.g., "无法" or "错误" or "连接")
      6. Assert does NOT contain "Error:", "Failed:", "Cannot" (English error patterns)
      7. Take screenshot
    Expected Result: Error message is in Chinese like "无法连接到服务器"
    Failure Indicators: English error text, raw exception messages
    Evidence: .sisyphus/evidence/task-18-chinese-errors.png
  ```

  **Commit**: YES
  - Message: `feat(web): add Chinese UI text throughout`
  - Files: `web/src/lib/i18n.ts`, various component files

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`

  **What to do**:
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  
  **QA Scenarios**:
  
  ```
  Scenario: Verify all Must Have items present
    Tool: Bash
    Preconditions: All 18 tasks completed
    Steps:
      1. Check FastAPI exists: `python -c "import api.main"`
      2. Check Next.js built: `test -d web/.next`
      3. Check chat page: `test -f web/src/app/chat/page.tsx`
      4. Check KB page: `test -f web/src/app/kb/page.tsx`
      5. Check corpus profile endpoint: `curl -s http://localhost:8000/api/corpus-profile | python -c "import sys,json; json.load(sys.stdin)"`
      6. Check health endpoint: `curl -s http://localhost:8000/api/health | grep -q '"status"'`
      7. Check SSE hook: `test -f web/src/hooks/useSSEStream.ts`
      8. Check citation component: `test -f web/src/components/CitationAccordion.tsx`
      9. Check file upload component: `test -f web/src/components/FileUpload.tsx`
    Expected Result: All commands exit 0, all files exist
    Failure Indicators: Any command fails (non-zero exit), missing files
    Evidence: .sisyphus/evidence/f1-must-have-audit.txt
  
  Scenario: Verify Must NOT Have absent
    Tool: Bash
    Preconditions: All 18 tasks completed
    Steps:
      1. Check no auth code: `! grep -r "authentication\|login\|logout" web/src/ api/ --include="*.ts" --include="*.tsx" --include="*.py" | grep -v node_modules`
      2. Check no i18n framework: `! grep -r "react-i18next\|next-intl" web/package.json`
      3. Check no debug panel: `! test -f web/src/components/DebugPanel.tsx`
      4. Check no test files: `! find api/ -name "test_*.py" -o -name "*_test.py" | grep .`
      5. Check no English UI labels: run Task 18 QA scenario text extraction
    Expected Result: All negation commands succeed (patterns NOT found)
    Failure Indicators: Forbidden patterns found in codebase
    Evidence: .sisyphus/evidence/f1-must-not-audit.txt
  
  Scenario: Verify evidence files exist
    Tool: Bash
    Preconditions: All task QA scenarios ran
    Steps:
      1. Count evidence files: `ls -la .sisyphus/evidence/ | wc -l`
      2. Check task evidence pattern: `ls .sisyphus/evidence/task-*.png | wc -l`
      3. Assert at least 15 evidence files exist (roughly 1-2 per task)
    Expected Result: Evidence directory populated with PNG screenshots
    Failure Indicators: Empty evidence directory, missing screenshots
    Evidence: .sisyphus/evidence/f1-evidence-audit.txt
  ```
  
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Evidence [N files] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`

  **What to do**:
  Run `ruff check api/` + `pnpm --prefix web lint` + verify no TypeScript errors with `pnpm --prefix web build`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp). Note: No pytest for api/ since tests are out of scope per plan.
  
  **QA Scenarios**:
  
  ```
  Scenario: Python linting passes
    Tool: Bash
    Preconditions: api/ directory exists
    Steps:
      1. Run ruff: `python -m ruff check api/ --output-format=text`
      2. Capture exit code
      3. Assert exit code is 0
    Expected Result: `ruff check api/` exits 0, no errors
    Failure Indicators: Non-zero exit, error output
    Evidence: .sisyphus/evidence/f2-ruff.txt
  
  Scenario: TypeScript linting passes
    Tool: Bash
    Preconditions: web/ directory exists
    Steps:
      1. Run lint: `pnpm --prefix web lint`
      2. Capture exit code
      3. Assert exit code is 0
    Expected Result: `pnpm lint` exits 0, no ESLint errors
    Failure Indicators: Non-zero exit, lint errors
    Evidence: .sisyphus/evidence/f2-lint.txt
  
  Scenario: Next.js build succeeds
    Tool: Bash
    Preconditions: web/ directory exists
    Steps:
      1. Run build: `pnpm --prefix web build`
      2. Capture exit code
      3. Assert exit code is 0
      4. Verify .next directory created: `test -d web/.next`
    Expected Result: Build exits 0, .next directory populated
    Failure Indicators: Build failure, TypeScript errors, missing .next
    Evidence: .sisyphus/evidence/f2-build.txt
  
  Scenario: No anti-patterns in code
    Tool: Bash
    Preconditions: All code written
    Steps:
      1. Check no `as any`: `! grep -r "as any" web/src/ --include="*.ts" --include="*.tsx"`
      2. Check no `@ts-ignore`: `! grep -r "@ts-ignore" web/src/ --include="*.ts" --include="*.tsx"`
      3. Check no empty catches: `! grep -rP "catch\s*\(\w*\)\s*\{\s*\}" api/ web/src/ --include="*.py" --include="*.ts" --include="*.tsx"`
      4. Check no console.log: `! grep -r "console.log" web/src/ --include="*.ts" --include="*.tsx" | grep -v "// DEBUG"`
    Expected Result: All negation commands succeed
    Failure Indicators: Anti-patterns found
    Evidence: .sisyphus/evidence/f2-antipatterns.txt
  ```
  
  Output: `Ruff [PASS/FAIL] | Lint [PASS/FAIL] | Build [PASS/FAIL] | Anti-patterns [CLEAN/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill)

  **What to do**:
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (chat + citations, KB builder + indexing). Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  
  **QA Scenarios**:
  
  ```
  Scenario: Execute all task QA scenarios
    Tool: Playwright
    Preconditions: FastAPI running (port 8000), Next.js running (port 3000)
    Steps:
      1. Start both servers via Python subprocess
      2. For each task (1-18):
         a. Read task's QA scenarios from plan
         b. Execute each scenario using Playwright or Bash as specified
         c. Capture screenshot/output as evidence
         d. Record pass/fail
      3. Stop servers
    Expected Result: All scenarios pass (or have documented acceptable failures)
    Failure Indicators: Any scenario fails without acceptable reason
    Evidence: .sisyphus/evidence/final-qa/all-scenarios-summary.json
  
  Scenario: Cross-task integration - Chat with Citations
    Tool: Playwright
    Preconditions: 
      - FastAPI running, Next.js running
      - Index exists in data/index/
      - .env configured with LLM keys
    Steps:
      1. Navigate to http://localhost:3000/chat
      2. Send message: "这个知识库包含什么内容?"
      3. Wait for streaming response to complete
      4. Verify CitationAccordion appears below response
      5. Click to expand a citation
      6. Verify citation content matches response claims
      7. Take screenshot
    Expected Result: Chat response includes citations, accordion expands correctly
    Failure Indicators: No citations, accordion missing, citations don't match response
    Evidence: .sisyphus/evidence/final-qa/integration-chat-citations.png
  
  Scenario: Cross-task integration - KB Builder + Indexing
    Tool: Playwright
    Preconditions: FastAPI running, Next.js running, .env with embedding keys
    Steps:
      1. Navigate to http://localhost:3000/kb
      2. Fill in corpus profile fields
      3. Click save, verify success toast
      4. Upload a .txt file via drag-drop
      5. Wait for indexing to complete (poll job status)
      6. Verify completion toast
      7. Navigate to /chat
      8. Ask question about uploaded content
      9. Verify response references new content
      10. Take screenshot of full flow
    Expected Result: Full KB → Index → Chat flow works end-to-end
    Failure Indicators: Save fails, indexing stuck, new content not in responses
    Evidence: .sisyphus/evidence/final-qa/integration-kb-indexing.png
  
  Scenario: Edge cases
    Tool: Playwright
    Preconditions: Servers running
    Steps:
      1. Empty state: Load /chat with no index, verify graceful handling
      2. Invalid input: Send empty message, verify validation
      3. Rapid actions: Send 5 messages rapidly, verify no crashes
      4. File validation: Upload invalid file type, verify rejection
      5. Form validation: Leave required KB fields empty, verify error
    Expected Result: All edge cases handled gracefully with Chinese messages
    Failure Indicators: Crashes, blank screens, English errors, unhandled exceptions
    Evidence: .sisyphus/evidence/final-qa/edge-cases.png
  ```
  
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`

  **What to do**:
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  
  **Shared Files Allowlist** (expected to be modified by multiple tasks):
  - `api/main.py` — Router registration by Tasks 6, 8, 10
  - `web/src/app/layout.tsx` — Navigation updates by Tasks 12, 14
  - `web/package.json` — Dependencies added by multiple tasks
  - `api/requirements.txt` or `pyproject.toml` — Dependencies added by Task 1
  
  These shared files are EXEMPT from cross-task contamination rules.
  
  **QA Scenarios**:
  
  ```
  Scenario: Verify each task built exactly what was specified
    Tool: Bash
    Preconditions: All commits made
    Steps:
      1. Get list of all commits: `git log --oneline --since="start of work"`
      2. For each task (1-18):
         a. Find task's commit by message pattern
         b. Get files changed: `git show --name-only <commit>`
         c. Compare to task's "Files:" specification
         d. Flag extra files (scope creep)
         e. Flag missing files (incomplete)
      3. Generate report
    Expected Result: Each commit touches exactly its specified files, no more, no less
    Failure Indicators: Extra files in commit, missing files, wrong commit message
    Evidence: .sisyphus/evidence/f4-commit-audit.txt
  
  Scenario: Verify Must NOT do compliance
    Tool: Bash
    Preconditions: All code written
    Steps:
      1. For each task's "Must NOT do" items:
         a. Search codebase for forbidden patterns
         b. Report any violations with file:line
      2. Aggregate into compliance report
    Expected Result: No "Must NOT do" violations found
    Failure Indicators: Forbidden features implemented, scope creep
    Evidence: .sisyphus/evidence/f4-must-not-compliance.txt
  
  Scenario: Check for cross-task contamination
    Tool: Bash
    Preconditions: All commits made
    Steps:
      1. Build file → task ownership map from plan
      2. Define shared files allowlist: api/main.py, web/src/app/layout.tsx, web/package.json, pyproject.toml
      3. For each commit:
         a. Identify which task it belongs to
         b. Check if it touches files owned by other tasks
         c. If file is in allowlist, SKIP (not contamination)
         d. If file NOT in allowlist, flag: "Task N modified Task M's file X"
      4. Generate contamination report
    Expected Result: No cross-task contamination outside allowlist
    Failure Indicators: Task commits touching non-allowlisted files owned by other tasks
    Evidence: .sisyphus/evidence/f4-contamination.txt
  
  Scenario: Check for unaccounted changes
    Tool: Bash
    Preconditions: All commits made
    Steps:
      1. Get all changed files: `git diff --name-only origin/main`
      2. Build expected file list from all task specifications
      3. Compare actual vs expected
      4. Flag files changed but not in any task spec
      5. Flag expected files not actually changed
    Expected Result: All changes accounted for in plan, no surprise files
    Failure Indicators: Unplanned files changed, planned files missing
    Evidence: .sisyphus/evidence/f4-unaccounted.txt
  ```
  
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Task | Commit Message | Files |
|------|----------------|-------|
| 1 | `feat(api): add FastAPI skeleton with health endpoint and CORS` | api/ |
| 2 | `feat(web): init Next.js 14 with pnpm, Tailwind, shadcn/ui` | web/ |
| 3 | `feat(api): add SQLite session persistence module` | api/db/ |
| 4 | `feat(api): add Pydantic models for API contracts` | api/models/ |
| 5 | `feat(api): add graph caching with fingerprint invalidation` | api/services/ |
| 6 | `feat(api): add corpus profile GET/PUT endpoints` | api/routers/ |
| 7 | `feat(api): add background job queue for indexing` | api/jobs/ |
| 8 | `feat(api): add chat session creation endpoint` | api/routers/ |
| 9 | `feat(api): add SSE streaming endpoint with LangGraph` | api/routers/ |
| 10 | `feat(api): add file upload and indexing endpoint` | api/routers/ |
| 11 | `feat(web): add useSSEStream hook with reconnection` | web/src/hooks/useSSEStream.ts, web/src/app/test-sse/page.tsx |
| 12 | `feat(web): add chat page with streaming messages` | web/src/app/chat/page.tsx |
| 13 | `feat(web): add citation accordion component` | web/src/components/CitationAccordion.tsx, web/src/app/test-citations/page.tsx |
| 14 | `feat(web): add knowledge base builder page` | web/src/app/kb/page.tsx |
| 15 | `feat(web): add file upload component with drag-drop` | web/src/components/FileUpload.tsx, web/src/app/test-upload/page.tsx |
| 16 | `feat(web): connect chat page to FastAPI SSE stream` | web/src/app/chat/page.tsx |
| 17 | `feat(web): connect KB builder to FastAPI endpoints` | web/src/app/kb/page.tsx |
| 18 | `feat(web): add Chinese UI text throughout` | web/src/lib/i18n.ts, various component files |

---

## Success Criteria

### Verification Commands
```bash
# FastAPI health check
curl http://localhost:8000/api/health
# Expected: {"status": "ok"}

# Corpus profile read
curl http://localhost:8000/api/corpus-profile
# Expected: {"name": "...", "summary": "...", ...}

# Chat session creation
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message": "你好"}'
# Expected: {"session_id": "uuid-string"}

# SSE streaming (partial check)
curl -N "http://localhost:8000/api/chat/stream?session_id=test" -H "Accept: text/event-stream" --max-time 5
# Expected: event: token\ndata: ...\n\n (or error if no session)

# Next.js build
pnpm --prefix web build
# Expected: exit 0, no errors

# Next.js dev server
pnpm --prefix web dev &
curl http://localhost:3000
# Expected: HTML response with "知识库" or similar Chinese text
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] FastAPI serves on port 8000
- [ ] Next.js serves on port 3000
- [ ] Chat streaming works end-to-end
- [ ] KB builder saves profile and triggers indexing
- [ ] Citations display in chat responses
- [ ] All UI text in Chinese
