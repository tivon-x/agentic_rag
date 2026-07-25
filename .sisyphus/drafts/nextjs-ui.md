# Draft: Next.js UI for Agentic RAG

## Requirements (confirmed)

### Primary Goal
- **Separate Production Frontend**: Keep Gradio for dev/debugging, Next.js for user-facing production

### Architecture Decisions
- **Backend API**: FastAPI layer exposing REST/SSE endpoints
- **Deployment Model**: Monorepo (Next.js in `/web` or `/frontend` directory)
- **Communication**: SSE for streaming chat responses

### Feature Scope (must preserve)
- [x] Knowledge Base Builder - Define corpus profile, upload documents, trigger indexing
- [x] Chat Interface - Streaming Q&A with conversation history
- [x] Citation/Evidence Display - Structured citations and document tree hit visualization
- [ ] Debug Panel - EXCLUDED from MVP (Gradio serves this purpose)
- [ ] Index Statistics - EXCLUDED from MVP

### Technical Choices
- **Styling**: shadcn/ui + Tailwind CSS
- **State Management**: React Server Components + minimal client state
- **Language**: Chinese (Simplified) - matching current Gradio UI
- **Authentication**: None for MVP
- **Testing**: No tests for MVP, add later

## Scope Boundaries

### INCLUDE
- FastAPI backend layer with:
  - `/api/chat` - SSE streaming endpoint for Q&A
  - `/api/corpus-profile` - CRUD for knowledge base profile
  - `/api/index` - Trigger indexing, upload documents
  - `/api/status` - Health check, index status
- Next.js 14+ frontend with:
  - Chat page with streaming responses
  - Knowledge base builder page
  - Citation display in chat responses
  - Evidence/source visualization
- shadcn/ui components
- Tailwind CSS styling
- SSE for real-time streaming

### EXCLUDE
- Debug panel (keep in Gradio)
- Index statistics display
- Authentication system
- Automated tests
- i18n support (single language: Chinese)
- WebSocket (use SSE instead)
- Complex state management (Redux, etc.)

## Research Findings

### Library Research (from librarian agents)

**shadcn/ui Components to Use:**
- Core: `form`, `input`, `textarea`, `select`, `checkbox`, `accordion`, `card`, `button`, `scroll-area`, `toast`
- Chat: **Vercel AI SDK Elements** (recommended by shadcn-chat author, actively maintained)
- File Upload: `react-dropzone` + custom styling or `shadcn-file-upload`
- Forms: React Hook Form + Zod for validation

**Additional Decisions (from Metis review):**
- Indexing: Background Task + Polling (POST /api/index → 202 + job_id → GET /api/jobs/<id>)
- Session Persistence: SQLite (conversations survive server restarts)
- Error Messages: Chinese (matches UI)

**SSE Architecture Pattern:**
1. Use **EventSource** (not fetch) for standard SSE consumption
2. Two-step pattern for POST + SSE: `POST /chat` returns session_id → `GET /stream?session=<id>` for SSE
3. SSE consumption MUST be in Client Components ("use client")
4. Implement exponential backoff for reconnection (1s, 2s, 4s, 8s, 16s)
5. Auto-scroll chat messages with useRef + useEffect

**Next.js Architecture:**
- Server Components: Initial data fetching, layout, static content
- Client Components: Chat interface (EventSource), forms with real-time state
- Custom hook: `useSSEStream` for SSE handling with reconnection

### Current Gradio UI Analysis
From `ui/gradio.py`:
- **Tab 1: 知识库构建** (Knowledge Base Builder)
  - Corpus profile fields: name, summary, coverage, non_coverage, usage_notes, domain_keywords, primary_entities, recommended_questions, forbidden_questions, answer_style
  - Index mode selection: flat vs hierarchical
  - File upload for .pdf, .md, .txt
  - Status display and corpus profile preview
- **Tab 2: 智能问答** (Smart Q&A)
  - Chat interface with session management
  - Citation accordion
  - Debug panel accordion (route, query plan, rewritten queries, etc.)
  - Tree hits visualization

### GraphState Fields (for API response design)
From `agent/states.py`:
- `routingDecision`, `routingReason` - for debug
- `corpusProfile`, `corpusProfileData` - corpus context
- `queryPlan`, `rewrittenQuestions` - for debug
- `groundedAnswer` - main answer with citations
- `evidenceGroups` - structured evidence
- `packedContexts` - for debug

### Key Python Functions to Expose via API
From `core/factory.py`:
- `build_graph(settings)` - compile agent graph
- `build_retriever(settings)` - create retriever

From `indexing/indexer.py`:
- `Indexer(cfg).index(file_path)` - index documents

From `core/corpus_profile.py`:
- `load_corpus_profile(index_dir)` - read profile
- `save_corpus_profile(...)` - write profile
- `format_corpus_profile(profile)` - format for display

From `core/rag_answer.py`:
- `render_grounded_answer(payload)` - format answer
- `render_grounded_citations(payload)` - format citations

## Open Questions

- [RESOLVED] What features to include? → KB Builder, Chat, Citations
- [RESOLVED] Streaming approach? → SSE
- [RESOLVED] State management? → RSC + minimal client state

## Implementation Details

- **Frontend Directory**: `/web` - clean separation from Python backend
- **Package Manager**: pnpm
- **Dev Ports**: 
  - FastAPI backend: 8000
  - Next.js frontend: 3000
- **File Upload Flow**: Next.js → FastAPI (frontend receives files, forwards to backend)

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ KB Builder   │  │ Chat Page    │  │ Evidence Panel  │   │
│  │ (RSC)        │  │ (Client)     │  │ (Client)        │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ /corpus-     │  │ /chat (SSE)  │  │ /index          │   │
│  │  profile     │  │              │  │ /upload         │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              Existing Python RAG System                      │
│  agent/ │ core/ │ indexing/ │ llms/                         │
└─────────────────────────────────────────────────────────────┘
```
