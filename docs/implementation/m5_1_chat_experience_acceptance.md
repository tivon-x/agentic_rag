# M5.1 Chat experience acceptance

## Scope

M5.1 adds a durable chat-session list and an evidence-first chat shell. The
existing fixed retrieval stream remains the only answer path; no model, index,
or migration changes are included.

## API

- `GET /api/chat?limit=1..100&offset>=0` returns `{items,total,limit,offset}`.
- Each item contains `session_id`, `title`, `message_count`, `created_at`, and
  `updated_at`. `message_count` counts valid role/content message dictionaries
  and is zero for empty or corrupt JSON.
- Titles use the first non-empty user message after whitespace compression and
  are capped at 60 characters. Empty or invalid message JSON is shown as
  `未命名会话`.
- `POST /api/chat`, `GET /api/chat/{session_id}`, and
  `GET /api/chat/stream` remain compatible with the existing contract.

## Client state

- `session` in the URL is the selected session; changing it hydrates the
  session through the detail endpoint.
- A user message is optimistic only until `POST /api/chat` succeeds. A POST
  failure removes it. Once POST succeeds, an SSE failure retains the persisted
  user message and exposes a retry action.
- Answers render with `react-markdown` and `skipHtml`. Evidence is grouped by
  answer and is rendered only from the server payload; no sentence-level
  citations are inferred.
- The center display removes only a trailing explicit `\n## Evidence\n` block
  when structured evidence exists; persisted assistant content is unchanged.
- Desktop evidence rail is opt-in (248px session rail + flexible center by
  default, then a 352px rail when an answer with evidence is active). Tablet
  evidence uses an open-only dialog overlay. Mobile keeps each answer's
  evidence in collapsed native `details` disclosures and does not render the
  global evidence overlay.
- Enter submits the composer, Shift+Enter inserts a newline, and IME composing
  never submits. The transcript follows the bottom only while the reader is
  near the bottom; submit and final answer explicitly scroll there.

## Layout

- The root layout owns only document metadata, global CSS, and the skip link.
- The `(editorial)` route group owns the masthead/footer for `/`, `/library`,
  `/search`, `/papers/[id]`, and `/kb` without changing URLs.
- Chat is an independent `100dvh` shell: 248px session rail / flexible message
  column / 352px evidence rail at desktop; evidence overlay at tablet; session
  drawer and inline evidence on mobile.

## Verification

- Focused backend tests: `uv run pytest tests/test_chat_sessions.py tests/test_chat_evidence.py tests/test_streaming.py -q` - **7 passed**.
- Full backend suite: `uv run pytest -q` - **281 passed, 3 dependency warnings**.
- Ruff: `.venv\\Scripts\\ruff.exe check .` - **passed**.
- Frontend lint: `npm --prefix web run lint` - **passed**.
- Frontend production build: `npm --prefix web run build` - **passed**.
- Diff check: `git diff --check` - **passed** (only Git's LF/CRLF normalization warnings).
- Independent rendered-browser validation used isolated persisted sessions and
  the real FastAPI/Next.js applications:
  - 1440 x 900: session switching, answer numbering, Markdown rendering,
    answer-bound evidence rail, source metadata, and paper-page links passed.
  - 1024 x 768: evidence dialog opens without horizontal overflow and closes
    with Escape.
  - 375 x 812: session drawer, inline evidence disclosure, 40px minimum visible
    controls, fixed composer, and no horizontal overflow passed.
  - Reload preserved the selected session and its stored evidence. Browser
    console errors: **0**.

## Rollback

Revert the M5.1 source and lockfile changes. No database migration or data
rewrite is required; existing chat rows remain readable by the previous API.

## Files

- Backend: `api/models/chat.py`, `api/db/database.py`, `api/routers/chat.py`,
  `tests/test_chat_sessions.py`.
- Frontend: `web/package.json`, `web/pnpm-lock.yaml`,
  `web/src/components/ChatExperience.tsx`, `web/src/lib/api.ts`,
  `web/src/lib/types.ts`, `web/src/app/chat/page.tsx`, route-group layouts and
  pages, `web/src/app/globals.css`, `web/DESIGN.md`.
