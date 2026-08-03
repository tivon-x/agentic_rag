"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import { useSSEStream } from "@/hooks/useSSEStream";
import {
  createChatMessage,
  fetchChatSession,
  fetchChatSessions,
} from "@/lib/api";
import {
  evidenceHref,
  evidencePageLabel,
  evidenceSectionLabel,
} from "@/lib/evidence";
import type {
  ChatEvidence,
  ChatMessage,
  ChatSessionSummary,
  StreamToken,
} from "@/lib/types";

const EMPTY_TITLE = "未命名会话";

export default function ChatExperience() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedSession = searchParams.get("session");
  const [sessionId, setSessionId] = useState<string | null>(requestedSession);
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [listError, setListError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [canRetry, setCanRetry] = useState(false);
  const [activeAnswer, setActiveAnswer] = useState<number | null>(null);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const pendingEvidence = useRef<ChatEvidence[]>([]);
  const pendingUserMessage = useRef("");
  const persistedUser = useRef(false);
  const composing = useRef(false);
  const threadRef = useRef<HTMLElement | null>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 767px)");
    const update = () => {
      setIsMobile(media.matches);
    };
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    function closePanels(event: KeyboardEvent) {
      if (event.key !== "Escape") {
        return;
      }
      setSessionsOpen(false);
      setEvidenceOpen(false);
    }
    window.addEventListener("keydown", closePanels);
    return () => window.removeEventListener("keydown", closePanels);
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const response = await fetchChatSessions({ limit: 100 });
      setSessions(response.items);
      setListError("");
    } catch (caught) {
      setListError(resolveError(caught, "会话列表加载失败"));
    }
  }, []);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    let cancelled = false;
    if (!requestedSession) {
      setSessionId(null);
      setMessages([]);
      setActiveAnswer(null);
      setError("");
      return () => {
        cancelled = true;
      };
    }
    setSessionId(requestedSession);
    void fetchChatSession(requestedSession)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setMessages(payload.messages);
        const lastAnswer = findLastAnswer(payload.messages);
        setActiveAnswer(lastAnswer);
        setError("");
        pendingEvidence.current = [];
        stickToBottom.current = true;
        requestAnimationFrame(() => scrollToBottom(true));
      })
      .catch((caught) => {
        if (!cancelled) {
          setMessages([]);
          setError(resolveError(caught, "会话加载失败"));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [requestedSession]);

  const stream = useSSEStream({
    onEvidence(payload: StreamToken) {
      pendingEvidence.current = payload.evidence ?? [];
      const answerIndex = findLastAnswer(messages);
      if (answerIndex >= 0) {
        setActiveAnswer(answerIndex);
      }
    },
    onFinal(payload: StreamToken) {
      const content = payload.content?.trim() ?? "";
      if (content) {
        setMessages((current) => {
          const next = [
            ...current,
            {
              role: "assistant" as const,
              content,
              evidence: payload.evidence ?? pendingEvidence.current,
            },
          ];
          setActiveAnswer(next.length - 1);
          return next;
        });
      }
      pendingEvidence.current = [];
      pendingUserMessage.current = "";
      persistedUser.current = false;
      setCanRetry(false);
      setIsSubmitting(false);
      stickToBottom.current = true;
      requestAnimationFrame(() => scrollToBottom(true));
      void loadSessions();
    },
    onError(message: string) {
      pendingEvidence.current = [];
      setError(`${message}。已保留本轮提问，可以重试。`);
      setCanRetry(true);
      setIsSubmitting(false);
    },
  });

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const userMessage = input.trim();
    if (!userMessage || isBusy) {
      if (!userMessage) {
        setError("请输入问题。");
      }
      return;
    }

    setError("");
    setInput("");
    setIsSubmitting(true);
    pendingEvidence.current = [];
    pendingUserMessage.current = userMessage;
    persistedUser.current = false;
    setCanRetry(false);
    stickToBottom.current = true;
    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: userMessage,
      },
    ]);
    requestAnimationFrame(() => scrollToBottom(true));

    try {
      const response = await createChatMessage({
        message: userMessage,
        sessionId,
      });
      persistedUser.current = true;
      setCanRetry(false);
      setSessionId(response.session_id);
      router.replace(`/chat?session=${encodeURIComponent(response.session_id)}`);
      await loadSessions();
      stream.openStream(response.session_id);
    } catch (caught) {
      persistedUser.current = false;
      setCanRetry(false);
      removeOptimisticUser(userMessage);
      pendingUserMessage.current = "";
      setError(`${resolveError(caught, "提问提交失败")}，请重试。`);
      setIsSubmitting(false);
    }
  }

  function removeOptimisticUser(content: string) {
    setMessages((current) => {
      const index = [...current]
        .map((message) => message.role === "user" && message.content === content)
        .lastIndexOf(true);
      return index >= 0 ? current.toSpliced(index, 1) : current;
    });
  }

  function retryStream() {
    if (!sessionId || !persistedUser.current || isBusy) {
      return;
    }
    setError("");
    setCanRetry(false);
    setIsSubmitting(true);
    pendingUserMessage.current = "";
    stream.openStream(sessionId);
  }

  function scrollToBottom(force = false) {
    const element = threadRef.current;
    if (!element || (!force && !stickToBottom.current)) {
      return;
    }
    element.scrollTop = element.scrollHeight;
  }

  function handleThreadScroll() {
    const element = threadRef.current;
    if (!element) {
      return;
    }
    stickToBottom.current =
      element.scrollHeight - element.scrollTop - element.clientHeight < 120;
  }

  function handleComposerKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing &&
      !composing.current
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function selectSession(nextSessionId: string) {
    stream.closeStream();
    setSessionsOpen(false);
    setEvidenceOpen(false);
    setError("");
    router.push(`/chat?session=${encodeURIComponent(nextSessionId)}`);
  }

  function createNewSession() {
    stream.closeStream();
    setMessages([]);
    setSessionId(null);
    setInput("");
    setError("");
    setActiveAnswer(null);
    pendingEvidence.current = [];
    pendingUserMessage.current = "";
    persistedUser.current = false;
    setCanRetry(false);
    setIsSubmitting(false);
    setSessionsOpen(false);
    setEvidenceOpen(false);
    router.push("/chat");
  }

  const messageRows = useMemo(() => {
    let assistantOrdinal = 0;
    return messages.map((message, index) => {
      const answerNumber =
        message.role === "assistant" ? ++assistantOrdinal : null;
      return { message, index, answerNumber };
    });
  }, [messages]);
  const selectedAnswer =
    activeAnswer === null ? null : messages[activeAnswer] ?? null;
  const selectedEvidence =
    selectedAnswer?.role === "assistant" ? selectedAnswer.evidence ?? [] : [];
  const hasSelectedEvidence =
    !isMobile && evidenceOpen && selectedEvidence.length > 0;
  const isBusy = stream.isStreaming || isSubmitting;

  return (
    <div className="chat-shell">
      <header className="chat-topbar">
        <button
          type="button"
          className="chat-mobile-trigger"
          aria-expanded={sessionsOpen}
          aria-controls="chat-session-drawer"
          onClick={() => setSessionsOpen((value) => !value)}
        >
          会话
        </button>
        <div className="chat-topbar-brand">
          <span className="brand-name">Paper Index</span>
          <span className="brand-caption">CHAT / FIXED RAG</span>
        </div>
        <div className="chat-topbar-actions">
          <span className="chat-baseline">固定基线 · v1_flat_rerank</span>
          {!isMobile && selectedEvidence.length > 0 ? (
            <button
              type="button"
              className="chat-evidence-trigger"
              aria-expanded={evidenceOpen}
              onClick={() => setEvidenceOpen((value) => !value)}
            >
              证据 {selectedEvidence.length}
            </button>
          ) : null}
        </div>
      </header>

      <div className={hasSelectedEvidence ? "chat-workspace has-evidence" : "chat-workspace"}>
        <SessionSidebar
          sessions={sessions}
          activeSessionId={sessionId}
          listError={listError}
          onSelect={selectSession}
          onCreate={createNewSession}
          className="chat-sessions-desktop"
        />

        <main id="main-content" className="chat-main">
          <div className="chat-mobile-context">
            <p className="editorial-kicker">Evidence-first conversation</p>
            <h1>和论文对话</h1>
            <p>每条回答只展示已有的结构化证据，不伪造句级引用。</p>
          </div>

          <section
            ref={threadRef}
            className="chat-thread"
            aria-live="polite"
            aria-label="会话消息"
            onScroll={handleThreadScroll}
          >
            {messages.length === 0 ? (
              <div className="chat-empty-state">
                <span className="evidence-marker" aria-hidden="true" />
                <p className="chat-empty-title">从一个论文问题开始</p>
                <p>例如询问方法定义、实验结果或论文限制。回答完成后，证据会跟随对应回答出现。</p>
              </div>
            ) : null}
            {messageRows.map(({ message, index, answerNumber }) => (
              <MessageRow
                key={`${message.role}-${index}`}
                message={message}
                answerNumber={answerNumber}
                active={activeAnswer === index}
                onEvidence={() => {
                  setActiveAnswer(index);
                  if (!isMobile && message.evidence?.length) {
                    setEvidenceOpen(true);
                  }
                }}
              />
            ))}
            {error ? (
              <div className="chat-error" role="alert">
                <span>{error}</span>
                {canRetry && sessionId ? (
                  <button type="button" onClick={retryStream}>
                    重试回答
                  </button>
                ) : null}
              </div>
            ) : null}
          </section>

          <form className="chat-composer-fixed" onSubmit={handleSubmit}>
            <label htmlFor="chat-input" className="editorial-kicker">
              新问题
            </label>
            <div className="chat-composer-row">
              <textarea
                id="chat-input"
                value={input}
                disabled={isBusy}
                placeholder="问论文中的方法、证据或限制…"
                rows={2}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                onCompositionStart={() => {
                  composing.current = true;
                }}
                onCompositionEnd={() => {
                  composing.current = false;
                }}
              />
              <button type="submit" disabled={isBusy || !input.trim()}>
                {isBusy ? "回答中" : "发送"}
              </button>
            </div>
            <p className="chat-composer-meta">
              {sessionId ? `会话 ${sessionId.slice(0, 12)}` : "新会话尚未创建"}
            </p>
          </form>
        </main>

        {hasSelectedEvidence ? (
          <EvidencePanel
            answer={selectedAnswer}
            answerNumber={
              messageRows.find((row) => row.index === activeAnswer)?.answerNumber ?? 0
            }
            className="chat-evidence-desktop"
          />
        ) : null}
      </div>

      {sessionsOpen ? (
        <>
          <div
            className="chat-mobile-drawer-backdrop is-open"
            aria-hidden="true"
            onClick={() => setSessionsOpen(false)}
          />
          <aside
            id="chat-session-drawer"
            className="chat-session-drawer is-open"
            role="dialog"
            aria-modal="true"
            aria-label="会话列表"
          >
            <SessionSidebar
              sessions={sessions}
              activeSessionId={sessionId}
              listError={listError}
              onSelect={selectSession}
              onCreate={createNewSession}
              onClose={() => setSessionsOpen(false)}
            />
          </aside>
        </>
      ) : null}
      {!isMobile && evidenceOpen && selectedEvidence.length > 0 ? (
        <div
          className="chat-evidence-overlay is-open"
          role="dialog"
          aria-modal="true"
          aria-label="当前回答证据"
        >
          <EvidencePanel
            answer={selectedAnswer}
            answerNumber={
              messageRows.find((row) => row.index === activeAnswer)?.answerNumber ?? 0
            }
            onClose={() => setEvidenceOpen(false)}
          />
        </div>
      ) : null}
    </div>
  );
}

function SessionSidebar({
  sessions,
  activeSessionId,
  listError,
  onSelect,
  onCreate,
  onClose,
  className = "",
}: {
  sessions: ChatSessionSummary[];
  activeSessionId: string | null;
  listError: string;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
  onClose?: () => void;
  className?: string;
}) {
  return (
    <aside className={`chat-sessions ${className}`} aria-label="会话列表">
      <div className="chat-sessions-header">
        <div>
          <p className="editorial-kicker">Sessions</p>
          <h2>会话</h2>
        </div>
        <div className="chat-sessions-actions">
          {onClose ? (
            <button type="button" className="chat-panel-close" onClick={onClose}>
              关闭
            </button>
          ) : null}
          <button type="button" onClick={onCreate} aria-label="新建会话">
            ＋
          </button>
        </div>
      </div>
      <nav className="chat-sidebar-links" aria-label="论文工作区">
        <Link href="/library">论文目录</Link>
        <Link href="/search">搜索证据</Link>
      </nav>
      {listError ? <p className="chat-sidebar-error">{listError}</p> : null}
      <ol className="chat-session-list">
        {sessions.length === 0 ? (
          <li className="chat-session-empty">还没有已保存会话</li>
        ) : (
          sessions.map((session) => (
            <li key={session.session_id}>
              <button
                type="button"
                className={
                  session.session_id === activeSessionId
                    ? "chat-session-item is-active"
                    : "chat-session-item"
                }
                onClick={() => onSelect(session.session_id)}
              >
                <span>{session.title || EMPTY_TITLE}</span>
                <time dateTime={session.updated_at}>
                  {formatSessionDate(session.updated_at)}
                </time>
              </button>
            </li>
          ))
        )}
      </ol>
    </aside>
  );
}

function MessageRow({
  message,
  answerNumber,
  active,
  onEvidence,
}: {
  message: ChatMessage;
  answerNumber: number | null;
  active: boolean;
  onEvidence: () => void;
}) {
  if (message.role === "system") {
    return null;
  }
  const isUser = message.role === "user";
  const evidenceCount = message.evidence?.length ?? 0;
  return (
    <article className={isUser ? "chat-message chat-message-user" : "chat-message"}>
      <div className="chat-message-meta">
        <span>
          {isUser ? "提问" : `回答 ${String(answerNumber ?? 0).padStart(2, "0")}`}
        </span>
        {!isUser ? (
          <button
            type="button"
            className={active ? "chat-answer-evidence is-active" : "chat-answer-evidence"}
            onClick={onEvidence}
            disabled={!evidenceCount}
          >
            {evidenceCount ? `证据 ${evidenceCount}` : "无结构化证据"}
          </button>
        ) : null}
      </div>
      {isUser ? (
        <p className="chat-user-copy">{message.content}</p>
      ) : (
        <div className="chat-markdown">
          <ReactMarkdown
            skipHtml
            components={{
              a: ({ href, children, ...props }) => {
                const external = href?.startsWith("http://") || href?.startsWith("https://");
                return (
                  <a
                    {...props}
                    href={href}
                    target={external ? "_blank" : undefined}
                    rel={external ? "noreferrer" : undefined}
                  >
                    {children}
                  </a>
                );
              },
            }}
          >
            {formatAssistantDisplayContent(message.content, evidenceCount > 0)}
          </ReactMarkdown>
        </div>
      )}
      {!isUser && evidenceCount > 0 ? (
        <div className="chat-mobile-evidence">
          <p className="chat-mobile-evidence-label">本条回答的 {evidenceCount} 条证据</p>
          <ol className="chat-mobile-evidence-list">
            {message.evidence?.map((item, evidenceIndex) => (
              <MobileEvidenceItem
                key={`${item.node_id}-${evidenceIndex}`}
                item={item}
                number={evidenceIndex + 1}
              />
            ))}
          </ol>
        </div>
      ) : null}
    </article>
  );
}

function EvidencePanel({
  answer,
  answerNumber,
  className = "",
  onClose,
}: {
  answer: ChatMessage | null;
  answerNumber: number;
  className?: string;
  onClose?: () => void;
}) {
  const evidence = answer?.evidence ?? [];
  return (
    <aside className={`chat-evidence-panel ${className}`} aria-label="当前回答证据">
      <div className="chat-evidence-heading">
        <div>
          <p className="editorial-kicker">Evidence rail</p>
          <h2>{answerNumber > 0 ? `回答 ${String(answerNumber).padStart(2, "0")}` : "回答证据"}</h2>
        </div>
        {onClose ? (
          <button type="button" className="chat-panel-close" onClick={onClose}>
            关闭
          </button>
        ) : null}
      </div>
      {evidence.length === 0 ? (
        <div className="chat-evidence-empty">
          <span className="evidence-marker" aria-hidden="true" />
          <p>选择一条有结构化证据的回答。</p>
          <small>没有证据时不会补写引用。</small>
        </div>
      ) : (
        <ol className="chat-evidence-list">
          {evidence.map((item, index) => (
            <EvidenceItem key={`${item.node_id}-${index}`} item={item} number={index + 1} />
          ))}
        </ol>
      )}
    </aside>
  );
}

function EvidenceItem({ item, number }: { item: ChatEvidence; number: number }) {
  const href = evidenceHref(item);
  const label = item.paper_title || item.source;
  const preview =
    item.quote.length > 180 ? `${item.quote.slice(0, 180).trimEnd()}…` : item.quote;
  return (
    <li className="chat-evidence-item">
      <div className="chat-evidence-item-head">
        <span className="evidence-number" aria-hidden="true">
          {String(number).padStart(2, "0")}
        </span>
        <div>
          <p>{label}</p>
          <span>{evidencePageLabel(item)}</span>
        </div>
      </div>
      <p className="chat-evidence-section">{evidenceSectionLabel(item)}</p>
      <blockquote>{preview}</blockquote>
      {item.quote.length > 180 ? (
        <details className="chat-evidence-disclosure">
          <summary>查看完整原文摘录</summary>
          <blockquote>{item.quote}</blockquote>
        </details>
      ) : null}
      {item.relevance ? <p className="chat-evidence-relevance">关联说明：{item.relevance}</p> : null}
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          aria-label={`在新标签打开 ${label} ${evidencePageLabel(item)}`}
          className="text-link"
        >
          打开论文原页 ↗
        </a>
      ) : (
        <span className="chat-evidence-unavailable">该来源暂无论文目录链接</span>
      )}
    </li>
  );
}

function MobileEvidenceItem({ item, number }: { item: ChatEvidence; number: number }) {
  const href = evidenceHref(item);
  const label = item.paper_title || item.source;
  return (
    <li className="chat-mobile-evidence-item">
      <details>
        <summary>
          <span className="evidence-number" aria-hidden="true">
            {String(number).padStart(2, "0")}
          </span>
          <span>
            <strong>{label}</strong>
            <em>{evidencePageLabel(item)}</em>
          </span>
        </summary>
        <p className="chat-evidence-section">{evidenceSectionLabel(item)}</p>
        <blockquote>{item.quote}</blockquote>
        {item.relevance ? (
          <p className="chat-evidence-relevance">关联说明：{item.relevance}</p>
        ) : null}
        {href ? (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            aria-label={`在新标签打开 ${label} ${evidencePageLabel(item)}`}
            className="text-link"
          >
            打开论文原页 ↗
          </a>
        ) : null}
      </details>
    </li>
  );
}

export function formatAssistantDisplayContent(content: string, hasEvidence: boolean) {
  if (!hasEvidence) {
    return content;
  }
  const marker = "\n## Evidence\n";
  const markerIndex = content.lastIndexOf(marker);
  return markerIndex >= 0 ? content.slice(0, markerIndex).trimEnd() : content;
}

function findLastAnswer(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === "assistant") {
      return index;
    }
  }
  return -1;
}

function formatSessionDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
  }).format(date);
}

function resolveError(caught: unknown, fallback: string) {
  return caught instanceof Error && caught.message ? caught.message : fallback;
}
