"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

import { EvidenceRail } from "@/components/CitationAccordion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useSSEStream } from "@/hooks/useSSEStream";
import { createChatMessage, fetchChatSession } from "@/lib/api";
import { text } from "@/lib/i18n";
import type { ChatEvidence, ChatMessage, StreamToken } from "@/lib/types";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const pendingEvidence = useRef<ChatEvidence[]>([]);
  const pendingUserMessage = useRef("");

  useEffect(() => {
    let cancelled = false;
    const requestedSession = new URLSearchParams(window.location.search).get(
      "session",
    );
    if (!requestedSession || requestedSession === sessionId) {
      return () => {
        cancelled = true;
      };
    }
    const requestedSessionId = requestedSession;

    async function hydrateSession() {
      try {
        const payload = await fetchChatSession(requestedSessionId);
        if (cancelled) {
          return;
        }
        setSessionId(payload.session_id);
        setMessages(payload.messages);
        setError("");
        pendingEvidence.current = [];
      } catch (caught) {
        if (cancelled) {
          return;
        }
        const message =
          caught instanceof Error ? caught.message : text.chat.connectError;
        setError(`${text.chat.errorPrefix}：${message}`);
      }
    }

    void hydrateSession();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const stream = useSSEStream({
    onEvidence(payload: StreamToken) {
      pendingEvidence.current = payload.evidence ?? [];
    },
    onFinal(payload: StreamToken) {
      const content = payload.content || "";
      if (content) {
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content,
            evidence: payload.evidence ?? pendingEvidence.current,
          },
        ]);
      }
      pendingEvidence.current = [];
      pendingUserMessage.current = "";
      setIsSubmitting(false);
    },
    onError(message: string) {
      removeOptimisticUser(pendingUserMessage.current);
      pendingEvidence.current = [];
      pendingUserMessage.current = "";
      setError(`${text.chat.errorPrefix}：${message}`);
      setIsSubmitting(false);
    },
  });

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!input.trim()) {
      setError(text.chat.validation);
      return;
    }

    const userMessage = input.trim();
    setError("");
    setInput("");
    setIsSubmitting(true);
    pendingEvidence.current = [];
    pendingUserMessage.current = userMessage;
    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    try {
      const response = await createChatMessage({
        message: userMessage,
        sessionId,
      });
      setSessionId(response.session_id);
      window.history.replaceState(
        null,
        "",
        `/chat?session=${encodeURIComponent(response.session_id)}`,
      );
      stream.openStream(response.session_id);
    } catch (caught) {
      removeOptimisticUser(userMessage);
      pendingUserMessage.current = "";
      const message =
        caught instanceof Error ? caught.message : text.chat.connectError;
      setError(`${text.chat.errorPrefix}：${message}`);
      setIsSubmitting(false);
    }
  }

  function removeOptimisticUser(content: string) {
    if (!content) {
      return;
    }
    setMessages((current) => {
      const index = [...current]
        .map((message) => message.role === "user" && message.content === content)
        .lastIndexOf(true);
      return index >= 0 ? current.toSpliced(index, 1) : current;
    });
  }

  function resetSession() {
    stream.closeStream();
    setMessages([]);
    setSessionId(null);
    setInput("");
    setError("");
    pendingEvidence.current = [];
    pendingUserMessage.current = "";
    setIsSubmitting(false);
    window.history.replaceState(null, "", "/chat");
  }

  const isBusy = stream.isStreaming || isSubmitting;

  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col px-5 py-9 sm:px-8 sm:py-14"
    >
      <header className="chat-header">
        <div>
          <p className="editorial-kicker">Chat / Fixed RAG</p>
          <h1 className="page-title mt-4">和论文对话</h1>
          <p className="page-description mt-5">
            每轮回答都保留自己的证据。打开右侧来源，回到论文章节和原始页码核验。
          </p>
        </div>
        <div className="flex flex-col items-start gap-3 sm:items-end">
          <span className="status-marker">
            <span className="status-dot" aria-hidden="true" />
            固定检索基线
          </span>
          <Button variant="secondary" type="button" onClick={resetSession}>
            {text.chat.newSession}
          </Button>
        </div>
      </header>

      <div className="chat-layout">
        <section className="chat-column" aria-labelledby="chat-transcript-title">
          <div className="flex items-baseline justify-between gap-4 border-b border-[var(--line)] pb-3">
            <h2 id="chat-transcript-title" className="font-serif text-2xl">
              会话记录
            </h2>
            <span className="font-mono text-xs text-[var(--muted-ink)]">
              {messages.length} 条消息
            </span>
          </div>

          <div className="chat-transcript" aria-live="polite">
            {messages.length === 0 ? (
              <div className="chat-empty">
                <span className="evidence-marker" aria-hidden="true" />
                <p className="font-serif text-2xl">先问一个论文问题</p>
                <p className="mt-3 max-w-lg text-sm leading-7 text-[var(--muted-ink)]">
                  例如询问某种方法的定义、实验结果或限制。回答完成后，右侧会出现可回到原页的证据。
                </p>
              </div>
            ) : null}

            {messages.map((message, index) => (
              <ChatMessageRow key={`${message.role}-${index}`} message={message} />
            ))}
          </div>

          <form className="chat-composer" onSubmit={handleSubmit}>
            <label htmlFor="chat-input" className="editorial-kicker">
              新问题
            </label>
            <Textarea
              id="chat-input"
              value={input}
              disabled={isBusy}
              placeholder={text.chat.inputPlaceholder}
              onChange={(event) => setInput(event.target.value)}
            />
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs leading-6 text-[var(--muted-ink)]">
                {isBusy
                  ? text.chat.sending
                  : sessionId
                    ? `会话 ${sessionId.slice(0, 12)}`
                    : "新会话尚未创建"}
              </p>
              <Button disabled={isBusy} type="submit">
                {isBusy ? text.chat.sending : text.chat.send}
              </Button>
            </div>
            {error ? (
              <p role="alert" className="form-error">
                {error}
              </p>
            ) : null}
          </form>
        </section>

        <EvidenceRail messages={messages} />
      </div>
    </main>
  );
}

function ChatMessageRow({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const evidenceCount = message.evidence?.length ?? 0;

  return (
    <article className={isUser ? "chat-message chat-message-user" : "chat-message"}>
      <div className="flex items-center justify-between gap-3 border-b border-[var(--line)] pb-2">
        <span className="chat-role-label font-mono text-xs uppercase tracking-[0.14em]">
          {isUser ? "提问" : "回答"}
        </span>
        {!isUser ? (
          <span className="evidence-count">
            <span className="status-dot" aria-hidden="true" />
            {evidenceCount ? `${evidenceCount} 条证据` : "无结构化证据"}
          </span>
        ) : null}
      </div>
      <p className="prose-block mt-4 text-sm leading-8">{message.content}</p>
    </article>
  );
}
