"use client";

import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";

import { CitationAccordion } from "@/components/CitationAccordion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useSSEStream } from "@/hooks/useSSEStream";
import { createChatMessage, fetchChatSession } from "@/lib/api";
import { text } from "@/lib/i18n";
import type { ChatMessage as ChatMessageType, StreamToken } from "@/lib/types";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [draftReply, setDraftReply] = useState("");
  const draftReplyRef = useRef("");
  const deferredDraftReply = useDeferredValue(draftReply);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [citations, setCitations] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    draftReplyRef.current = draftReply;
  }, [draftReply]);

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
    onToken(payload: StreamToken) {
      startTransition(() => {
        setDraftReply((current) => current + (payload.content ?? ""));
      });
    },
    onCitations(payload: StreamToken) {
      setCitations(payload.citations_markdown ?? "");
    },
    onDone(payload: StreamToken) {
      startTransition(() => {
        const content = draftReplyRef.current || payload.content || "";
        if (content) {
          setMessages((current) => [
            ...current,
            {
              role: "assistant",
              content,
            },
          ]);
        }
        setDraftReply("");
      });
      setIsSubmitting(false);
    },
    onError(message: string) {
      setError(`${text.chat.errorPrefix}：${message}`);
      startTransition(() => {
        const content = draftReplyRef.current || message;
        if (content) {
          setMessages((current) => [
            ...current,
            {
              role: "assistant",
              content,
            },
          ]);
        }
        setDraftReply("");
      });
      setIsSubmitting(false);
    },
  });

  async function handleSubmit() {
    if (!input.trim()) {
      setError(text.chat.validation);
      return;
    }

    const userMessage = input.trim();
    setError("");
    setCitations("");
    setInput("");
    setIsSubmitting(true);
    startTransition(() => {
      setMessages((current) => [
        ...current,
        {
          role: "user",
          content: userMessage,
        },
      ]);
    });

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
      const message =
        caught instanceof Error ? caught.message : text.chat.connectError;
      setError(`${text.chat.errorPrefix}：${message}`);
      setIsSubmitting(false);
    }
  }

  function resetSession() {
    stream.closeStream();
    setMessages([]);
    setDraftReply("");
    setSessionId(null);
    setInput("");
    setError("");
    setCitations("");
    setIsSubmitting(false);
    window.history.replaceState(null, "", "/chat");
  }

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-10">
      <section className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <Card className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-2">
              <p className="status-pill">{text.nav.chat}</p>
              <h1 className="text-3xl font-semibold tracking-tight text-slate-950">
                {text.chat.title}
              </h1>
              <p className="max-w-2xl text-sm leading-7 text-slate-600">
                {text.chat.description}
              </p>
            </div>
            <Button variant="secondary" onClick={resetSession}>
              {text.chat.newSession}
            </Button>
          </div>

          <div className="space-y-4 rounded-[28px] border border-white/60 bg-white/80 p-4">
            {messages.length === 0 && !deferredDraftReply ? (
              <div className="rounded-[24px] bg-slate-900 px-5 py-6 text-slate-100">
                <p className="text-lg font-semibold">{text.chat.emptyTitle}</p>
                <p className="mt-2 text-sm leading-7 text-slate-300">
                  {text.chat.emptyBody}
                </p>
              </div>
            ) : null}

            {messages.map((message, index) => (
              <article
                key={`${message.role}-${index}-${message.content.slice(0, 24)}`}
                className={
                  message.role === "user"
                    ? "ml-auto max-w-[80%] rounded-[26px] bg-slate-950 px-5 py-4 text-sm leading-7 text-white"
                    : "max-w-[88%] rounded-[26px] border border-slate-200 bg-white px-5 py-4 text-sm leading-7 text-slate-800"
                }
              >
                {message.content}
              </article>
            ))}

            {deferredDraftReply ? (
              <article className="max-w-[88%] rounded-[26px] border border-emerald-200 bg-emerald-50 px-5 py-4 text-sm leading-7 text-slate-800">
                {deferredDraftReply}
              </article>
            ) : null}
          </div>

          <div className="space-y-3">
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={text.chat.inputPlaceholder}
            />
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm text-slate-600">
                {stream.isStreaming || isSubmitting
                  ? text.chat.sending
                  : sessionId
                    ? `会话 ID：${sessionId}`
                    : "新会话尚未创建"}
              </p>
              <Button disabled={stream.isStreaming || isSubmitting} onClick={handleSubmit}>
                {stream.isStreaming || isSubmitting ? text.chat.sending : text.chat.send}
              </Button>
            </div>
            {error ? <p className="text-sm text-rose-700">{error}</p> : null}
          </div>
        </Card>

        <div className="space-y-6">
          <CitationAccordion value={citations || "当前回答还没有引用内容。"} />
          <Card className="bg-slate-950 text-slate-100">
            <div className="space-y-3">
              <p className="text-sm font-semibold tracking-[0.2em] text-emerald-300 uppercase">
                SSE
              </p>
              <p className="text-sm leading-7 text-slate-300">
                当前前端通过 `/api/chat` 创建或追加消息，再用 `/api/chat/stream`
                订阅同一会话的流式输出。回答完成后，后端会再发一条 `citations`
                事件，用来填充右侧引用面板。
              </p>
            </div>
          </Card>
        </div>
      </section>
    </main>
  );
}
