"use client";

import { useEffect, useRef, useState } from "react";

import type { StreamToken } from "@/lib/types";

type Handlers = {
  onProgress?: (payload: StreamToken) => void;
  onEvidence?: (payload: StreamToken) => void;
  onFinal?: (payload: StreamToken) => void;
  onError?: (message: string) => void;
};

export function useSSEStream(handlers: Handlers) {
  const eventSourceRef = useRef<EventSource | null>(null);
  const handlersRef = useRef(handlers);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, []);

  function openStream(sessionId: string) {
    eventSourceRef.current?.close();

    const eventSource = new EventSource(
      `/api/chat/stream?session_id=${encodeURIComponent(sessionId)}`,
    );
    eventSourceRef.current = eventSource;
    setIsStreaming(true);

    eventSource.addEventListener("progress", (event) => {
      handlersRef.current.onProgress?.(
        JSON.parse((event as MessageEvent<string>).data) as StreamToken,
      );
    });
    eventSource.addEventListener("evidence", (event) => {
      handlersRef.current.onEvidence?.(
        JSON.parse((event as MessageEvent<string>).data) as StreamToken,
      );
    });
    eventSource.addEventListener("answer.final", (event) => {
      const payload = JSON.parse(
        (event as MessageEvent<string>).data,
      ) as StreamToken;
      handlersRef.current.onFinal?.(payload);
      setIsStreaming(false);
      eventSource.close();
      eventSourceRef.current = null;
    });
    eventSource.addEventListener("stream-error", (event) => {
      const payload = JSON.parse(
        (event as MessageEvent<string>).data,
      ) as StreamToken;
      handlersRef.current.onError?.(payload.error ?? "无法连接到服务器");
      setIsStreaming(false);
      eventSource.close();
      eventSourceRef.current = null;
    });
    eventSource.onerror = () => {
      handlersRef.current.onError?.("无法连接到服务器");
      setIsStreaming(false);
      eventSource.close();
      eventSourceRef.current = null;
    };
  }

  function closeStream() {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setIsStreaming(false);
  }

  return {
    isStreaming,
    openStream,
    closeStream,
  };
}
