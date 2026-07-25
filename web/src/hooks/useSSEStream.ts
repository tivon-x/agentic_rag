"use client";

import { useEffect, useRef, useState } from "react";

import type { StreamToken } from "@/lib/types";

type Handlers = {
  onToken?: (payload: StreamToken) => void;
  onCitations?: (payload: StreamToken) => void;
  onDone?: (payload: StreamToken) => void;
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

    eventSource.addEventListener("token", (event) => {
      handlersRef.current.onToken?.(
        JSON.parse((event as MessageEvent<string>).data) as StreamToken,
      );
    });
    eventSource.addEventListener("citations", (event) => {
      handlersRef.current.onCitations?.(
        JSON.parse((event as MessageEvent<string>).data) as StreamToken,
      );
    });
    eventSource.addEventListener("done", (event) => {
      const payload = JSON.parse(
        (event as MessageEvent<string>).data,
      ) as StreamToken;
      handlersRef.current.onDone?.(payload);
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
