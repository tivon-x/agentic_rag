import { Suspense } from "react";

import ChatExperience from "@/components/ChatExperience";

export default function ChatPage() {
  return (
    <Suspense fallback={<ChatLoading />}>
      <ChatExperience />
    </Suspense>
  );
}

function ChatLoading() {
  return (
    <main id="main-content" className="chat-loading" aria-busy="true">
      <p className="editorial-kicker">Chat / Fixed RAG</p>
      <h1>正在打开会话…</h1>
    </main>
  );
}
