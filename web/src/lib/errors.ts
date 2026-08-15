const KNOWN_ERROR_MESSAGES: Array<[RegExp, string]> = [
  [
    /no (?:active )?index(?: loaded| is available)?|index .*not loaded|没有.*索引/i,
    "当前还没有可搜索的论文索引，请先前往论文库导入并解析文件。",
  ],
  [
    /paper not found|论文.*不存在|找不到.*论文/i,
    "找不到这篇论文，请返回论文库重新选择。",
  ],
  [
    /chat session not found|session not found|找不到.*会话/i,
    "找不到这个会话，请返回聊天页开始新会话。",
  ],
  [
    /retrieval.*(?:failed|unavailable)|检索失败|unable to retrieve/i,
    "检索论文失败，请稍后重试。",
  ],
  [
    /save.*failed|persist|保存失败|回答保存失败/i,
    "回答保存失败，回答没有保存，请重试。",
  ],
  [
    /answer.*(generation|graph).*failed|no answers were generated|回答生成失败/i,
    "回答生成失败，回答没有保存，请重试。",
  ],
  [
    /network|failed to fetch|无法连接|连接到服务器/i,
    "暂时无法连接服务，请检查网络后重试。",
  ],
];

const DEFAULT_ERROR = "操作未完成，请稍后重试。";

function trimTerminalPunctuation(value: string): string {
  return value.trim().replace(/[。！？；：.!?;:]+$/u, "").trim();
}

/** Map backend/provider details to stable Chinese recovery copy for the UI. */
export function toUserError(caught: unknown, fallback = DEFAULT_ERROR): string {
  const raw =
    typeof caught === "string"
      ? caught
      : caught instanceof Error
        ? caught.message
        : "";
  const normalized = trimTerminalPunctuation(raw);

  if (normalized) {
    for (const [pattern, message] of KNOWN_ERROR_MESSAGES) {
      if (pattern.test(normalized)) {
        return message;
      }
    }
    if (typeof window !== "undefined") {
      console.error("Web UI request failed:", normalized);
    }
  }

  return `${trimTerminalPunctuation(fallback) || DEFAULT_ERROR}`.replace(
    /[^。！？]$/u,
    "$&。",
  );
}

export function appendRecoveryHint(message: string, hint: string): string {
  const normalized = trimTerminalPunctuation(message);
  return `${normalized || DEFAULT_ERROR}。${hint}`;
}
