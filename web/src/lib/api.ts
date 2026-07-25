import type {
  ChatResponse,
  ChatSessionResponse,
  CorpusProfile,
  FileUploadResponse,
  IndexingJobResponse,
} from "@/lib/types";

const API_PREFIX = "/api";

export async function fetchCorpusProfile(): Promise<CorpusProfile> {
  const response = await fetch(`${API_PREFIX}/corpus-profile`, {
    cache: "no-store",
  });
  return handleJson<CorpusProfile>(response);
}

export async function saveCorpusProfile(
  profile: CorpusProfile,
): Promise<CorpusProfile> {
  const response = await fetch(`${API_PREFIX}/corpus-profile`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(profile),
  });
  return handleJson<CorpusProfile>(response);
}

export async function createChatMessage(input: {
  message: string;
  sessionId?: string | null;
}): Promise<ChatResponse> {
  const response = await fetch(`${API_PREFIX}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: input.message,
      session_id: input.sessionId ?? undefined,
    }),
  });
  return handleJson<ChatResponse>(response);
}

export async function fetchChatSession(
  sessionId: string,
): Promise<ChatSessionResponse> {
  const response = await fetch(`${API_PREFIX}/chat/${sessionId}`, {
    cache: "no-store",
  });
  return handleJson<ChatSessionResponse>(response);
}

export async function uploadKnowledgeFiles(input: {
  files: File[];
  indexMode: "flat" | "hierarchical";
}): Promise<FileUploadResponse[]> {
  const formData = new FormData();
  formData.set("index_mode", input.indexMode);
  for (const file of input.files) {
    formData.append("files", file);
  }

  const response = await fetch(`${API_PREFIX}/index/files`, {
    method: "POST",
    headers: {
      "Idempotency-Key": crypto.randomUUID(),
    },
    body: formData,
  });
  return handleJson<FileUploadResponse[]>(response);
}

export async function fetchIndexingJob(
  jobId: string,
): Promise<IndexingJobResponse> {
  const response = await fetch(`${API_PREFIX}/indexing-jobs/${jobId}`, {
    cache: "no-store",
  });
  return handleJson<IndexingJobResponse>(response);
}

async function handleJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await extractError(response);
    throw new Error(message);
  }
  return (await response.json()) as T;
}

async function extractError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      return payload.detail;
    }
  } catch {}
  return `${response.status} ${response.statusText}`;
}
