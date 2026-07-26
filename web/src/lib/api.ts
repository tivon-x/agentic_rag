import type {
  ChatResponse,
  ChatSessionResponse,
  CorpusProfile,
  FileUploadResponse,
  IndexingJobResponse,
  PaperDetail,
  PaperListResponse,
  SearchResponse,
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

export async function fetchPapers(input?: {
  query?: string;
  parseStatus?: string;
}): Promise<PaperListResponse> {
  const params = new URLSearchParams();
  if (input?.query) {
    params.set("q", input.query);
  }
  if (input?.parseStatus) {
    params.set("parse_status", input.parseStatus);
  }
  const suffix = params.size ? `?${params.toString()}` : "";
  const response = await fetch(`${API_PREFIX}/papers${suffix}`, {
    cache: "no-store",
  });
  return handleJson<PaperListResponse>(response);
}

export async function fetchPaper(paperId: string): Promise<PaperDetail> {
  const response = await fetch(
    `${API_PREFIX}/papers/${encodeURIComponent(paperId)}`,
    { cache: "no-store" },
  );
  return handleJson<PaperDetail>(response);
}

export async function updatePaperMetadata(
  paperId: string,
  metadataVersion: number,
  payload: Partial<{
    title: string | null;
    authors: string[];
    year: number | null;
    venue: string | null;
    doi: string | null;
    arxiv_id: string | null;
  }>,
): Promise<PaperDetail> {
  const response = await fetch(
    `${API_PREFIX}/papers/${encodeURIComponent(paperId)}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "If-Match": String(metadataVersion),
      },
      body: JSON.stringify(payload),
    },
  );
  return handleJson<PaperDetail>(response);
}

export async function searchLibrary(input: {
  query: string;
  paperId?: string;
  limit?: number;
}): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: input.query,
    limit: String(input.limit ?? 20),
  });
  if (input.paperId) {
    params.set("paper_id", input.paperId);
  }
  const response = await fetch(`${API_PREFIX}/search?${params.toString()}`, {
    cache: "no-store",
  });
  return handleJson<SearchResponse>(response);
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
