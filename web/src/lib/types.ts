export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ChatResponse = {
  session_id: string;
  created_at: string;
  message: ChatMessage;
};

export type ChatSessionResponse = {
  session_id: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
};

export type CorpusProfile = {
  name: string;
  summary: string;
  coverage: string;
  non_coverage: string;
  usage_notes: string;
  source_examples: string[];
  recommended_questions: string[];
  forbidden_questions: string[];
  domain_keywords: string[];
  preferred_answer_style: string;
  primary_entities: string[];
};

export type IndexingJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type FileUploadResponse = {
  job_id: string;
  filename: string;
  status: IndexingJobStatus;
};

export type IndexingJobResponse = {
  id: string;
  status: IndexingJobStatus;
  created_at: string;
  updated_at: string;
  error_message: string | null;
  attempt_count: number;
  max_attempts: number;
  progress: Record<string, unknown> | null;
  active_version_before: string | null;
  target_version: string | null;
};

export type StreamToken = {
  type: "progress" | "evidence" | "answer.final" | "error";
  content?: string;
  session_id: string;
  citations_markdown?: string;
  error?: string;
};
