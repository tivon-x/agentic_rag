export type ChatRole = "user" | "assistant" | "system";

export type ChatEvidence = {
  node_id: string;
  paper_id: string | null;
  paper_title: string | null;
  source: string;
  section_path: string[];
  page: number | null;
  quote: string;
  score: number | null;
  relevance: string | null;
};

export type ChatMessage = {
  role: ChatRole;
  content: string;
  evidence?: ChatEvidence[] | null;
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
  evidence?: ChatEvidence[] | null;
  error?: string;
};

export type ParseStatus =
  | "queued"
  | "parsing"
  | "parsed"
  | "degraded"
  | "needs_ocr"
  | "failed";

export type MetadataEvidence = {
  value: string | number | string[] | null;
  source: string;
  confidence: number;
};

export type PaperSummary = {
  id: string;
  content_hash: string;
  file_name: string;
  source_type: string;
  size_bytes: number;
  title: string | null;
  authors: string[];
  year: number | null;
  venue: string | null;
  doi: string | null;
  arxiv_id: string | null;
  metadata: Record<string, MetadataEvidence>;
  metadata_status: "needs_review" | "verified";
  metadata_version: number;
  parse_status: ParseStatus;
  parse_error: string | null;
  fallback_reason: string | null;
  latest_version_id: string | null;
  created_at: string;
  updated_at: string;
  file_url: string;
};

export type PaperSection = {
  id: string;
  parent_id: string | null;
  title: string;
  level: number;
  ordinal: number;
  page_start: number;
  page_end: number;
  heading_path: string[];
};

export type PaperVersion = {
  id: string;
  parser_name: string;
  parser_version: string;
  normalization_version: string;
  status: "parsed" | "degraded" | "needs_ocr" | "failed";
  fallback_reason: string | null;
  quality: Record<string, unknown>;
  page_count: number;
  duration_ms: number;
  created_at: string;
};

export type PaperDetail = PaperSummary & {
  paper_version: PaperVersion | null;
  sections: PaperSection[];
  reindex_job_id?: string | null;
};

export type PaperListResponse = {
  items: PaperSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type SearchScores = {
  vector: number | null;
  bm25: number | null;
  fusion: number;
  boosts: Record<string, number>;
  final: number;
  rerank_rank: number;
};

export type SearchResult = {
  passage_id: string;
  paper_id: string;
  paper_title: string | null;
  authors: string[];
  year: number | null;
  section_id: string;
  section_title: string;
  page_start: number;
  page_end: number;
  quote_text: string;
  block_type: string;
  scores: SearchScores;
  paper_url: string;
  pdf_url: string;
};

export type SearchResponse = {
  query: string;
  index_version: string;
  total: number;
  results: SearchResult[];
  degraded_reason: string | null;
};
