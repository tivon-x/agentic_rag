import type { ChatEvidence } from "@/lib/types";

export function evidenceHref(evidence: ChatEvidence): string | null {
  const paperId = evidence.paper_id?.trim();
  if (!paperId) {
    return null;
  }
  const page = evidence.page && evidence.page > 0 ? `?page=${evidence.page}` : "";
  return `/papers/${encodeURIComponent(paperId)}${page}`;
}

export function evidencePageLabel(evidence: ChatEvidence): string {
  return evidence.page && evidence.page > 0 ? `P.${evidence.page}` : "页码未知";
}

export function evidenceSectionLabel(evidence: ChatEvidence): string {
  return evidence.section_path.length
    ? evidence.section_path.join(" / ")
    : "章节未标注";
}
