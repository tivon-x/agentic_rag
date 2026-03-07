from __future__ import annotations

from collections import defaultdict
from langchain_core.documents import Document


def _clean_source(meta: dict) -> str | None:
    src = str(meta.get("source", "")).strip()
    if not src:
        return None
    # Only keep the basename-ish tail for readability.
    return src.replace("\\", "/").split("/")[-1]


def format_retrieval_only_answer(
    question: str,
    docs: list[Document],
    *,
    max_snippets: int = 6,
    max_chars_per_snippet: int = 600,
) -> str:
    """Generate a deterministic answer without calling any LLM.

    This is a fallback for offline/demo mode or missing API credentials.
    """
    if not docs:
        return "I couldn't find any information to answer your question in the available sources."

    grouped: dict[str, list[Document]] = defaultdict(list)
    unknown_key = "unknown"
    for d in docs:
        grouped[_clean_source(d.metadata) or unknown_key].append(d)

    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append(
        "Offline mode is enabled (or API credentials are missing), so this response is a retrieval-only view of the most relevant excerpts."  # noqa: E501
    )
    lines.append("")
    lines.append("## Top excerpts")

    snippet_count = 0
    for source, items in sorted(
        grouped.items(), key=lambda x: (x[0] == unknown_key, x[0])
    ):
        for doc in items:
            if snippet_count >= max_snippets:
                break
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            if len(snippet) > max_chars_per_snippet:
                snippet = snippet[: max_chars_per_snippet - 1] + "…"
            page = doc.metadata.get("page")
            page_str = (
                f" p.{page}"
                if isinstance(page, int | str) and str(page).strip()
                else ""
            )
            src_label = source if source != unknown_key else "(unknown source)"
            lines.append(f"- **{src_label}{page_str}**: {snippet}")
            snippet_count += 1
        if snippet_count >= max_snippets:
            break

    sources: list[str] = []
    for source in grouped.keys():
        if source != unknown_key and "." in source:
            sources.append(source)
    sources = sorted(set(sources))
    if sources:
        lines.append("\n---\n**Sources:**")
        lines.extend([f"- {s}" for s in sources])
    return "\n".join(lines)
