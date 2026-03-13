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


def render_grounded_answer(payload: dict) -> str:
    """Render a structured grounded answer payload to Markdown."""
    answer = str(payload.get("answer", "")).strip()
    reasoning_summary = str(payload.get("reasoning_summary", "")).strip()
    limitations = str(payload.get("limitations", "")).strip()
    confidence = payload.get("confidence")
    evidence = payload.get("evidence", []) or []

    if not answer:
        return "I couldn't find any relevant information in the available sources to answer your question."

    lines = [answer]

    if reasoning_summary:
        lines.extend(["", "## Reasoning summary", reasoning_summary])

    if confidence is not None:
        try:
            confidence_pct = round(float(confidence) * 100)
            lines.extend(["", f"**Confidence:** {confidence_pct}%"])
        except (TypeError, ValueError):
            pass

    if limitations:
        lines.extend(["", "## Limitations", limitations])

    if evidence:
        lines.extend(["", "## Evidence"])
        for item in evidence:
            source = str(item.get("source", "unknown")).strip() or "unknown"
            doc_id = str(item.get("doc_id", "")).strip()
            node_id = str(item.get("node_id", "")).strip()
            page = item.get("page")
            section_path = item.get("section_path", []) or []
            location = " > ".join(str(part) for part in section_path if str(part).strip())

            details: list[str] = []
            if doc_id:
                details.append(f"doc_id={doc_id}")
            if node_id:
                details.append(f"node_id={node_id}")
            if isinstance(page, int):
                details.append(f"p.{page}")
            if location:
                details.append(location)

            label = f"- **{source}**"
            if details:
                label += f" ({'; '.join(details)})"
            lines.append(label)

            quote = str(item.get("quote", "")).strip()
            if quote:
                lines.append(f"  > {quote}")

            relevance = str(item.get("relevance", "")).strip()
            if relevance:
                lines.append(f"  Relevance: {relevance}")

    return "\n".join(lines)


def render_grounded_citations(payload: dict) -> str:
    """Render grouped citations directly from GroundedAnswer.evidence."""
    evidence = payload.get("evidence", []) or []
    if not evidence:
        return "当前回答没有可展示的结构化引用。"

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in evidence:
        source = str(item.get("source", "unknown")).strip() or "unknown"
        section_path = item.get("section_path", []) or []
        if isinstance(section_path, str):
            section_path = [section_path]
        section_label = " > ".join(
            str(part).strip() for part in section_path if str(part).strip()
        ) or "Unscoped section"
        grouped[(source, section_label)].append(item)

    lines = ["## Citations"]
    for (source, section_label), items in sorted(grouped.items()):
        lines.extend(["", f"### {source}", f"**Section:** {section_label}"])
        for item in items:
            details: list[str] = []
            doc_id = str(item.get("doc_id", "")).strip()
            node_id = str(item.get("node_id", "")).strip()
            page = item.get("page")
            if doc_id:
                details.append(f"doc_id={doc_id}")
            if node_id:
                details.append(f"node_id={node_id}")
            if isinstance(page, int):
                details.append(f"p.{page}")

            bullet = "- Citation"
            if details:
                bullet += f" ({'; '.join(details)})"
            lines.append(bullet)

            quote = str(item.get("quote", "")).strip()
            if quote:
                lines.append(f"  > {quote}")

            relevance = str(item.get("relevance", "")).strip()
            if relevance:
                lines.append(f"  Relevance: {relevance}")

    return "\n".join(lines)


def render_out_of_scope_answer(payload: dict) -> str:
    """Render a structured out-of-scope response to Markdown."""
    reason = str(payload.get("reason", "")).strip()
    boundary = str(payload.get("boundary", "")).strip()
    suggestion = str(payload.get("suggestion", "")).strip()
    next_action = str(payload.get("next_action", "")).strip()

    lines: list[str] = []
    if reason:
        lines.append(reason)
    if boundary:
        lines.extend(["", "## Current coverage", boundary])
    if suggestion:
        lines.extend(["", "## Better question", suggestion])
    if next_action:
        lines.extend(["", "## Next step", next_action])
    return "\n".join(lines).strip() or "This question is outside the current knowledge base."
