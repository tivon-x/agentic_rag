from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any

from langchain_core.documents import Document

from indexing.token_count import estimate_token_count


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass
class RetrievalCandidate:
    document: Document
    score: float
    source_scores: dict[str, float] = field(default_factory=dict)
    boosts: dict[str, float] = field(default_factory=dict)
    subquery: str = ""

    @property
    def final_score(self) -> float:
        return self.score + sum(self.boosts.values())


@dataclass
class PackedContext:
    passages: list[Document]
    total_tokens: int
    dropped_candidates: int
    packing_strategy: str
    debug: dict[str, Any] = field(default_factory=dict)


def normalize_query_plan(
    query: str, query_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    normalized = dict(query_plan or {})
    subqueries = [
        str(item).strip()
        for item in normalized.get("subqueries", []) or []
        if str(item).strip()
    ]
    preferred_node_types = [
        str(item).strip()
        for item in normalized.get("preferred_node_types", []) or []
        if str(item).strip()
    ]
    if not subqueries:
        subqueries = [query.strip()] if query.strip() else [query]
    if not preferred_node_types:
        preferred_node_types = ["paragraph"]
    normalized["intent"] = str(normalized.get("intent", "fact")).strip() or "fact"
    normalized["subqueries"] = subqueries[:3]
    normalized["preferred_node_types"] = preferred_node_types
    return normalized


def document_key(document: Document) -> str:
    metadata = document.metadata
    raw = (
        str(metadata.get("node_id", ""))
        + "|"
        + str(metadata.get("source", ""))
        + "|"
        + str(metadata.get("page", ""))
        + "|"
        + document.page_content
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def query_terms(text: str) -> set[str]:
    expanded_text = _CAMEL_BOUNDARY_RE.sub(" ", text)
    return {term.casefold() for term in _WORD_RE.findall(expanded_text) if term.strip()}


def profile_terms(
    corpus_profile: dict[str, Any] | None,
    *,
    keys: tuple[str, ...],
) -> set[str]:
    if not corpus_profile:
        return set()

    values: list[str] = []
    for key in keys:
        value = corpus_profile.get(key, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return query_terms(" ".join(values))


def corpus_terms(corpus_profile: dict[str, Any] | None) -> set[str]:
    return profile_terms(
        corpus_profile,
        keys=(
            "name",
            "summary",
            "coverage",
            "usage_notes",
            "domain_keywords",
            "primary_entities",
        ),
    )


def lexical_overlap_score(query: str, text: str) -> float:
    q_terms = query_terms(query)
    if not q_terms:
        return 0.0
    t_terms = query_terms(text)
    if not t_terms:
        return 0.0
    return len(q_terms & t_terms) / len(q_terms)


def build_document_from_node(node, *, include_children: bool = False) -> Document:
    metadata = {
        **node.metadata,
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "doc_id": node.doc_id,
        "node_type": node.node_type,
        "title": node.title,
        "order": node.order,
        "level": node.level,
        "token_count": node.token_count,
    }
    if include_children:
        metadata["is_parent_context"] = True
    return Document(page_content=node.text, metadata=metadata)


def merge_documents(documents: list[Document], *, merge_label: str) -> Document:
    primary = documents[0]
    merged_ids = [
        str(doc.metadata.get("node_id", "")).strip()
        for doc in documents
        if str(doc.metadata.get("node_id", "")).strip()
    ]
    merged_text = "\n\n".join(doc.page_content.strip() for doc in documents if doc.page_content)
    metadata = dict(primary.metadata)
    metadata["merged_node_ids"] = merged_ids
    metadata["merged_count"] = len(documents)
    metadata["packing_strategy"] = merge_label
    metadata["token_count"] = sum(
        int(doc.metadata.get("token_count") or estimate_token_count(doc.page_content))
        for doc in documents
    )
    return Document(page_content=merged_text, metadata=metadata)
