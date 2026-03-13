from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_PROFILE = {
    "name": "",
    "summary": "",
    "coverage": "",
    "non_coverage": "",
    "usage_notes": "",
    "source_examples": [],
    "recommended_questions": [],
    "forbidden_questions": [],
    "domain_keywords": [],
    "preferred_answer_style": "",
    "primary_entities": [],
}

_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_string_list(values: Any, *, limit: int = 10) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def _profile_terms(text: str) -> set[str]:
    expanded_text = _CAMEL_BOUNDARY_RE.sub(" ", text)
    return {term.casefold() for term in _WORD_RE.findall(expanded_text) if term.strip()}


def _matching_phrases(
    query: str,
    phrases: list[str],
    *,
    min_term_overlap: int = 1,
    allow_compact_substring: bool = True,
) -> list[str]:
    query_terms = _profile_terms(query)
    compact_query = _NON_WORD_RE.sub("", query).casefold()
    if not query_terms:
        return []

    matches: list[str] = []
    for phrase in phrases:
        phrase_terms = _profile_terms(phrase)
        compact_phrase = _NON_WORD_RE.sub("", phrase).casefold()
        required_overlap = max(1, min(min_term_overlap, len(phrase_terms)))
        if phrase_terms and len(query_terms & phrase_terms) >= required_overlap:
            matches.append(phrase)
            continue
        if (
            allow_compact_substring
            and compact_phrase
            and compact_query
            and (
            compact_phrase in compact_query or compact_query in compact_phrase
            )
        ):
            matches.append(phrase)
    return matches


def normalize_corpus_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CORPUS_PROFILE)
    if isinstance(profile, dict):
        merged.update(profile)

    return {
        "name": str(merged.get("name", "")).strip(),
        "summary": str(merged.get("summary", "")).strip(),
        "coverage": str(merged.get("coverage", "")).strip(),
        "non_coverage": str(merged.get("non_coverage", "")).strip(),
        "usage_notes": str(merged.get("usage_notes", "")).strip(),
        "source_examples": _normalize_string_list(merged.get("source_examples", [])),
        "recommended_questions": _normalize_string_list(
            merged.get("recommended_questions", []),
            limit=12,
        ),
        "forbidden_questions": _normalize_string_list(
            merged.get("forbidden_questions", []),
            limit=12,
        ),
        "domain_keywords": _normalize_string_list(
            merged.get("domain_keywords", []),
            limit=20,
        ),
        "preferred_answer_style": str(
            merged.get("preferred_answer_style", "")
        ).strip(),
        "primary_entities": _normalize_string_list(
            merged.get("primary_entities", []),
            limit=20,
        ),
    }


def corpus_profile_path(index_dir: Path) -> Path:
    return index_dir / "corpus_profile.json"


def load_corpus_profile(index_dir: Path) -> dict[str, Any]:
    path = corpus_profile_path(index_dir)
    if not path.exists():
        return normalize_corpus_profile(None)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return normalize_corpus_profile(None)

    return normalize_corpus_profile(data if isinstance(data, dict) else None)


def save_corpus_profile(
    index_dir: Path,
    *,
    name: str,
    summary: str,
    coverage: str,
    usage_notes: str,
    source_examples: list[str],
    non_coverage: str = "",
    recommended_questions: list[str] | None = None,
    forbidden_questions: list[str] | None = None,
    domain_keywords: list[str] | None = None,
    preferred_answer_style: str = "",
    primary_entities: list[str] | None = None,
) -> Path:
    path = corpus_profile_path(index_dir)
    payload = normalize_corpus_profile(
        {
            "name": name.strip(),
            "summary": summary.strip(),
            "coverage": coverage.strip(),
            "non_coverage": non_coverage.strip(),
            "usage_notes": usage_notes.strip(),
            "source_examples": source_examples[:10],
            "recommended_questions": recommended_questions or [],
            "forbidden_questions": forbidden_questions or [],
            "domain_keywords": domain_keywords or [],
            "preferred_answer_style": preferred_answer_style.strip(),
            "primary_entities": primary_entities or [],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def format_corpus_profile(profile: dict[str, Any]) -> str:
    normalized = normalize_corpus_profile(profile)
    lines = [
        f"知识库名称: {normalized['name'] or '未命名知识库'}",
        f"内容摘要: {normalized['summary'] or '未填写'}",
        f"覆盖范围: {normalized['coverage'] or '未填写'}",
        f"不覆盖范围: {normalized['non_coverage'] or '未填写'}",
        f"使用说明: {normalized['usage_notes'] or '未填写'}",
        "推荐提问: "
        + (
            "; ".join(normalized["recommended_questions"])
            if normalized["recommended_questions"]
            else "暂无"
        ),
        "禁止/不建议问题: "
        + (
            "; ".join(normalized["forbidden_questions"])
            if normalized["forbidden_questions"]
            else "暂无"
        ),
        "领域关键词: "
        + (
            ", ".join(normalized["domain_keywords"])
            if normalized["domain_keywords"]
            else "暂无"
        ),
        "核心实体: "
        + (
            ", ".join(normalized["primary_entities"])
            if normalized["primary_entities"]
            else "暂无"
        ),
        f"偏好回答风格: {normalized['preferred_answer_style'] or '未填写'}",
        "代表性文件: "
        + (
            ", ".join(normalized["source_examples"])
            if normalized["source_examples"]
            else "暂无"
        ),
    ]
    return "\n".join(lines)


def build_corpus_profile_context(profile: dict[str, Any]) -> str:
    normalized = normalize_corpus_profile(profile)
    if not any(normalized.values()):
        return "No corpus profile is available."

    lines = [
        f"Knowledge Base Name: {normalized['name'] or 'Unnamed knowledge base'}",
        f"Summary: {normalized['summary'] or 'Not provided'}",
        f"Coverage: {normalized['coverage'] or 'Not provided'}",
        f"Non-Coverage: {normalized['non_coverage'] or 'Not provided'}",
        f"Usage Notes: {normalized['usage_notes'] or 'Not provided'}",
        "Representative Sources: "
        + (
            ", ".join(normalized["source_examples"][:10])
            if normalized["source_examples"]
            else "None listed"
        ),
        "Recommended Questions: "
        + (
            "; ".join(normalized["recommended_questions"][:8])
            if normalized["recommended_questions"]
            else "None listed"
        ),
        "Forbidden Questions: "
        + (
            "; ".join(normalized["forbidden_questions"][:8])
            if normalized["forbidden_questions"]
            else "None listed"
        ),
        "Domain Keywords: "
        + (
            ", ".join(normalized["domain_keywords"][:12])
            if normalized["domain_keywords"]
            else "None listed"
        ),
        "Primary Entities: "
        + (
            ", ".join(normalized["primary_entities"][:12])
            if normalized["primary_entities"]
            else "None listed"
        ),
        "Preferred Answer Style: "
        + (normalized["preferred_answer_style"] or "Not provided"),
    ]
    return "\n".join(lines)


def build_answer_style_instruction(profile: dict[str, Any] | None) -> str:
    normalized = normalize_corpus_profile(profile)
    return str(normalized.get("preferred_answer_style", "")).strip()


def analyze_corpus_profile_match(query: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_corpus_profile(profile)
    query_text = str(query).strip()
    query_terms = _profile_terms(query_text)
    if not query_terms:
        return {
            "matched_domain_keywords": [],
            "matched_primary_entities": [],
            "matched_non_coverage": [],
            "matched_forbidden_questions": [],
            "coverage_term_overlap": 0,
            "force_out_of_scope": False,
            "reason": "",
        }

    matched_domain_keywords = _matching_phrases(query_text, normalized["domain_keywords"])
    matched_primary_entities = _matching_phrases(
        query_text,
        normalized["primary_entities"],
    )
    matched_non_coverage = _matching_phrases(
        query_text,
        [normalized["non_coverage"]],
        min_term_overlap=2,
        allow_compact_substring=False,
    )
    matched_forbidden_questions = _matching_phrases(
        query_text,
        normalized["forbidden_questions"],
        min_term_overlap=1,
        allow_compact_substring=True,
    )

    coverage_terms = _profile_terms(
        " ".join(
            [
                normalized["name"],
                normalized["summary"],
                normalized["coverage"],
                normalized["usage_notes"],
                *normalized["domain_keywords"],
                *normalized["primary_entities"],
                *normalized["recommended_questions"],
            ]
        )
    )
    coverage_overlap = len(query_terms & coverage_terms)
    force_out_of_scope = bool(
        matched_forbidden_questions
        or (
            matched_non_coverage
            and coverage_overlap == 0
            and not matched_domain_keywords
            and not matched_primary_entities
        )
    )

    reason_parts: list[str] = []
    if matched_domain_keywords:
        reason_parts.append(
            "matched domain keywords: " + ", ".join(matched_domain_keywords[:3])
        )
    if matched_primary_entities:
        reason_parts.append(
            "matched primary entities: " + ", ".join(matched_primary_entities[:3])
        )
    if matched_forbidden_questions:
        reason_parts.append(
            "matched forbidden topics: "
            + ", ".join(matched_forbidden_questions[:3])
        )
    elif matched_non_coverage:
        reason_parts.append(
            "matched non-coverage topics: " + ", ".join(matched_non_coverage[:3])
        )

    return {
        "matched_domain_keywords": matched_domain_keywords,
        "matched_primary_entities": matched_primary_entities,
        "matched_non_coverage": matched_non_coverage,
        "matched_forbidden_questions": matched_forbidden_questions,
        "coverage_term_overlap": coverage_overlap,
        "force_out_of_scope": force_out_of_scope,
        "reason": "; ".join(reason_parts),
    }


def apply_profile_query_plan_prior(
    query_plan: dict[str, Any],
    *,
    original_query: str,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    plan = dict(query_plan or {})
    subqueries = _normalize_string_list(plan.get("subqueries", []), limit=3)
    if not subqueries:
        subqueries = [str(original_query).strip()]

    analysis = analyze_corpus_profile_match(original_query, profile)
    guidance_terms = [
        *analysis["matched_primary_entities"][:2],
        *analysis["matched_domain_keywords"][:2],
    ]

    enriched_subqueries: list[str] = []
    for query in subqueries:
        additions = [
            term for term in guidance_terms if term.casefold() not in query.casefold()
        ]
        if additions:
            enriched_subqueries.append(f"{query} {' '.join(additions[:2])}".strip())
        else:
            enriched_subqueries.append(query)

    preferred_node_types = _normalize_string_list(
        plan.get("preferred_node_types", []),
        limit=3,
    ) or ["paragraph"]
    if analysis["matched_primary_entities"] and "paragraph" not in preferred_node_types:
        preferred_node_types.append("paragraph")

    plan["subqueries"] = enriched_subqueries[:3]
    plan["preferred_node_types"] = preferred_node_types[:3]
    plan["profile_hints"] = {
        "matched_domain_keywords": analysis["matched_domain_keywords"][:3],
        "matched_primary_entities": analysis["matched_primary_entities"][:3],
    }
    return plan


def expand_queries_with_corpus_profile(
    queries: list[str],
    *,
    original_query: str,
    query_plan: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> list[str]:
    normalized_queries = _normalize_string_list(queries, limit=6)
    analysis = analyze_corpus_profile_match(original_query, profile)
    plan_hints = dict(query_plan or {}).get("profile_hints", {})
    guidance_terms = _normalize_string_list(
        [
            *analysis["matched_primary_entities"],
            *analysis["matched_domain_keywords"],
            *(plan_hints.get("matched_primary_entities", []) or []),
            *(plan_hints.get("matched_domain_keywords", []) or []),
        ],
        limit=4,
    )

    if not guidance_terms:
        return normalized_queries[:3]

    expanded = list(normalized_queries)
    for query in normalized_queries:
        additions = [
            term for term in guidance_terms if term.casefold() not in query.casefold()
        ]
        if not additions:
            continue
        expanded.append(f"{query} {' '.join(additions[:2])}".strip())
        if len(expanded) >= 3:
            break
    return _normalize_string_list(expanded, limit=3)
