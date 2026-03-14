from __future__ import annotations

import math
import re
from pathlib import Path


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
    "why",
    "和",
    "是",
    "什么",
    "如何",
    "以及",
    "一个",
    "一种",
    "这个",
    "那个",
    "这些",
    "那些",
    "作者",
    "论文",
}


def normalize_identifier(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    return Path(text).name.casefold()


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text or "")]


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in _STOPWORDS and (len(token) > 1 or any("\u4e00" <= ch <= "\u9fff" for ch in token))
    }


def route_accuracy(expected_route: str, predicted_route: str) -> float:
    return 1.0 if expected_route == predicted_route else 0.0


def recall_at_k(relevances: list[int], gold_count: int, *, k: int) -> float:
    if gold_count <= 0:
        return 1.0
    return min(sum(relevances[:k]), gold_count) / gold_count


def reciprocal_rank(relevances: list[int]) -> float:
    for index, relevance in enumerate(relevances, start=1):
        if relevance:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevances: list[int], gold_count: int, *, k: int) -> float:
    def _dcg(scores: list[int]) -> float:
        return sum(score / math.log2(index + 2) for index, score in enumerate(scores))

    observed = _dcg(relevances[:k])
    ideal = _dcg([1] * min(gold_count, k))
    if ideal == 0:
        return 1.0
    return observed / ideal


def redundancy_rate(values: list[str]) -> float:
    if not values:
        return 0.0
    normalized = [value.strip().casefold() for value in values if value.strip()]
    if not normalized:
        return 0.0
    return max(0.0, 1.0 - (len(set(normalized)) / len(normalized)))


def groundedness_score(answer: str, evidence_quotes: list[str]) -> float:
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 1.0
    evidence_terms = set().union(*(content_terms(quote) for quote in evidence_quotes))
    if not evidence_terms:
        return 0.0
    return len(answer_terms & evidence_terms) / len(answer_terms)


def citation_precision(
    cited_doc_ids: list[str],
    gold_doc_ids: list[str],
    *,
    cited_node_ids: list[str] | None = None,
    gold_node_ids: list[str] | None = None,
) -> float:
    cited_docs = {normalize_identifier(item) for item in cited_doc_ids if normalize_identifier(item)}
    gold_docs = {normalize_identifier(item) for item in gold_doc_ids if normalize_identifier(item)}
    cited_nodes = {str(item).strip().casefold() for item in cited_node_ids or [] if str(item).strip()}
    gold_nodes = {str(item).strip().casefold() for item in gold_node_ids or [] if str(item).strip()}

    if gold_nodes:
        if not cited_nodes:
            return 0.0
        return len(cited_nodes & gold_nodes) / len(cited_nodes)

    if gold_docs:
        if not cited_docs:
            return 0.0
        return len(cited_docs & gold_docs) / len(cited_docs)

    total = len(cited_docs) + len(cited_nodes)
    if total == 0:
        return 0.0
    return 0.0


def answer_completeness(answer: str, reference_answer: str) -> float:
    reference_terms = content_terms(reference_answer)
    if not reference_terms:
        return 1.0
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 0.0
    return len(reference_terms & answer_terms) / len(reference_terms)


def hallucination_rate_rule(
    answer: str,
    evidence_quotes: list[str],
    *,
    reference_answer: str = "",
) -> float:
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 0.0
    support_terms = set().union(*(content_terms(quote) for quote in evidence_quotes))
    support_terms |= content_terms(reference_answer)
    if not support_terms:
        return 1.0
    unsupported = answer_terms - support_terms
    return len(unsupported) / len(answer_terms)
