"""Fixed retrieval pipeline registry and shared retrieval data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Literal

from langchain_core.documents import Document

from indexing.token_count import estimate_token_count


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PREFIX_LINE_RE = re.compile(
    r"^\[(TITLE|AUTHORS|YEAR|SECTION|BLOCK)\]\s*(.*)$",
    re.MULTILINE,
)

FusionMethod = Literal["minmax", "rrf"]
TokenizerName = Literal["whitespace_v1", "mixed_v1"]


@dataclass(frozen=True, slots=True)
class RetrievalPipelineConfig:
    """Configuration for one deterministic retrieval pipeline."""

    name: str
    use_metadata_prefix: bool
    tokenizer: TokenizerName
    use_sparse: bool
    use_dense: bool
    fusion_method: FusionMethod
    use_rerank: bool
    neighbor_window: int
    sparse_top_k: int = 40
    dense_top_k: int = 40
    rrf_k: int = 60
    rerank_top_n: int = 30
    final_top_k: int = 8
    max_context_passages: int = 12
    context_token_budget: int = 8000
    metadata_prefix_fields: tuple[str, ...] = (
        "title",
        "authors",
        "year",
        "section",
        "block",
    )

    def __post_init__(self) -> None:
        if not self.use_sparse and not self.use_dense:
            raise ValueError("A retrieval pipeline must enable sparse or dense recall.")
        if self.rrf_k != 60 and self.fusion_method == "rrf":
            raise ValueError("V2 RRF pipelines must use k=60.")
        for value in (
            self.sparse_top_k,
            self.dense_top_k,
            self.rerank_top_n,
            self.final_top_k,
            self.max_context_passages,
            self.context_token_budget,
        ):
            if value <= 0:
                raise ValueError("Retrieval pipeline budgets must be positive.")
        if self.neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative.")
        if self.rerank_top_n > self.sparse_top_k + self.dense_top_k:
            raise ValueError("rerank_top_n exceeds the fixed recall candidate budget.")
        if self.final_top_k > self.rerank_top_n:
            raise ValueError("final_top_k cannot exceed rerank_top_n.")
        if self.max_context_passages < self.final_top_k:
            raise ValueError("max_context_passages cannot be less than final_top_k.")

    @property
    def retrieval_schema(self) -> str:
        return (
            "metadata-prefixed-v1"
            if self.use_metadata_prefix
            else "quote-only-v1"
        )

    def index_contract(self) -> dict[str, Any]:
        """Return fields that determine persisted lexical and vector content."""
        return {
            "retrieval_schema": self.retrieval_schema,
            "tokenizer": self.tokenizer,
            "metadata_prefix_fields": (
                list(self.metadata_prefix_fields)
                if self.use_metadata_prefix
                else []
            ),
        }

    def config_hash(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_B2 = RetrievalPipelineConfig(
    name="v2_fixed",
    use_metadata_prefix=True,
    tokenizer="mixed_v1",
    use_sparse=True,
    use_dense=True,
    fusion_method="rrf",
    use_rerank=True,
    neighbor_window=0,
)

PIPELINE_REGISTRY = MappingProxyType(
    {
        "b0": RetrievalPipelineConfig(
            name="v2_b0",
            use_metadata_prefix=False,
            tokenizer="whitespace_v1",
            use_sparse=True,
            use_dense=True,
            fusion_method="minmax",
            use_rerank=False,
            neighbor_window=0,
        ),
        "b1": RetrievalPipelineConfig(
            name="v1_flat_rerank",
            use_metadata_prefix=False,
            tokenizer="whitespace_v1",
            use_sparse=True,
            use_dense=True,
            fusion_method="minmax",
            use_rerank=True,
            neighbor_window=0,
        ),
        "b2": _B2,
        "b3": replace(_B2, name="v2_expanded", neighbor_window=1),
        "b2_no_metadata": replace(
            _B2,
            name="v2_fixed_no_metadata",
            use_metadata_prefix=False,
        ),
        "b2_no_sparse": replace(
            _B2,
            name="v2_fixed_no_sparse",
            use_sparse=False,
        ),
        "b2_no_dense": replace(
            _B2,
            name="v2_fixed_no_dense",
            use_dense=False,
        ),
        "b2_minmax": replace(
            _B2,
            name="v2_fixed_minmax",
            fusion_method="minmax",
        ),
        "b2_no_rerank": replace(
            _B2,
            name="v2_fixed_no_rerank",
            use_rerank=False,
        ),
    }
)

_PIPELINE_ALIASES = MappingProxyType(
    {
        "v1_flat_rerank": "b1",
        "v2_fixed": "b2",
        "v2_expanded": "b3",
    }
)


def get_pipeline_config(name: str) -> RetrievalPipelineConfig:
    key = name.strip().lower()
    key = _PIPELINE_ALIASES.get(key, key)
    try:
        return PIPELINE_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(sorted((*PIPELINE_REGISTRY, *_PIPELINE_ALIASES)))
        raise ValueError(
            f"Unknown RETRIEVAL_PIPELINE {name!r}; choose one of: {choices}."
        ) from exc


def prepare_index_documents(
    documents: list[Document],
    pipeline: RetrievalPipelineConfig,
) -> list[Document]:
    """Materialize the exact persisted retrieval representation."""
    prepared: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        retrieval_text = str(
            metadata.get("retrieval_text") or document.page_content
        ).strip()
        quote_text = str(
            metadata.get("quote_text") or strip_metadata_prefix(retrieval_text)
        ).strip()
        if not quote_text:
            continue
        metadata["quote_text"] = quote_text
        metadata["retrieval_text"] = retrieval_text
        page_content = (
            select_metadata_prefix_fields(
                retrieval_text,
                quote_text=quote_text,
                fields=pipeline.metadata_prefix_fields,
            )
            if pipeline.use_metadata_prefix
            else quote_text
        )
        prepared.append(Document(page_content=page_content, metadata=metadata))
    return prepared


def select_metadata_prefix_fields(
    retrieval_text: str,
    *,
    quote_text: str,
    fields: tuple[str, ...],
) -> str:
    wanted = {field.casefold() for field in fields}
    lines = [
        f"[{match.group(1)}] {match.group(2).strip()}"
        for match in _PREFIX_LINE_RE.finditer(retrieval_text)
        if match.group(1).casefold() in wanted and match.group(2).strip()
    ]
    return "\n".join([*lines, quote_text]).strip()


def strip_metadata_prefix(text: str) -> str:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and _PREFIX_LINE_RE.fullmatch(lines[index]):
        index += 1
    return "\n".join(lines[index:]).strip()


def quote_document(document: Document) -> Document:
    """Return source-faithful context while retaining retrieval metadata."""
    quote = str(
        document.metadata.get("quote_text")
        or strip_metadata_prefix(document.page_content)
    ).strip()
    metadata = dict(document.metadata)
    metadata["quote_text"] = quote
    return Document(page_content=quote, metadata=metadata)


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
    stable_id = str(
        metadata.get("passage_id") or metadata.get("node_id") or ""
    ).strip()
    if stable_id:
        return stable_id
    raw = (
        str(metadata.get("source", ""))
        + "|"
        + str(metadata.get("page", ""))
        + "|"
        + str(metadata.get("quote_text") or document.page_content)
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def query_terms(text: str) -> set[str]:
    expanded_text = _CAMEL_BOUNDARY_RE.sub(" ", text)
    return {
        term.casefold()
        for term in _WORD_RE.findall(expanded_text)
        if term.strip()
    }


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


def build_document_from_node(
    node: Any, *, include_children: bool = False
) -> Document:
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
        "quote_text": node.text,
    }
    if include_children:
        metadata["is_parent_context"] = True
    return Document(page_content=node.text, metadata=metadata)


def merge_documents(
    documents: list[Document], *, merge_label: str
) -> Document:
    primary = documents[0]
    merged_ids = [
        str(doc.metadata.get("node_id", "")).strip()
        for doc in documents
        if str(doc.metadata.get("node_id", "")).strip()
    ]
    merged_text = "\n\n".join(
        quote_document(doc).page_content
        for doc in documents
        if doc.page_content
    )
    metadata = dict(primary.metadata)
    metadata["merged_node_ids"] = merged_ids
    metadata["merged_count"] = len(documents)
    metadata["packing_strategy"] = merge_label
    metadata["quote_text"] = merged_text
    metadata["token_count"] = sum(
        int(
            doc.metadata.get("token_count")
            or estimate_token_count(quote_document(doc).page_content)
        )
        for doc in documents
    )
    return Document(page_content=merged_text, metadata=metadata)
