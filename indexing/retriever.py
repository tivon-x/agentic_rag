from __future__ import annotations

from time import perf_counter
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from indexing.retrieval_pipeline import (
    PackedContext,
    RetrievalCandidate,
    RetrievalPipelineConfig,
    build_document_from_node,
    corpus_terms,
    document_key,
    get_pipeline_config,
    lexical_overlap_score,
    merge_documents,
    normalize_query_plan,
    normalize_text,
    profile_terms,
    query_terms,
    quote_document,
)
from indexing.stores.lexical_store import LexicalStore
from indexing.stores.node_store import NodeStore
from indexing.stores.vector_store import VectorStore
from indexing.token_count import estimate_token_count


def get_similarity_retriever(
    vectorstore: VectorStore, k: int, filter: dict | None = None
) -> BaseRetriever:
    return vectorstore.get_retriever(
        search_type="similarity",
        k=k,
        filter=filter,
    )


class BM25Retriever(BaseRetriever):
    lexical_store: LexicalStore
    k: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.lexical_store.query(query, k=self.k)


class FusionRetriever(BaseRetriever):
    """Fixed hybrid retrieval with explicit stage traces."""

    model_config = {"arbitrary_types_allowed": True}

    vectorstore: VectorStore
    lexical_store: LexicalStore
    pipeline: RetrievalPipelineConfig = get_pipeline_config("b1")
    alpha: float = 0.5
    k: int = 8
    fetch_k: int = 40
    token_budget: int = 8000
    reranker_backend: str = "flashrank"
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    flashrank_cache_dir: str = ""
    flashrank_top_n: int = 30
    strict_reranker: bool = False
    node_store: NodeStore | None = None
    corpus_profile: dict[str, Any] | None = None
    _flashrank_reranker: Any = PrivateAttr(default=None)
    _cached_nodes: list[Any] | None = PrivateAttr(default=None)
    _section_documents: dict[str, list[Document]] | None = PrivateAttr(
        default=None
    )

    def model_post_init(self, __context: Any) -> None:
        self.fetch_k = max(
            self.fetch_k,
            self.pipeline.sparse_top_k,
            self.pipeline.dense_top_k,
        )
        self.k = min(self.k, self.pipeline.final_top_k)
        self.token_budget = min(
            self.token_budget,
            self.pipeline.context_token_budget,
        )
        self.flashrank_top_n = min(
            self.flashrank_top_n,
            self.pipeline.rerank_top_n,
        )

    def retrieve(
        self,
        query: str,
        *,
        query_plan: dict[str, Any] | None = None,
    ) -> PackedContext:
        candidates, debug = self.search_scored(
            query,
            query_plan=query_plan,
        )
        return self._pack_context(
            candidates[: self.k],
            debug["query_plan"],
            debug["retrieval"],
            debug["dedupe"],
            debug["rerank"],
            timings=debug["timings_ms"],
            stages=debug["stages"],
        )

    def search_scored(
        self,
        query: str,
        *,
        query_plan: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        """Return ranked candidates and deterministic per-stage traces."""
        started = perf_counter()
        plan = normalize_query_plan(query, query_plan)
        normalized_at = perf_counter()
        retrieval_candidates, retrieval_debug = self._retrieve_candidates(plan)
        recalled_at = perf_counter()
        deduped_candidates, dedupe_debug = self._dedupe_candidates(
            retrieval_candidates
        )
        deduped_at = perf_counter()
        reranked_candidates, rerank_debug = self._rerank_candidates(
            query,
            deduped_candidates,
            plan,
        )
        reranked_at = perf_counter()
        if limit is not None:
            reranked_candidates = reranked_candidates[:limit]

        stages = {
            "recall": retrieval_debug.get("stage_rows", []),
            "fusion": self._candidate_rows(deduped_candidates),
            "rerank": self._candidate_rows(reranked_candidates),
        }
        return reranked_candidates, {
            "query_plan": plan,
            "retrieval": retrieval_debug,
            "dedupe": dedupe_debug,
            "rerank": rerank_debug,
            "stages": stages,
            "timings_ms": {
                "query_normalization": _milliseconds(
                    started,
                    normalized_at,
                ),
                "recall_and_fusion": _milliseconds(
                    normalized_at,
                    recalled_at,
                ),
                "dedupe": _milliseconds(recalled_at, deduped_at),
                "rerank": _milliseconds(deduped_at, reranked_at),
                "search_total": _milliseconds(started, reranked_at),
            },
        }

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.retrieve(query).passages

    def _retrieve_candidates(
        self,
        query_plan: dict[str, Any],
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        all_candidates: list[RetrievalCandidate] = []
        raw_counts: dict[str, int] = {}
        stage_rows: list[dict[str, Any]] = []

        for subquery in query_plan["subqueries"]:
            fused, fused_debug = self._retrieve_from_fusion(subquery)
            all_candidates.extend(fused)
            stage_rows.append(
                {
                    "subquery": subquery,
                    "dense": fused_debug["dense"],
                    "sparse": fused_debug["sparse"],
                    "fused": self._candidate_rows(fused),
                }
            )
            structured = self._retrieve_structured_nodes(
                subquery,
                query_plan["preferred_node_types"],
            )
            all_candidates.extend(structured)
            raw_counts[subquery] = len(structured)

        return all_candidates, {
            "query_plan": query_plan,
            "pipeline": self.pipeline.name,
            "raw_candidates": len(all_candidates),
            "structured_candidates": raw_counts,
            "stage_rows": stage_rows,
        }

    def _retrieve_from_fusion(
        self,
        query: str,
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        epsilon = 1e-8
        vector_rows = (
            self.vectorstore.search_with_score(
                query,
                k=self.pipeline.dense_top_k,
            )
            if self.pipeline.use_dense
            else []
        )
        sparse_rows = (
            self.lexical_store.topk_with_scores(
                query,
                k=self.pipeline.sparse_top_k,
            )
            if self.pipeline.use_sparse
            else []
        )

        doc_by_key: dict[str, Document] = {}
        dense_raw: dict[str, float] = {}
        sparse_raw: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}

        for rank, (document, distance) in enumerate(vector_rows, start=1):
            key = document_key(document)
            doc_by_key.setdefault(key, document)
            dense_raw[key] = 1.0 / (float(distance) + epsilon)
            dense_rank[key] = rank
        for rank, (document, score) in enumerate(sparse_rows, start=1):
            key = document_key(document)
            doc_by_key.setdefault(key, document)
            sparse_raw[key] = float(score)
            sparse_rank[key] = rank

        dense_normalized = _minmax(dense_raw)
        sparse_normalized = _minmax(sparse_raw)
        candidates: list[RetrievalCandidate] = []
        for key, document in doc_by_key.items():
            source_scores: dict[str, float] = {}
            if key in dense_raw:
                source_scores["vector"] = dense_raw[key]
                source_scores["vector_rank"] = float(dense_rank[key])
            if key in sparse_raw:
                source_scores["bm25"] = sparse_raw[key]
                source_scores["bm25_rank"] = float(sparse_rank[key])

            if self.pipeline.fusion_method == "rrf":
                dense_rrf = (
                    1.0 / (self.pipeline.rrf_k + dense_rank[key])
                    if key in dense_rank
                    else 0.0
                )
                sparse_rrf = (
                    1.0 / (self.pipeline.rrf_k + sparse_rank[key])
                    if key in sparse_rank
                    else 0.0
                )
                source_scores["vector_rrf"] = dense_rrf
                source_scores["bm25_rrf"] = sparse_rrf
                score = dense_rrf + sparse_rrf
            else:
                score = (
                    self.alpha * dense_normalized.get(key, 0.0)
                    + (1.0 - self.alpha)
                    * sparse_normalized.get(key, 0.0)
                )
            candidates.append(
                RetrievalCandidate(
                    document=document,
                    score=score,
                    source_scores=source_scores,
                    subquery=query,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates, {
            "fusion_method": self.pipeline.fusion_method,
            "rrf_k": (
                self.pipeline.rrf_k
                if self.pipeline.fusion_method == "rrf"
                else None
            ),
            "dense": [
                self._document_row(document, rank=rank, score=float(distance))
                for rank, (document, distance) in enumerate(
                    vector_rows,
                    start=1,
                )
            ],
            "sparse": [
                self._document_row(document, rank=rank, score=float(score))
                for rank, (document, score) in enumerate(
                    sparse_rows,
                    start=1,
                )
            ],
        }

    def _retrieve_structured_nodes(
        self,
        query: str,
        preferred_node_types: list[str],
    ) -> list[RetrievalCandidate]:
        if self.node_store is None:
            return []
        wanted = {
            node_type
            for node_type in preferred_node_types
            if node_type != "paragraph"
        }
        if not wanted:
            return []

        candidates: list[RetrievalCandidate] = []
        for node in self._load_cached_nodes():
            if node.node_type not in wanted or not node.text.strip():
                continue
            score = lexical_overlap_score(
                query,
                f"{node.title or ''} {node.text}",
            )
            if score <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    document=build_document_from_node(
                        node,
                        include_children=True,
                    ),
                    score=min(1.0, score),
                    source_scores={
                        "structured_lexical": min(1.0, score)
                    },
                    subquery=query,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.fetch_k]

    def _dedupe_candidates(
        self,
        candidates: list[RetrievalCandidate],
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        deduped_by_key: dict[str, RetrievalCandidate] = {}
        merge_log: list[str] = []
        text_keys: dict[str, str] = {}

        for candidate in candidates:
            document = candidate.document
            key = document_key(document)
            text_key = normalize_text(
                str(
                    document.metadata.get("quote_text")
                    or document.page_content
                )
            )
            canonical = text_keys.get(text_key)
            if canonical and canonical != key:
                existing = deduped_by_key.get(canonical)
                if existing is None:
                    deduped_by_key[key] = candidate
                    text_keys[text_key] = key
                    continue
                existing.score = max(existing.score, candidate.score)
                existing.source_scores.update(candidate.source_scores)
                merge_log.append(f"text:{key}->{canonical}")
                continue
            if key in deduped_by_key:
                existing = deduped_by_key[key]
                existing.score = max(existing.score, candidate.score)
                existing.source_scores.update(candidate.source_scores)
                merge_log.append(f"key:{key}")
                continue
            deduped_by_key[key] = candidate
            text_keys[text_key] = key

        deduped = sorted(
            deduped_by_key.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return deduped, {
            "raw_count": len(candidates),
            "deduped_count": len(deduped),
            "merge_log": merge_log,
        }

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        query_plan: dict[str, Any],
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        if not self.pipeline.use_rerank:
            return sorted(
                candidates,
                key=lambda item: item.score,
                reverse=True,
            ), {
                "enabled": False,
                "reason": "pipeline_disabled",
                "top_candidates": self._candidate_rows(candidates),
            }

        q_terms = query_terms(query)
        coverage_terms = corpus_terms(self.corpus_profile)
        domain_keyword_terms = profile_terms(
            self.corpus_profile,
            keys=("domain_keywords",),
        )
        primary_entity_terms = profile_terms(
            self.corpus_profile,
            keys=("primary_entities",),
        )
        non_coverage_terms = profile_terms(
            self.corpus_profile,
            keys=("non_coverage", "forbidden_questions"),
        )
        preferred_node_types = set(
            query_plan.get("preferred_node_types", [])
        )

        for candidate in candidates:
            metadata = candidate.document.metadata
            searchable = " ".join(
                [
                    str(metadata.get("title", "")),
                    str(metadata.get("paper_title", "")),
                    str(metadata.get("section_title", "")),
                    str(metadata.get("source", "")),
                    candidate.document.page_content,
                ]
            )
            searchable_terms = query_terms(searchable)
            title_terms = query_terms(
                " ".join(
                    [
                        str(metadata.get("title", "")),
                        str(metadata.get("paper_title", "")),
                        str(metadata.get("section_title", "")),
                    ]
                )
            )
            if q_terms and title_terms:
                overlap = len(q_terms & title_terms) / len(q_terms)
                if overlap > 0:
                    candidate.boosts["title_match"] = round(
                        0.15 * overlap,
                        4,
                    )
            if (
                preferred_node_types
                and str(metadata.get("node_type", ""))
                in preferred_node_types
            ):
                candidate.boosts["node_type_match"] = 0.08
            _apply_profile_boost(
                candidate,
                searchable_terms,
                coverage_terms,
                "corpus_profile",
                0.12,
            )
            _apply_profile_boost(
                candidate,
                searchable_terms,
                domain_keyword_terms,
                "domain_keyword",
                0.08,
            )
            _apply_profile_boost(
                candidate,
                searchable_terms,
                primary_entity_terms,
                "primary_entity",
                0.1,
            )
            if (
                non_coverage_terms
                and searchable_terms
                and not (q_terms & searchable_terms)
            ):
                overlap = len(
                    non_coverage_terms & searchable_terms
                ) / max(len(non_coverage_terms), 1)
                if overlap > 0:
                    candidate.boosts["non_coverage_penalty"] = round(
                        -min(0.12, overlap),
                        4,
                    )

        boosted = sorted(
            candidates,
            key=lambda item: item.final_score,
            reverse=True,
        )
        if self.reranker_backend != "flashrank":
            return boosted, {
                "top_candidates": self._candidate_rows(boosted),
                "flashrank": {
                    "enabled": False,
                    "reason": f"backend_{self.reranker_backend}",
                },
            }
        reranked, flashrank_debug = self._apply_flashrank_rerank(
            query,
            boosted,
        )
        return reranked, {
            "top_candidates": self._candidate_rows(reranked),
            "flashrank": flashrank_debug,
        }

    def _pack_context(
        self,
        candidates: list[RetrievalCandidate],
        query_plan: dict[str, Any],
        retrieval_debug: dict[str, Any],
        dedupe_debug: dict[str, Any],
        rerank_debug: dict[str, Any],
        *,
        timings: dict[str, float] | None = None,
        stages: dict[str, Any] | None = None,
    ) -> PackedContext:
        expansion_started = perf_counter()
        expanded, expansion_rows = self._expand_candidates(
            candidates,
            query_plan,
        )
        expansion_finished = perf_counter()

        passages: list[Document] = []
        seen_passages: set[str] = set()
        total_tokens = 0
        packing_rows: list[dict[str, Any]] = []
        for document in expanded:
            quoted = quote_document(document)
            key = document_key(quoted)
            if key in seen_passages:
                continue
            token_count = int(
                quoted.metadata.get("token_count")
                or estimate_token_count(quoted.page_content)
            )
            accepted = (
                total_tokens + token_count <= self.token_budget
                and len(passages) < self.pipeline.max_context_passages
            )
            packing_rows.append(
                {
                    **self._document_row(
                        quoted,
                        rank=len(packing_rows) + 1,
                    ),
                    "token_count": token_count,
                    "accepted": accepted,
                }
            )
            if not accepted:
                continue
            passages.append(quoted)
            seen_passages.add(key)
            total_tokens += token_count

        packed_at = perf_counter()
        timing_rows = dict(timings or {})
        timing_rows["expansion"] = _milliseconds(
            expansion_started,
            expansion_finished,
        )
        timing_rows["context_packing"] = _milliseconds(
            expansion_finished,
            packed_at,
        )
        timing_rows["retrieval_total"] = round(
            timing_rows.get("search_total", 0.0)
            + timing_rows["expansion"]
            + timing_rows["context_packing"],
            4,
        )
        stage_rows = dict(stages or {})
        stage_rows["expansion"] = expansion_rows
        stage_rows["context_packing"] = packing_rows
        return PackedContext(
            passages=passages,
            total_tokens=total_tokens,
            dropped_candidates=max(0, len(expanded) - len(passages)),
            packing_strategy="score_then_section_neighbors",
            debug={
                **retrieval_debug,
                "query_plan": query_plan,
                "dedupe": dedupe_debug,
                "rerank": rerank_debug,
                "pipeline": self.pipeline.name,
                "pipeline_config_hash": self.pipeline.config_hash(),
                "stages": stage_rows,
                "timings_ms": timing_rows,
                "packed_count": len(passages),
                "total_tokens": total_tokens,
            },
        )

    def _apply_flashrank_rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        try:
            reranker = self._get_flashrank_reranker()
        except Exception as exc:
            if self.strict_reranker:
                raise RuntimeError(
                    f"Required reranker is unavailable: {exc}"
                ) from exc
            return candidates, {"enabled": False, "error": str(exc)}

        selected = candidates[: self.pipeline.rerank_top_n]
        documents = [candidate.document for candidate in selected]
        if not documents:
            return candidates, {
                "enabled": True,
                "model": self.flashrank_model,
                "top_n": 0,
            }
        try:
            reranked_documents = list(
                reranker.compress_documents(
                    documents=documents,
                    query=query,
                )
            )
        except Exception as exc:
            if self.strict_reranker:
                raise RuntimeError(
                    f"Required reranker failed: {exc}"
                ) from exc
            return candidates, {"enabled": False, "error": str(exc)}

        reranked_by_key = {
            document_key(document): index
            for index, document in enumerate(reranked_documents)
        }
        reranked_selected = sorted(
            selected,
            key=lambda candidate: (
                reranked_by_key.get(
                    document_key(candidate.document),
                    len(reranked_documents),
                ),
                -candidate.final_score,
            ),
        )
        for candidate in reranked_selected:
            rank = reranked_by_key.get(document_key(candidate.document))
            if rank is not None:
                candidate.boosts["flashrank"] = round(
                    1.0 / (rank + 1),
                    4,
                )
        return [*reranked_selected, *candidates[len(selected) :]], {
            "enabled": True,
            "model": self.flashrank_model,
            "top_n": len(reranked_documents),
        }

    def _get_flashrank_reranker(self) -> Any:
        if self._flashrank_reranker is not None:
            return self._flashrank_reranker
        from langchain_community.document_compressors import FlashrankRerank

        kwargs: dict[str, Any] = {
            "model": self.flashrank_model,
            "top_n": self.pipeline.rerank_top_n,
        }
        if self.flashrank_cache_dir.strip():
            kwargs["cache_dir"] = self.flashrank_cache_dir
        self._flashrank_reranker = FlashrankRerank(**kwargs)
        return self._flashrank_reranker

    def _expand_candidates(
        self,
        candidates: list[RetrievalCandidate],
        query_plan: dict[str, Any],
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        documents: list[Document] = []
        trace: list[dict[str, Any]] = []
        seen: set[str] = set()

        def append_document(
            document: Document,
            *,
            seed_rank: int,
            is_seed: bool,
        ) -> bool:
            key = document_key(document)
            if key in seen:
                return False
            seen.add(key)
            documents.append(document)
            trace.append(
                {
                    **self._document_row(
                        document,
                        rank=len(documents),
                    ),
                    "seed_rank": seed_rank,
                    "is_seed": is_seed,
                }
            )
            return len(documents) >= self.pipeline.max_context_passages

        for seed_rank, candidate in enumerate(candidates, start=1):
            if append_document(
                candidate.document,
                seed_rank=seed_rank,
                is_seed=True,
            ):
                return documents, trace
        if self.pipeline.neighbor_window <= 0:
            return documents, trace

        for seed_rank, candidate in enumerate(candidates, start=1):
            seed = candidate.document
            neighbors = [
                document
                for document in self._catalog_neighbors(seed)
                if document_key(document) != document_key(seed)
            ]
            if not neighbors and self.node_store is not None:
                expanded = self._expand_candidate(candidate, query_plan)
                if document_key(expanded) != document_key(seed):
                    neighbors = [expanded]
            for neighbor in neighbors:
                if append_document(
                    neighbor,
                    seed_rank=seed_rank,
                    is_seed=False,
                ):
                    return documents, trace
        return documents, trace

    def _catalog_neighbors(self, seed: Document) -> list[Document]:
        section_id = str(seed.metadata.get("section_id") or "").strip()
        if not section_id:
            return [seed]
        if self._section_documents is None:
            grouped: dict[str, list[Document]] = {}
            for document in self.vectorstore.get_all_documents():
                current_section = str(
                    document.metadata.get("section_id") or ""
                ).strip()
                if current_section:
                    grouped.setdefault(current_section, []).append(document)
            for documents in grouped.values():
                documents.sort(
                    key=lambda item: int(item.metadata.get("order") or 0)
                )
            self._section_documents = grouped
        siblings = self._section_documents.get(section_id, [])
        if not siblings:
            return [seed]
        seed_order = int(seed.metadata.get("order") or 0)
        window = self.pipeline.neighbor_window
        selected = [
            document
            for document in siblings
            if abs(int(document.metadata.get("order") or 0) - seed_order)
            <= window
        ]
        selected.sort(
            key=lambda item: (
                abs(int(item.metadata.get("order") or 0) - seed_order),
                int(item.metadata.get("order") or 0),
            )
        )
        return selected or [seed]

    def _expand_candidate(
        self,
        candidate: RetrievalCandidate,
        query_plan: dict[str, Any],
    ) -> Document:
        if self.node_store is None:
            return candidate.document
        metadata = candidate.document.metadata
        node_id = str(metadata.get("node_id", "")).strip()
        parent_id = str(metadata.get("parent_id", "")).strip()
        node_type = str(metadata.get("node_type", "")).strip()
        preferred_node_types = set(
            query_plan.get("preferred_node_types", [])
        )
        if (
            "section" in preferred_node_types
            and parent_id
            and node_type == "paragraph"
        ):
            parent = self.node_store.get_parent(node_id)
            candidate_token_count = int(
                metadata.get("token_count")
                or estimate_token_count(candidate.document.page_content)
            )
            max_section_tokens = max(300, candidate_token_count * 3)
            if (
                query_plan.get("intent") == "summary"
                and parent is not None
                and parent.node_type == "section"
                and parent.text.strip()
                and int(
                    parent.token_count
                    or estimate_token_count(parent.text)
                )
                <= max_section_tokens
            ):
                return build_document_from_node(
                    parent,
                    include_children=True,
                )
        if not parent_id or node_type != "paragraph":
            return candidate.document
        siblings = sorted(
            self.node_store.get_children(parent_id),
            key=lambda item: item.order,
        )
        current_order = int(metadata.get("order", 0))
        window_documents = [
            build_document_from_node(sibling)
            for sibling in siblings
            if abs(sibling.order - current_order)
            <= self.pipeline.neighbor_window
            and sibling.text.strip()
        ]
        if len(window_documents) <= 1:
            return candidate.document
        return merge_documents(
            window_documents,
            merge_label="window_merge",
        )

    def _load_cached_nodes(self) -> list[Any]:
        if self.node_store is None:
            return []
        if self._cached_nodes is None:
            self._cached_nodes = list(self.node_store.load_nodes())
        return list(self._cached_nodes)

    def _candidate_rows(
        self,
        candidates: list[RetrievalCandidate],
    ) -> list[dict[str, Any]]:
        return [
            {
                **self._document_row(
                    candidate.document,
                    rank=rank,
                    score=candidate.score,
                ),
                "final_score": round(candidate.final_score, 8),
                "source_scores": {
                    key: round(value, 8)
                    for key, value in candidate.source_scores.items()
                },
                "boosts": dict(candidate.boosts),
            }
            for rank, candidate in enumerate(candidates, start=1)
        ]

    @staticmethod
    def _document_row(
        document: Document,
        *,
        rank: int,
        score: float | None = None,
    ) -> dict[str, Any]:
        metadata = document.metadata
        row: dict[str, Any] = {
            "rank": rank,
            "passage_id": str(
                metadata.get("passage_id")
                or metadata.get("node_id")
                or ""
            ),
            "paper_id": str(
                metadata.get("paper_id")
                or metadata.get("doc_id")
                or ""
            ),
            "section_id": str(metadata.get("section_id") or ""),
            "page": metadata.get("page_start") or metadata.get("page"),
        }
        row["node_id"] = row["passage_id"]
        if score is not None:
            row["score"] = round(float(score), 8)
        return row


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum - minimum <= 1e-8:
        return {key: 0.0 for key in values}
    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in values.items()
    }


def _apply_profile_boost(
    candidate: RetrievalCandidate,
    searchable_terms: set[str],
    profile_values: set[str],
    label: str,
    cap: float,
) -> None:
    if not profile_values or not searchable_terms:
        return
    overlap = len(profile_values & searchable_terms) / max(
        len(profile_values),
        1,
    )
    if overlap > 0:
        candidate.boosts[label] = round(min(cap, overlap), 4)


def _milliseconds(started: float, finished: float) -> float:
    return round((finished - started) * 1000, 4)
