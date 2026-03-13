from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from indexing.bm25_index import BM25Bundle
from indexing.retrieval_pipeline import (
    PackedContext,
    RetrievalCandidate,
    build_document_from_node,
    corpus_terms,
    document_key,
    lexical_overlap_score,
    merge_documents,
    normalize_query_plan,
    normalize_text,
    query_terms,
)
from indexing.token_count import estimate_token_count
from indexing.vectorstore import VectorStore


def get_similarity_retriever(
    vectorstore: VectorStore, k: int, filter: dict | None = None
) -> BaseRetriever:
    return vectorstore.get_retriever(search_type="similarity", k=k, filter=filter)


class BM25Retriever(BaseRetriever):
    bundle: BM25Bundle
    k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.bundle.query(query, k=self.k)


class FusionRetriever(BaseRetriever):
    model_config = {"arbitrary_types_allowed": True}

    vectorstore: VectorStore
    bm25: BM25Bundle
    alpha: float = 0.5
    k: int = 10
    fetch_k: int = 40
    token_budget: int = 1800
    reranker_backend: str = "flashrank"
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    flashrank_cache_dir: str = ""
    flashrank_top_n: int = 10
    node_store: Any = None
    corpus_profile: dict[str, Any] | None = None
    _flashrank_reranker: Any = PrivateAttr(default=None)
    _cached_nodes: list[Any] | None = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        if self.fetch_k <= 0:
            self.fetch_k = max(40, self.k * 4)
        else:
            self.fetch_k = max(self.fetch_k, self.k)
        self.token_budget = max(self.token_budget, 200)
        self.flashrank_top_n = max(self.flashrank_top_n, self.k)

    def retrieve(
        self, query: str, *, query_plan: dict[str, Any] | None = None
    ) -> PackedContext:
        plan = normalize_query_plan(query, query_plan)
        retrieval_candidates, retrieval_debug = self._retrieve_candidates(plan)
        deduped_candidates, dedupe_debug = self._dedupe_candidates(retrieval_candidates)
        reranked_candidates, rerank_debug = self._rerank_candidates(
            query, deduped_candidates, plan
        )
        return self._pack_context(reranked_candidates, plan, retrieval_debug, dedupe_debug, rerank_debug)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query).passages

    def _retrieve_candidates(
        self, query_plan: dict[str, Any]
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        all_candidates: list[RetrievalCandidate] = []
        raw_counts: dict[str, int] = {}

        for subquery in query_plan["subqueries"]:
            all_candidates.extend(self._retrieve_from_fusion(subquery))
            structured = self._retrieve_structured_nodes(
                subquery, query_plan["preferred_node_types"]
            )
            all_candidates.extend(structured)
            raw_counts[subquery] = len(structured)

        return all_candidates, {
            "query_plan": query_plan,
            "raw_candidates": len(all_candidates),
            "structured_candidates": raw_counts,
        }

    def _retrieve_from_fusion(self, query: str) -> list[RetrievalCandidate]:
        epsilon = 1e-8
        vec_results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        vec_sims = [1.0 / (dist + epsilon) for _, dist in vec_results]
        if vec_sims:
            vmin, vmax = float(min(vec_sims)), float(max(vec_sims))
        else:
            vmin = vmax = 0.0

        vec_score_by_key: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}
        for (doc, _), sim in zip(vec_results, vec_sims, strict=False):
            key = document_key(doc)
            doc_by_key.setdefault(key, doc)
            vec_score_by_key[key] = 0.0 if vmax - vmin <= epsilon else (sim - vmin) / (vmax - vmin)

        bm_results = self.bm25.topk_with_scores(query, k=self.fetch_k)
        bm_scores = [score for _, score in bm_results]
        if bm_scores:
            bmin, bmax = float(min(bm_scores)), float(max(bm_scores))
        else:
            bmin = bmax = 0.0

        bm_score_by_key: dict[str, float] = {}
        for doc, score in bm_results:
            key = document_key(doc)
            doc_by_key.setdefault(key, doc)
            bm_score_by_key[key] = 0.0 if bmax - bmin <= epsilon else (score - bmin) / (bmax - bmin)

        candidates: list[RetrievalCandidate] = []
        for key, doc in doc_by_key.items():
            vector_score = vec_score_by_key.get(key, 0.0)
            bm25_score = bm_score_by_key.get(key, 0.0)
            candidates.append(
                RetrievalCandidate(
                    document=doc,
                    score=(self.alpha * vector_score) + ((1.0 - self.alpha) * bm25_score),
                    source_scores={"vector": vector_score, "bm25": bm25_score},
                    subquery=query,
                )
            )
        return candidates

    def _retrieve_structured_nodes(
        self, query: str, preferred_node_types: list[str]
    ) -> list[RetrievalCandidate]:
        if self.node_store is None:
            return []

        wanted = {node_type for node_type in preferred_node_types if node_type != "paragraph"}
        if not wanted:
            return []

        candidates: list[RetrievalCandidate] = []
        for node in self._load_cached_nodes():
            if node.node_type not in wanted or not node.text.strip():
                continue
            score = lexical_overlap_score(query, f"{node.title or ''} {node.text}")
            if score <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    document=build_document_from_node(node, include_children=True),
                    score=min(1.0, score),
                    source_scores={"structured_lexical": min(1.0, score)},
                    subquery=query,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.fetch_k]

    def _dedupe_candidates(
        self, candidates: list[RetrievalCandidate]
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        deduped_by_key: dict[str, RetrievalCandidate] = {}
        merge_log: list[str] = []
        text_keys: dict[str, str] = {}

        for candidate in candidates:
            doc = candidate.document
            key = str(doc.metadata.get("node_id", "")).strip() or document_key(doc)
            text_key = normalize_text(doc.page_content)

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

        deduped = list(deduped_by_key.values())
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
        q_terms = query_terms(query)
        profile_terms = corpus_terms(self.corpus_profile)
        preferred_node_types = set(query_plan.get("preferred_node_types", []))

        for candidate in candidates:
            metadata = candidate.document.metadata
            searchable = " ".join(
                [
                    str(metadata.get("title", "")),
                    str(metadata.get("parent_title", "")),
                    str(metadata.get("source", "")),
                    candidate.document.page_content,
                ]
            )
            searchable_terms = query_terms(searchable)

            title_terms = query_terms(
                " ".join(
                    [
                        str(metadata.get("title", "")),
                        str(metadata.get("parent_title", "")),
                    ]
                )
            )

            if q_terms and title_terms:
                overlap = len(q_terms & title_terms) / len(q_terms)
                if overlap > 0:
                    candidate.boosts["title_match"] = round(0.15 * overlap, 4)

            if preferred_node_types and str(metadata.get("node_type", "")) in preferred_node_types:
                candidate.boosts["node_type_match"] = 0.08

            if profile_terms and searchable_terms:
                overlap = len(profile_terms & searchable_terms) / max(len(profile_terms), 1)
                if overlap > 0:
                    candidate.boosts["corpus_profile"] = round(min(0.12, overlap), 4)

        reranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)
        flashrank_debug = {"enabled": False}
        if self.reranker_backend == "flashrank":
            reranked, flashrank_debug = self._apply_flashrank_rerank(query, reranked)
        debug_rows = [
            {
                "node_id": candidate.document.metadata.get("node_id"),
                "node_type": candidate.document.metadata.get("node_type"),
                "score": round(candidate.score, 4),
                "final_score": round(candidate.final_score, 4),
                "boosts": dict(candidate.boosts),
            }
            for candidate in reranked[: self.k * 2]
        ]
        return reranked, {
            "top_candidates": debug_rows,
            "flashrank": flashrank_debug,
        }

    def _pack_context(
        self,
        candidates: list[RetrievalCandidate],
        query_plan: dict[str, Any],
        retrieval_debug: dict[str, Any],
        dedupe_debug: dict[str, Any],
        rerank_debug: dict[str, Any],
    ) -> PackedContext:
        passages: list[Document] = []
        seen_passages: set[str] = set()
        total_tokens = 0

        for candidate in candidates:
            packed_document = self._expand_candidate(candidate, query_plan)
            packed_key = document_key(packed_document)
            if packed_key in seen_passages:
                continue

            token_count = int(
                packed_document.metadata.get("token_count")
                or estimate_token_count(packed_document.page_content)
            )
            if passages and total_tokens + token_count > self.token_budget:
                continue

            passages.append(packed_document)
            seen_passages.add(packed_key)
            total_tokens += token_count
            if len(passages) >= self.k:
                break

        return PackedContext(
            passages=passages,
            total_tokens=total_tokens,
            dropped_candidates=max(0, len(candidates) - len(passages)),
            packing_strategy="score_then_contiguity",
            debug={
                **retrieval_debug,
                "dedupe": dedupe_debug,
                "rerank": rerank_debug,
                "packed_count": len(passages),
                "total_tokens": total_tokens,
            },
        )

    def _apply_flashrank_rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        try:
            reranker = self._get_flashrank_reranker()
        except Exception as exc:
            return candidates, {"enabled": False, "error": str(exc)}

        docs = [candidate.document for candidate in candidates[: self.flashrank_top_n]]
        if not docs:
            return candidates, {"enabled": True, "model": self.flashrank_model, "top_n": 0}

        reranked_docs = reranker.compress_documents(documents=docs, query=query)
        reranked_by_key = {
            document_key(doc): index for index, doc in enumerate(reranked_docs)
        }

        reranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                reranked_by_key.get(document_key(candidate.document), len(reranked_docs)),
                -candidate.final_score,
            ),
        )
        for candidate in reranked_candidates:
            rank = reranked_by_key.get(document_key(candidate.document))
            if rank is not None:
                candidate.boosts["flashrank"] = round(1.0 / (rank + 1), 4)
        return reranked_candidates, {
            "enabled": True,
            "model": self.flashrank_model,
            "top_n": len(reranked_docs),
        }

    def _get_flashrank_reranker(self) -> Any:
        if self._flashrank_reranker is not None:
            return self._flashrank_reranker

        from langchain_community.document_compressors import FlashrankRerank

        kwargs: dict[str, Any] = {
            "model": self.flashrank_model,
            "top_n": self.flashrank_top_n,
        }
        if self.flashrank_cache_dir.strip():
            kwargs["cache_dir"] = self.flashrank_cache_dir
        self._flashrank_reranker = FlashrankRerank(**kwargs)
        return self._flashrank_reranker

    def _expand_candidate(
        self, candidate: RetrievalCandidate, query_plan: dict[str, Any]
    ) -> Document:
        if self.node_store is None:
            return candidate.document

        metadata = candidate.document.metadata
        node_id = str(metadata.get("node_id", "")).strip()
        parent_id = str(metadata.get("parent_id", "")).strip()
        node_type = str(metadata.get("node_type", "")).strip()
        preferred_node_types = set(query_plan.get("preferred_node_types", []))

        if "section" in preferred_node_types and parent_id and node_type == "paragraph":
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
                and int(parent.token_count or estimate_token_count(parent.text))
                <= max_section_tokens
            ):
                return build_document_from_node(parent, include_children=True)

        if not parent_id or node_type != "paragraph":
            return candidate.document

        siblings = sorted(
            self.node_store.get_children(parent_id),
            key=lambda item: item.order,
        )
        if len(siblings) <= 1:
            return candidate.document

        current_order = int(metadata.get("order", 0))
        window_docs = [
            build_document_from_node(sibling)
            for sibling in siblings
            if abs(sibling.order - current_order) <= 1 and sibling.text.strip()
        ]
        if len(window_docs) <= 1:
            return candidate.document
        return merge_documents(window_docs, merge_label="window_merge")

    def _load_cached_nodes(self) -> list[Any]:
        if self.node_store is None:
            return []
        if self._cached_nodes is None:
            self._cached_nodes = list(self.node_store.load_nodes())
        return list(self._cached_nodes)
