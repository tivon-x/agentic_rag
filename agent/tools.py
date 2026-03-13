from langchain_core.retrievers import BaseRetriever
from langchain.tools import tool
import contextvars
import logging


class ToolFactory:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self._active_query_plan: contextvars.ContextVar[dict | None] = (
            contextvars.ContextVar("active_query_plan", default=None)
        )

    def set_active_query_plan(self, query_plan: dict | None):
        return self._active_query_plan.set(dict(query_plan or {}))

    def reset_active_query_plan(self, token) -> None:
        self._active_query_plan.reset(token)

    def get_active_query_plan(self) -> dict | None:
        return self._active_query_plan.get()

    def _build_evidence_item(self, doc, *, subquery: str) -> dict:
        metadata = dict(doc.metadata)
        section_path = metadata.get("section_path") or metadata.get("title_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        quote = (doc.page_content or "").strip()
        if len(quote) > 400:
            quote = quote[:399] + "…"
        return {
            "doc_id": str(metadata.get("doc_id", "")).strip()
            or str(metadata.get("source", "")).strip(),
            "node_id": str(metadata.get("node_id", "")).strip()
            or str(metadata.get("id", "")).strip()
            or f"{subquery}:{hash(quote)}",
            "source": str(metadata.get("source", "")).strip() or "unknown",
            "section_path": [str(item) for item in section_path if str(item).strip()],
            "page": metadata.get("page")
            if isinstance(metadata.get("page"), int)
            else None,
            "quote": quote,
            "score": metadata.get("score")
            if isinstance(metadata.get("score"), int | float)
            else None,
            "relevance": None,
        }

    def _search_documents(self, query: str):
        logging.getLogger(__name__).debug("Tool search query: %s", query)
        active_query_plan = self.get_active_query_plan() or {}
        packed_context = (
            self.retriever.retrieve(query, query_plan=active_query_plan)
            if hasattr(self.retriever, "retrieve")
            else None
        )
        retrieved_docs = (
            packed_context.passages if packed_context is not None else self.retriever.invoke(query)
        )

        sections: list[str] = []
        if packed_context is not None:
            sections.append(
                "Retrieval Debug: "
                f"query_plan={packed_context.debug.get('query_plan')} "
                f"dedupe={packed_context.debug.get('dedupe')} "
                f"packed_count={packed_context.debug.get('packed_count')} "
                f"total_tokens={packed_context.debug.get('total_tokens')}"
            )

        sections.extend(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
        )
        serialized = "\n\n".join(sections)
        evidence = [
            self._build_evidence_item(doc, subquery=query) for doc in retrieved_docs
        ]
        artifact = {
            "subquery": query,
            "query_plan": active_query_plan,
            "packed_context": {
                "total_tokens": getattr(packed_context, "total_tokens", None),
                "dropped_candidates": getattr(packed_context, "dropped_candidates", None),
                "packing_strategy": getattr(packed_context, "packing_strategy", None),
                "passage_count": len(retrieved_docs),
            },
            "passages": [
                {
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                }
                for doc in retrieved_docs
            ],
            "evidence": evidence,
            "debug": getattr(packed_context, "debug", {}),
        }
        return serialized, artifact

    def create_tools(self):
        search_document_tool = tool(
            "search_relevant_chunks",
            description="Search for relevant document chunks by query string. Returns serialized excerpts + the raw Document list.",
            response_format="content_and_artifact",
        )(self._search_documents)
        return [search_document_tool]
