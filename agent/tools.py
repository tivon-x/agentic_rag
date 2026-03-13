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

    def _search_documents(self, query: str):
        logging.getLogger(__name__).debug("Tool search query: %s", query)
        packed_context = (
            self.retriever.retrieve(query, query_plan=self.get_active_query_plan())
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
        return serialized, retrieved_docs

    def create_tools(self):
        search_document_tool = tool(
            "search_relevant_chunks",
            description="Search for relevant document chunks by query string. Returns serialized excerpts + the raw Document list.",
            response_format="content_and_artifact",
        )(self._search_documents)
        return [search_document_tool]
