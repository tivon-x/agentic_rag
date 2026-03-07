from langchain_core.retrievers import BaseRetriever
from langchain.tools import tool
import logging


class ToolFactory:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def _search_documents(self, query: str):
        logging.getLogger(__name__).debug("Tool search query: %s", query)
        retrieved_docs = self.retriever.invoke(query)

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def create_tools(self):
        search_document_tool = tool(
            "search_child_chunks",
            description="Search for relevant document chunks by query string. Returns serialized excerpts + the raw Document list.",
            response_format="content_and_artifact",
        )(self._search_documents)
        return [search_document_tool]
