from langchain_core.retrievers import BaseRetriever
from langchain.tools import tool

class ToolFactory:
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        
    def _search_documents(self, query: str):
        retrieved_docs = self.retriever.invoke(query)
        
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    def create_tools(self):
        search_document_tool = tool(
            self._search_documents,
            description="Search for relevant documents based on the query. Input should be a string query, and output will be a list of relevant documents.",
            response_format="content_and_artifact"
        )
        return [search_document_tool]