"""
Data processing module containing processors for different file types.
Each processor inherits from the DataProcessor base class and implements the process method to handle specific file types.

Currently implements PdfProcessor for PDF files, using PyPDFLoader to load PDF content with text cleaning to remove line breaks and hyphenated line breaks.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DataProcessor(ABC):
    """Base class for all data processors"""

    @abstractmethod
    def process(self, file_path: str) -> list[Document]:
        """
        Process file and return document list
        Args:
            file_path (str): File path
        Returns:
            List[Document]: Processed document list
        """
        pass


class PdfProcessor(DataProcessor):
    def process(self, file_path: str) -> list[Document]:
        try:
            loader = PyPDFLoader(file_path)
            pages = []
            for page in loader.lazy_load():
                page.page_content = clean_text(page.page_content)
                pages.append(page)
            return pages
        except Exception as e:
            raise ValueError(f"PdfProcessor error: {e}")


class TextProcessor(DataProcessor):
    """Plain text / markdown loader."""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def process(self, file_path: str) -> list[Document]:
        try:
            loader = TextLoader(file_path, encoding=self.encoding)
            docs = loader.load()
            for d in docs:
                d.page_content = clean_text(d.page_content)
            return docs
        except Exception as e:
            raise ValueError(f"TextProcessor error: {e}")


def clean_text(text: str) -> str:
    """
    Text cleaning function:
    1. Merge words broken by line breaks (e.g., xxx-\nxxx → xxxxxx)
    2. Convert line breaks to spaces
    """
    # Step 1: Handle hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Step 2: Handle normal line breaks
    text = re.sub(r"\n", " ", text)

    return text.strip()
