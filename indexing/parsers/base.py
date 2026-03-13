from __future__ import annotations

from pathlib import Path
from typing import Protocol

from indexing.models.doc_tree import ParsedDocumentTree


SUPPORTED_HIERARCHICAL_SUFFIXES = frozenset({".md", ".txt", ".pdf"})


class HierarchicalParser(Protocol):
    def parse(self, file_path: str) -> ParsedDocumentTree: ...


def build_parser(file_path: str) -> HierarchicalParser:
    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_HIERARCHICAL_SUFFIXES:
        raise ValueError(f"Unsupported file type for hierarchical parsing: {suffix}")
    if suffix == ".md":
        from indexing.parsers.markdown_parser import MarkdownHierarchicalParser

        return MarkdownHierarchicalParser()
    if suffix == ".pdf":
        from indexing.parsers.pdf_parser import PdfHierarchicalParser

        return PdfHierarchicalParser()
    if suffix == ".txt":
        from indexing.parsers.txt_parser import TxtHierarchicalParser

        return TxtHierarchicalParser()

    raise ValueError(f"No hierarchical parser registered for file type: {suffix}")
