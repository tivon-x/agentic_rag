"""Hierarchical parsers for multi-format documents."""
from indexing.parsers.legacy_paper_parser import LegacyPaperParser
from indexing.parsers.paper_parser import PaperParser
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser

__all__ = [
    "LegacyPaperParser",
    "PaperParser",
    "PyMuPDF4LLMPaperParser",
]
