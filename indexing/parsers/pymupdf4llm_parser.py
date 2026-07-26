"""PyMuPDF4LLM parser adapter producing the project-owned paper schema."""

from __future__ import annotations

import time

import pymupdf
import pymupdf4llm

from indexing.parsers.metadata import extract_pdf_metadata
from indexing.parsers.paper_parser import (
    NORMALIZATION_VERSION,
    ParsedPage,
    ParsedPaper,
)
from indexing.parsers.structure_normalizer import normalize_structure


class PyMuPDF4LLMPaperParser:
    """Extract page-addressable Markdown without exposing third-party types."""

    name = "pymupdf4llm"
    version = "0.3.4"

    def parse(self, file_path: str) -> ParsedPaper:
        started = time.monotonic()
        with pymupdf.open(file_path) as document:
            page_count = document.page_count
        chunks = pymupdf4llm.to_markdown(
            file_path,
            page_chunks=True,
            show_progress=False,
            write_images=False,
            embed_images=False,
            extract_words=False,
        )
        pages = [
            ParsedPage(
                page_number=int(chunk.get("metadata", {}).get("page") or index),
                text=str(chunk.get("text") or "").strip(),
                tables=[
                    str(table.get("text") or table.get("markdown") or table)
                    for table in chunk.get("tables", [])
                    if table
                ],
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        return ParsedPaper(
            source_path=file_path,
            page_count=page_count,
            parser_name=self.name,
            parser_version=self.version,
            normalization_version=NORMALIZATION_VERSION,
            metadata=extract_pdf_metadata(file_path),
            pages=pages,
            sections=normalize_structure(pages),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
