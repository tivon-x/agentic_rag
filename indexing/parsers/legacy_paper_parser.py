"""Legacy page-level PDF parser used as an explicit fallback."""

from __future__ import annotations

import time

from langchain_community.document_loaders import PyPDFLoader

from indexing.parsers.metadata import extract_pdf_metadata
from indexing.parsers.paper_parser import (
    NORMALIZATION_VERSION,
    ParsedBlock,
    ParsedPage,
    ParsedPaper,
    ParsedSection,
)
from indexing.parsers.pdf_parser import _clean_pdf_page


class LegacyPaperParser:
    """Preserve the existing PyPDFLoader page-level behavior."""

    name = "legacy"
    version = "pypdfloader-v1"

    def parse(self, file_path: str) -> ParsedPaper:
        started = time.monotonic()
        documents = list(PyPDFLoader(file_path).lazy_load())
        pages = [
            ParsedPage(
                page_number=index,
                text=_clean_pdf_page(document.page_content),
            )
            for index, document in enumerate(documents, start=1)
        ]
        sections = [
            ParsedSection(
                title=f"Page {page.page_number}",
                level=1,
                ordinal=index,
                page_start=page.page_number,
                page_end=page.page_number,
                heading_path=[f"Page {page.page_number}"],
                blocks=(
                    [
                        ParsedBlock(
                            page_number=page.page_number,
                            block_type="paragraph",
                            text=page.text,
                        )
                    ]
                    if page.text
                    else []
                ),
            )
            for index, page in enumerate(pages)
        ]
        return ParsedPaper(
            source_path=file_path,
            page_count=len(pages),
            parser_name=self.name,
            parser_version=self.version,
            normalization_version=NORMALIZATION_VERSION,
            metadata=extract_pdf_metadata(file_path),
            pages=pages,
            sections=sections,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
