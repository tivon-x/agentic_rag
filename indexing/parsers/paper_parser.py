"""Project-owned parser schema and stable identifier helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pymupdf


NORMALIZATION_VERSION = "structure-v1"


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def paper_id_for_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_evidence(path: str | Path) -> list[tuple[str, str]]:
    """Return a stable page fingerprint and deterministic source-word order."""
    evidence: list[tuple[str, str]] = []
    with pymupdf.open(path) as document:
        for page in document:
            digest = hashlib.sha256()
            digest.update(
                document.xref_object(page.xref, compressed=True).encode(
                    "utf-8",
                    errors="ignore",
                )
            )
            referenced_xrefs = {
                *page.get_contents(),
                *(int(image[0]) for image in page.get_images(full=True)),
            }
            for xref in sorted(referenced_xrefs):
                stream = document.xref_stream(xref)
                if stream:
                    digest.update(stream)
            source_text = " ".join(
                str(word[4])
                for word in page.get_text("words", sort=True)
            ).strip()
            evidence.append((digest.hexdigest(), source_text))
    return evidence


@dataclass(slots=True)
class MetadataField:
    value: str | int | list[str] | None
    source: str
    confidence: float


@dataclass(slots=True)
class PaperMetadata:
    title: MetadataField
    authors: MetadataField
    year: MetadataField
    venue: MetadataField
    doi: MetadataField
    arxiv_id: MetadataField

    def values(self) -> dict[str, Any]:
        return {
            name: getattr(self, name).value
            for name in (
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "arxiv_id",
            )
        }

    def evidence(self) -> dict[str, dict[str, Any]]:
        return {
            name: asdict(getattr(self, name))
            for name in (
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "arxiv_id",
            )
        }


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    source_fingerprint: str | None = None
    source_text: str | None = None


@dataclass(slots=True)
class ParsedBlock:
    page_number: int
    block_type: str
    text: str
    caption: str | None = None


@dataclass(slots=True)
class ParsedSection:
    title: str
    level: int
    ordinal: int
    page_start: int
    page_end: int
    heading_path: list[str]
    blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass(slots=True)
class ParserQuality:
    passed: bool
    page_coverage: float
    nonempty_page_ratio: float
    character_ratio_vs_legacy: float
    page_numbers_monotonic: bool
    needs_ocr: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedPaper:
    source_path: str
    page_count: int
    parser_name: str
    parser_version: str
    normalization_version: str
    metadata: PaperMetadata
    pages: list[ParsedPage]
    sections: list[ParsedSection]
    quality: ParserQuality | None = None
    status: str = "parsed"
    fallback_reason: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperParser(Protocol):
    name: str
    version: str

    def parse(self, file_path: str) -> ParsedPaper: ...
