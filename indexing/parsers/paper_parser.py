"""Project-owned parser schema and stable identifier helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


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
