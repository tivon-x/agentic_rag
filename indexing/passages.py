"""Stable passage construction and metadata-prefixed retrieval text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from indexing.parsers.paper_parser import (
    ParsedPaper,
    stable_hash,
)


@dataclass(slots=True)
class SectionRecord:
    id: str
    parent_id: str | None
    title: str
    level: int
    ordinal: int
    page_start: int
    page_end: int
    heading_path: list[str]


@dataclass(slots=True)
class PassageRecord:
    id: str
    section_id: str
    page_start: int
    page_end: int
    quote_text: str
    retrieval_text: str
    block_type: str
    ordinal: int


def paper_version_id(
    paper_id: str,
    *,
    parser_name: str,
    parser_version: str,
    normalization_version: str,
) -> str:
    return stable_hash(
        paper_id,
        parser_name,
        parser_version,
        normalization_version,
    )


def build_catalog_records(
    parsed: ParsedPaper,
    *,
    paper_id: str,
    metadata_values: dict[str, Any],
    metadata_evidence: dict[str, dict[str, Any]],
    max_input_chars: int,
) -> tuple[str, list[SectionRecord], list[PassageRecord]]:
    version_id = paper_version_id(
        paper_id,
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        normalization_version=parsed.normalization_version,
    )
    section_records: list[SectionRecord] = []
    passage_records: list[PassageRecord] = []
    section_ids_by_path: dict[tuple[str, ...], str] = {}
    passage_ordinal = 0

    for section in parsed.sections:
        heading_path = section.heading_path or [section.title]
        section_id = stable_hash(
            version_id,
            " / ".join(item.casefold().strip() for item in heading_path),
            section.ordinal,
        )
        parent_id = _parent_section_id(heading_path, section_ids_by_path)
        section_ids_by_path[tuple(heading_path)] = section_id
        section_records.append(
            SectionRecord(
                id=section_id,
                parent_id=parent_id,
                title=section.title,
                level=section.level,
                ordinal=section.ordinal,
                page_start=section.page_start,
                page_end=section.page_end,
                heading_path=heading_path,
            )
        )
        prefix = build_retrieval_prefix(
            metadata_values,
            metadata_evidence,
            heading_path=heading_path,
        )
        for block in section.blocks:
            block_prefix = f"{prefix}[BLOCK] {block.block_type}\n"
            available = max_input_chars - len(block_prefix)
            if available <= 0:
                raise ValueError(
                    "Metadata prefix exceeds EMBEDDING_MAX_INPUT_CHARS; "
                    "shorten the corrected metadata."
                )
            for quote in split_quote_text(block.text, max_chars=available):
                retrieval_text = f"{block_prefix}{quote}"
                if len(retrieval_text) > max_input_chars:
                    raise ValueError(
                        "Passage splitting failed to enforce "
                        "EMBEDDING_MAX_INPUT_CHARS."
                    )
                passage_id = stable_hash(
                    version_id,
                    section_id,
                    block.page_number,
                    passage_ordinal,
                    stable_hash(quote),
                )
                passage_records.append(
                    PassageRecord(
                        id=passage_id,
                        section_id=section_id,
                        page_start=block.page_number,
                        page_end=block.page_number,
                        quote_text=quote,
                        retrieval_text=retrieval_text,
                        block_type=block.block_type,
                        ordinal=passage_ordinal,
                    )
                )
                passage_ordinal += 1
    return version_id, section_records, passage_records


def build_retrieval_prefix(
    metadata_values: dict[str, Any],
    metadata_evidence: dict[str, dict[str, Any]],
    *,
    heading_path: list[str],
) -> str:
    lines: list[str] = []
    title = _trusted_value("title", metadata_values, metadata_evidence, 0.4)
    authors = _trusted_value("authors", metadata_values, metadata_evidence, 0.6)
    year = _trusted_value("year", metadata_values, metadata_evidence, 0.6)
    if title:
        lines.append(f"[TITLE] {title}")
    if authors:
        author_text = ", ".join(authors) if isinstance(authors, list) else str(authors)
        lines.append(f"[AUTHORS] {author_text}")
    if year:
        lines.append(f"[YEAR] {year}")
    if heading_path:
        lines.append(f"[SECTION] {' / '.join(heading_path)}")
    return "\n".join(lines) + ("\n" if lines else "")


def split_quote_text(text: str, *, max_chars: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("Passage max_chars must be positive.")
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]
    output: list[str] = []
    current = ""
    for paragraph in paragraphs or [normalized]:
        pieces = _split_oversized(paragraph, max_chars)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                output.append(current)
            current = piece
    if current:
        output.append(current)
    return output


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                pieces.append(sentence[start : start + max_chars])
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _trusted_value(
    name: str,
    values: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    minimum_confidence: float,
) -> Any:
    value = values.get(name)
    confidence = float((evidence.get(name) or {}).get("confidence") or 0.0)
    return value if value not in (None, "", []) and confidence >= minimum_confidence else None


def _parent_section_id(
    heading_path: list[str],
    section_ids_by_path: dict[tuple[str, ...], str],
) -> str | None:
    for length in range(len(heading_path) - 1, 0, -1):
        parent = section_ids_by_path.get(tuple(heading_path[:length]))
        if parent:
            return parent
    return None
