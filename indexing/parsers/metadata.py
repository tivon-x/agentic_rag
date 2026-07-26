"""Deterministic scholarly metadata extraction with source evidence."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from indexing.parsers.paper_parser import MetadataField, PaperMetadata


_BAD_TITLE_RE = re.compile(
    r"^(?:untitled|microsoft word|arxiv:\S+|bookversion(?:\.dvi)?|"
    r"latex|document\d*|main(?:\.tex)?)$",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"\barxiv\s*:\s*((?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_VENUE_RE = re.compile(
    r"\b((?:proceedings of\s+)?(?:neurips|nips|iclr|icml|cvpr|acl|emnlp|"
    r"aaai|ijcai|conference on|journal of)[^\n]{0,100})",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"@|https?://|www\.", re.IGNORECASE)


def extract_pdf_metadata(file_path: str) -> PaperMetadata:
    with pymupdf.open(file_path) as document:
        raw = document.metadata or {}
        first_page = document[0] if document.page_count else None
        first_text = first_page.get_text("text") if first_page else ""
        first_two = "\n".join(
            document[index].get_text("text")
            for index in range(min(2, document.page_count))
        )
        screen_title, screen_confidence, title_bottom = _first_screen_title(
            first_page
        )

    pdf_title = _clean(raw.get("title", ""))
    if not _valid_title(pdf_title):
        pdf_title = ""
    title = _choose_title(
        pdf_title=pdf_title,
        screen_title=screen_title,
        screen_confidence=screen_confidence,
        filename=Path(file_path).stem,
    )

    pdf_authors = _split_authors(_clean(raw.get("author", "")))
    screen_authors = _first_screen_authors(first_text, screen_title, title_bottom)
    authors = (
        MetadataField(pdf_authors, "pdf_metadata", 0.9)
        if pdf_authors
        else MetadataField(
            screen_authors,
            "first_page_heuristic" if screen_authors else "unknown",
            0.72 if screen_authors else 0.0,
        )
    )

    venue_match = _VENUE_RE.search(first_two)
    arxiv_match = _ARXIV_RE.search(first_two)
    date_value = _clean(raw.get("creationDate", ""))
    date_year = _first_year(date_value)
    venue_year = _first_year(venue_match.group(1)) if venue_match else None
    arxiv_year = _year_from_arxiv(arxiv_match.group(1)) if arxiv_match else None
    body_year = venue_year or arxiv_year or _first_year(first_text)
    filename_year = _first_year(Path(file_path).stem)
    year_value, year_source, year_confidence = _first_present(
        (date_year, "pdf_metadata", 0.82),
        (body_year, "first_page_heuristic", 0.78),
        (filename_year, "filename", 0.45),
    )

    venue = (
        MetadataField(
            _clean(venue_match.group(1))[:160],
            "first_two_pages",
            0.68,
        )
        if venue_match
        else MetadataField(None, "unknown", 0.0)
    )
    doi_match = _DOI_RE.search(first_two)
    doi = (
        MetadataField(
            doi_match.group(0).rstrip(".,;)"),
            "first_two_pages",
            0.95,
        )
        if doi_match
        else MetadataField(None, "unknown", 0.0)
    )
    arxiv_id = (
        MetadataField(arxiv_match.group(1), "first_two_pages", 0.95)
        if arxiv_match
        else MetadataField(None, "unknown", 0.0)
    )
    return PaperMetadata(
        title=title,
        authors=authors,
        year=MetadataField(year_value, year_source, year_confidence),
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def _choose_title(
    *,
    pdf_title: str,
    screen_title: str,
    screen_confidence: float,
    filename: str,
) -> MetadataField:
    if pdf_title and (
        not screen_title
        or _titles_equivalent(pdf_title, screen_title)
        or screen_confidence < 0.82
    ):
        return MetadataField(pdf_title, "pdf_metadata", 0.92)
    if screen_title:
        return MetadataField(
            screen_title,
            "first_page_heuristic",
            screen_confidence,
        )
    fallback = _clean_filename(filename)
    return MetadataField(
        fallback or None,
        "filename" if fallback else "unknown",
        0.45 if fallback else 0.0,
    )


def _first_screen_title(
    page: pymupdf.Page | None,
) -> tuple[str, float, float]:
    if page is None:
        return "", 0.0, 0.0
    candidates: list[tuple[float, float, str]] = []
    payload = page.get_text("dict")
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            text = "".join(
                str(span.get("text", "")) for span in line.get("spans", [])
            ).strip()
            if text:
                lines.append(text)
            sizes.extend(
                float(span.get("size", 0))
                for span in line.get("spans", [])
                if str(span.get("text", "")).strip()
            )
        text = _clean(" ".join(lines))
        bbox = block.get("bbox") or (0, 0, 0, 0)
        if not _valid_title(text) or not sizes or float(bbox[1]) > page.rect.height * 0.55:
            continue
        candidates.append((max(sizes), float(bbox[3]), text))
    if not candidates:
        return "", 0.0, 0.0
    candidates.sort(key=lambda item: (-item[0], item[1]))
    size, bottom, title = candidates[0]
    confidence = 0.88 if size >= 16 else 0.78
    return title[:300], confidence, bottom


def _first_screen_authors(
    first_text: str,
    title: str,
    title_bottom: float,
) -> list[str]:
    del title_bottom
    lines = [_clean(line) for line in first_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []
    start = 0
    if title:
        title_terms = set(title.casefold().split())
        for index, line in enumerate(lines[:30]):
            if len(title_terms & set(line.casefold().split())) >= max(
                1, len(title_terms) // 2
            ):
                start = index + 1
                break
    candidates: list[str] = []
    for line in lines[start : start + 24]:
        lower = line.casefold()
        if lower in {"abstract", "introduction"} or lower.startswith("abstract "):
            break
        if title and _normalized_title(line) in _normalized_title(title):
            continue
        if (
            _EMAIL_RE.search(line)
            or any(
                marker in lower
                for marker in (
                    "university",
                    "institute",
                    "department",
                    "google",
                    "openai",
                    "research",
                    "laboratory",
                    "conference",
                )
            )
            or len(line) > 180
        ):
            continue
        names = _split_authors(line)
        if names and all(_looks_like_person(name) for name in names):
            candidates.extend(names)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate.casefold() not in {item.casefold() for item in deduped}:
            deduped.append(candidate)
    return deduped[:30]


def _split_authors(value: str) -> list[str]:
    if not value:
        return []
    cleaned = re.sub(r"[*†‡]+", "", value)
    parts = re.split(r"\s*(?:;|\band\b|&|\n)\s*", cleaned)
    if len(parts) == 1 and cleaned.count(",") >= 2:
        parts = re.split(r"\s*,\s*", cleaned)
    return [part.strip(" ,") for part in parts if part.strip(" ,")]


def _looks_like_person(value: str) -> bool:
    words = value.split()
    return (
        2 <= len(words) <= 6
        and len(value) <= 80
        and not any(character.isdigit() for character in value)
        and sum(character.isalpha() for character in value) >= 5
    )


def _valid_title(value: str) -> bool:
    cleaned = _clean(value)
    if (
        not cleaned
        or _BAD_TITLE_RE.fullmatch(cleaned)
        or cleaned.casefold().startswith("arxiv:")
    ):
        return False
    if len(cleaned) < 5 or len(cleaned) > 300:
        return False
    return sum(character.isalpha() for character in cleaned) >= 4


def _titles_equivalent(left: str, right: str) -> bool:
    left_value = _normalized_title(left)
    right_value = _normalized_title(right)
    return bool(left_value and right_value) and (
        left_value in right_value or right_value in left_value
    )


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_filename(value: str) -> str:
    cleaned = re.sub(r"^\d+[-_.\s]*", "", value)
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    return _clean(cleaned)


def _first_year(value: str) -> int | None:
    match = _YEAR_RE.search(value)
    return int(match.group(1)) if match else None


def _year_from_arxiv(value: str) -> int | None:
    modern = re.match(r"(\d{2})\d{2}\.\d{4,5}", value)
    if modern:
        year = int(modern.group(1))
        return 2000 + year if year < 90 else 1900 + year
    legacy = re.search(r"/(\d{2})\d{5}", value)
    if legacy:
        year = int(legacy.group(1))
        return 2000 + year if year < 90 else 1900 + year
    return None


def _first_present(
    *values: tuple[int | None, str, float],
) -> tuple[int | None, str, float]:
    for value, source, confidence in values:
        if value is not None:
            return value, source, confidence
    return None, "unknown", 0.0
