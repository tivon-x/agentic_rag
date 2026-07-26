"""Automatic parser quality gates and fallback diagnostics."""

from __future__ import annotations

from indexing.parsers.paper_parser import ParsedPaper, ParserQuality


def assess_parser_quality(
    parsed: ParsedPaper,
    legacy: ParsedPaper,
) -> ParserQuality:
    page_count = max(parsed.page_count, 1)
    page_coverage = min(1.0, len(parsed.pages) / page_count)
    nonempty_pages = sum(bool(page.text.strip()) for page in parsed.pages)
    nonempty_page_ratio = nonempty_pages / page_count
    parsed_characters = sum(len(page.text) for page in parsed.pages)
    legacy_characters = sum(len(page.text) for page in legacy.pages)
    character_ratio = (
        parsed_characters / legacy_characters
        if legacy_characters
        else 1.0
    )
    page_numbers = [page.page_number for page in parsed.pages]
    monotonic = (
        page_numbers == sorted(page_numbers)
        and all(1 <= page <= parsed.page_count for page in page_numbers)
        and len(page_numbers) == len(set(page_numbers))
    )
    needs_ocr = nonempty_page_ratio < 0.5 and parsed_characters < page_count * 200

    reasons: list[str] = []
    if page_coverage < 0.95:
        reasons.append("page_coverage_below_95_percent")
    if nonempty_page_ratio < 0.9 and not needs_ocr:
        reasons.append("nonempty_page_ratio_below_90_percent")
    if character_ratio < 0.6:
        reasons.append("character_count_below_60_percent_of_legacy")
    if not monotonic:
        reasons.append("page_numbers_invalid")
    if needs_ocr:
        reasons.append("needs_ocr")
    return ParserQuality(
        passed=not [reason for reason in reasons if reason != "needs_ocr"],
        page_coverage=page_coverage,
        nonempty_page_ratio=nonempty_page_ratio,
        character_ratio_vs_legacy=character_ratio,
        page_numbers_monotonic=monotonic,
        needs_ocr=needs_ocr,
        reasons=reasons,
    )
