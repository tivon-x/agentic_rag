"""Deterministic normalization from page Markdown into sections and blocks."""

from __future__ import annotations

import re

from indexing.parsers.paper_parser import ParsedBlock, ParsedPage, ParsedSection


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+\**(.+?)\**\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^\n]{1,100})|"
    r"(Appendix(?:\s+[A-Z0-9]+)?(?:[.:]\s*|\s+).{0,100}))$",
    re.IGNORECASE,
)
_NUMBER_ONLY_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+")
_CAPTION_RE = re.compile(r"^(?:table|figure|fig\.)\s+\d+", re.IGNORECASE)
_FORMULA_RE = re.compile(r"(?:\$\$|\\\[|[=∑∏√∞≤≥≈]{1,}|\\(?:frac|sum|prod)\b)")
_BAD_HEADING_RE = re.compile(
    r"^(?:#\s+of\b|page\s+\d+$|under review|arxiv:|provided proper attribution)",
    re.IGNORECASE,
)


def normalize_structure(pages: list[ParsedPage]) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    heading_stack: list[tuple[int, str]] = []
    current: ParsedSection | None = None
    section_ordinal = 0

    for page in pages:
        units = _page_units(page)
        for unit_type, text, level in units:
            if unit_type == "heading":
                heading_stack = [
                    item for item in heading_stack if item[0] < level
                ]
                heading_stack.append((level, text))
                current = ParsedSection(
                    title=text,
                    level=level,
                    ordinal=section_ordinal,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    heading_path=[item[1] for item in heading_stack],
                )
                sections.append(current)
                section_ordinal += 1
                continue

            if current is None:
                current = ParsedSection(
                    title=(
                        "Front matter"
                        if page.page_number == 1
                        else f"Page {page.page_number}"
                    ),
                    level=1,
                    ordinal=section_ordinal,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    heading_path=[],
                )
                sections.append(current)
                section_ordinal += 1
            current.page_end = page.page_number
            current.blocks.append(
                ParsedBlock(
                    page_number=page.page_number,
                    block_type=unit_type,
                    text=text,
                    caption=text if unit_type == "caption" else None,
                )
            )

    if not sections:
        return [
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
                    if page.text.strip()
                    else []
                ),
            )
            for index, page in enumerate(pages)
        ]
    return sections


def _page_units(page: ParsedPage) -> list[tuple[str, str, int]]:
    lines = page.text.replace("\r\n", "\n").splitlines()
    units: list[tuple[str, str, int]] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = "\n".join(paragraph).strip()
        paragraph.clear()
        if not text:
            return
        block_type = "formula" if _FORMULA_RE.search(text) else "paragraph"
        units.append((block_type, text, 0))

    while index < len(lines):
        raw = lines[index].strip()
        if not raw:
            flush_paragraph()
            index += 1
            continue

        heading = _heading(raw)
        if heading is None and _NUMBER_ONLY_RE.fullmatch(raw):
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines):
                candidate = lines[next_index].strip()
                if _looks_like_title(candidate):
                    depth = raw.rstrip(".)").count(".") + 1
                    heading = (min(depth + 1, 6), f"{raw.rstrip('.)')} {candidate}")
                    index = next_index
        if heading is not None:
            flush_paragraph()
            level, title = heading
            units.append(("heading", title, level))
            index += 1
            continue

        if _is_table_start(lines, index):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index].rstrip())
                index += 1
            units.append(("table", "\n".join(table_lines).strip(), 0))
            continue

        if _CAPTION_RE.match(raw):
            flush_paragraph()
            caption_lines = [raw]
            index += 1
            while index < len(lines) and lines[index].strip():
                caption_lines.append(lines[index].strip())
                index += 1
            caption = "\n".join(caption_lines)
            units.append(
                (
                    "table" if raw.casefold().startswith("table") else "caption",
                    caption,
                    0,
                )
            )
            continue

        paragraph.append(raw)
        index += 1

    flush_paragraph()
    known_tables = {text for kind, text, _ in units if kind == "table"}
    for table in page.tables:
        normalized = table.strip()
        if normalized and normalized not in known_tables:
            units.append(("table", normalized, 0))
    return units


def _heading(line: str) -> tuple[int, str] | None:
    markdown = _MARKDOWN_HEADING_RE.match(line)
    if markdown:
        title = _clean_heading(markdown.group(2))
        if _looks_like_title(title):
            return len(markdown.group(1)), title

    plain_line = _clean_heading(line)
    numbered = _NUMBERED_HEADING_RE.match(plain_line)
    if numbered:
        if numbered.group(3):
            title = _clean_heading(numbered.group(3))
            return 1, title
        number = numbered.group(1) or ""
        title = _clean_heading(f"{number} {numbered.group(2) or ''}")
        if _looks_like_title(title):
            return min(number.count(".") + 1, 6), title
    return None


def _clean_heading(value: str) -> str:
    value = re.sub(r"[*_`]+", "", value)
    return re.sub(r"\s+", " ", value).strip(" #")


def _looks_like_title(value: str) -> bool:
    cleaned = _clean_heading(value)
    if (
        not cleaned
        or len(cleaned) > 120
        or _BAD_HEADING_RE.match(cleaned)
        or cleaned.endswith((".", ",", ";"))
    ):
        return False
    words = cleaned.split()
    if len(words) > 14:
        return False
    if re.match(r"^(?:19|20)\d{2}\b", cleaned) and len(words) > 8:
        return False
    letters = sum(character.isalpha() for character in cleaned)
    return letters >= 3


def _is_table_start(lines: list[str], index: int) -> bool:
    if "|" not in lines[index]:
        return False
    if _TABLE_SEPARATOR_RE.match(lines[index]):
        return True
    return (
        index + 1 < len(lines)
        and bool(_TABLE_SEPARATOR_RE.match(lines[index + 1]))
    )
