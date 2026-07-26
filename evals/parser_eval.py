"""Evaluate parser page evidence, boundary detection, metadata, and latency."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from indexing.parsers.legacy_paper_parser import LegacyPaperParser
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents = [
        item
        for item in payload["documents"]
        if args.split == "all" or item["split"] == args.split
    ]
    result = evaluate(documents, root=dataset_path.parents[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def evaluate(
    documents: list[dict[str, Any]],
    *,
    root: Path,
) -> dict[str, Any]:
    primary_parser = PyMuPDF4LLMPaperParser()
    legacy_parser = LegacyPaperParser()
    page_correct = 0
    page_total = 0
    anchor_correct = 0
    anchor_total = 0
    markdown_anchor_correct = 0
    markdown_anchor_total = 0
    markdown_evidence_correct = 0
    markdown_evidence_total = 0
    order_correct = 0
    order_total = 0
    fingerprint_correct = 0
    fingerprint_total = 0
    character_ratios: list[float] = []
    latency_ratios: list[float] = []
    title_correct = 0
    section_labels: list[tuple[bool, bool]] = []
    table_labels: list[tuple[bool, bool]] = []
    cases: list[dict[str, Any]] = []

    for item in documents:
        path = (root / item["path"]).resolve()
        started = time.monotonic()
        parsed = primary_parser.parse(str(path))
        primary_seconds = max(time.monotonic() - started, 0.0001)
        started = time.monotonic()
        legacy = legacy_parser.parse(str(path))
        legacy_seconds = max(time.monotonic() - started, 0.0001)
        latency_ratios.append(primary_seconds / legacy_seconds)
        title_matches = _normalize_title(str(parsed.metadata.title.value or "")) == (
            _normalize_title(str(item["expected_title"]))
        )
        title_correct += int(title_matches)
        parsed_pages = {page.page_number: page for page in parsed.pages}
        legacy_pages = {page.page_number: page for page in legacy.pages}
        case_pages: list[dict[str, Any]] = []
        for focus in item["focus_pages"]:
            page_number = int(focus["page"])
            page_total += 1
            evidence = evaluate_page_evidence(focus, parsed_pages)
            page_exists = bool(evidence["page_exists"])
            page_correct += int(evidence["page_correct"])
            if evidence["anchors_checked"]:
                anchor_total += 1
                anchor_correct += int(evidence["anchors_correct"])
            if evidence["markdown_anchors_checked"]:
                markdown_anchor_total += 1
                markdown_anchor_correct += int(
                    evidence["markdown_anchors_correct"]
                )
                markdown_evidence_total += 1
                markdown_evidence_correct += int(
                    evidence["markdown_evidence_correct"]
                )
            if evidence["order_checked"]:
                order_total += 1
                order_correct += int(evidence["order_correct"])
            fingerprint_total += 1
            fingerprint_correct += int(evidence["fingerprint_correct"])
            parsed_chars = len(parsed_pages.get(page_number).text) if page_exists else 0
            legacy_chars = len(legacy_pages.get(page_number).text) if page_number in legacy_pages else 0
            character_ratios.append(
                parsed_chars / legacy_chars if legacy_chars else 1.0
            )
            predicted_section = any(
                section.page_start == page_number
                and section.title not in {"Front matter", f"Page {page_number}"}
                for section in parsed.sections
            )
            predicted_table = any(
                block.page_number == page_number and block.block_type == "table"
                for section in parsed.sections
                for block in section.blocks
            )
            expected_section = bool(focus["section_boundary"])
            expected_table = bool(focus["table_boundary"])
            section_labels.append((expected_section, predicted_section))
            table_labels.append((expected_table, predicted_table))
            case_pages.append(
                {
                    "page": page_number,
                    **evidence,
                    "character_ratio_vs_legacy": round(
                        character_ratios[-1],
                        4,
                    ),
                    "section_expected": expected_section,
                    "section_predicted": predicted_section,
                    "table_expected": expected_table,
                    "table_predicted": predicted_table,
                }
            )
        cases.append(
            {
                "id": item["id"],
                "split": item["split"],
                "title_correct": title_matches,
                "primary_seconds": round(primary_seconds, 3),
                "legacy_seconds": round(legacy_seconds, 3),
                "latency_ratio": round(latency_ratios[-1], 3),
                "pages": case_pages,
            }
        )

    metrics = {
        "documents": len(documents),
        "focus_pages": page_total,
        "page_number_accuracy": page_correct / page_total if page_total else 0.0,
        "page_anchor_accuracy": (
            anchor_correct / anchor_total if anchor_total else 0.0
        ),
        "markdown_anchor_accuracy": (
            markdown_anchor_correct / markdown_anchor_total
            if markdown_anchor_total
            else 0.0
        ),
        "markdown_page_evidence_accuracy": (
            markdown_evidence_correct / markdown_evidence_total
            if markdown_evidence_total
            else 0.0
        ),
        "reading_order_accuracy": (
            order_correct / order_total if order_total else 0.0
        ),
        "source_fingerprint_accuracy": (
            fingerprint_correct / fingerprint_total
            if fingerprint_total
            else 0.0
        ),
        "median_character_recall_vs_legacy": (
            statistics.median(character_ratios) if character_ratios else 0.0
        ),
        "section_boundary_f1": _binary_f1(section_labels),
        "table_boundary_f1": _binary_f1(table_labels),
        "title_accuracy": title_correct / len(documents) if documents else 0.0,
        "parser_latency_p95_ratio": _percentile(latency_ratios, 0.95),
    }
    gates = {
        "page_number_accuracy": metrics["page_number_accuracy"] == 1.0,
        "page_anchor_accuracy": metrics["page_anchor_accuracy"] == 1.0,
        "markdown_page_evidence_accuracy": (
            metrics["markdown_page_evidence_accuracy"] == 1.0
        ),
        "reading_order_accuracy": metrics["reading_order_accuracy"] == 1.0,
        "source_fingerprint_accuracy": (
            metrics["source_fingerprint_accuracy"] == 1.0
        ),
        "character_recall": (
            metrics["median_character_recall_vs_legacy"] >= 1.0
        ),
        "section_boundary_f1": metrics["section_boundary_f1"] >= 0.8,
        "table_boundary_f1": metrics["table_boundary_f1"] >= 0.75,
        "title_accuracy": metrics["title_accuracy"] >= 0.9,
        "parser_latency": metrics["parser_latency_p95_ratio"] <= 15.0,
    }
    return {
        "passed": all(gates.values()),
        "metrics": {key: round(value, 4) for key, value in metrics.items()},
        "gates": gates,
        "cases": cases,
    }


def evaluate_page_evidence(
    focus: dict[str, Any],
    parsed_pages: dict[int, Any],
) -> dict[str, bool]:
    page_number = int(focus["page"])
    page = parsed_pages.get(page_number)
    page_exists = page is not None
    expected_fingerprint = str(focus.get("source_fingerprint") or "")
    fingerprint_correct = bool(
        page_exists
        and expected_fingerprint
        and page.source_fingerprint == expected_fingerprint
    )
    source_pages = {
        number: str(candidate.source_text or candidate.text)
        for number, candidate in parsed_pages.items()
    }
    normalized_markdown_pages = {
        number: _normalize_evidence(str(candidate.text))
        for number, candidate in parsed_pages.items()
    }
    markdown_pages = {
        number: str(candidate.text)
        for number, candidate in parsed_pages.items()
    }
    current_source = source_pages.get(page_number, "")
    current_markdown = normalized_markdown_pages.get(page_number, "")
    source_anchors = [
        str(anchor)
        for anchor in focus.get("text_anchors", [])
        if str(anchor).strip()
    ]
    markdown_anchors = [
        _normalize_evidence(str(anchor))
        for anchor in focus.get("text_anchors", [])
        if str(anchor).strip()
    ]
    source_anchor_positions = {
        anchor: {
            number: _source_anchor_positions(anchor, text)
            for number, text in source_pages.items()
        }
        for anchor in source_anchors
    }
    anchors_correct = bool(source_anchors) and all(
        bool(source_anchor_positions[anchor].get(page_number, []))
        and sum(
            bool(positions)
            for positions in source_anchor_positions[anchor].values()
        )
        == 1
        for anchor in source_anchors
    )
    markdown_anchors_correct = bool(markdown_anchors) and all(
        anchor in current_markdown
        and sum(
            anchor in text for text in normalized_markdown_pages.values()
        )
        == 1
        for anchor in markdown_anchors
    )
    markdown_anchor_pages = {
        anchor: _markdown_anchor_pages(anchor, markdown_pages)
        for anchor in source_anchors
    }
    markdown_evidence_correct = bool(source_anchors) and any(
        markdown_anchor_pages[anchor] == {page_number}
        for anchor in source_anchors
    )
    ordered_anchors = [
        str(anchor)
        for anchor in focus.get("ordered_anchors", [])
        if str(anchor).strip()
    ]
    ordered_positions = [
        _source_anchor_positions(anchor, current_source)
        for anchor in ordered_anchors
    ]
    order_correct = bool(ordered_anchors) and _has_increasing_positions(
        ordered_positions
    )
    anchors_checked = bool(source_anchors)
    markdown_anchors_checked = bool(markdown_anchors)
    order_checked = bool(ordered_anchors)
    page_correct = (
        page_exists
        and fingerprint_correct
        and (anchors_correct if anchors_checked else True)
        and (
            markdown_evidence_correct
            if markdown_anchors_checked
            else True
        )
        and (order_correct if order_checked else True)
    )
    return {
        "page_exists": page_exists,
        "page_correct": page_correct,
        "fingerprint_correct": fingerprint_correct,
        "anchors_checked": anchors_checked,
        "anchors_correct": anchors_correct,
        "markdown_anchors_checked": markdown_anchors_checked,
        "markdown_anchors_correct": markdown_anchors_correct,
        "markdown_evidence_correct": markdown_evidence_correct,
        "order_checked": order_checked,
        "order_correct": order_correct,
    }


def _binary_f1(labels: list[tuple[bool, bool]]) -> float:
    true_positive = sum(expected and predicted for expected, predicted in labels)
    false_positive = sum(
        not expected and predicted for expected, predicted in labels
    )
    false_negative = sum(
        expected and not predicted for expected, predicted in labels
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else 1.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _normalize_title(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _normalize_evidence(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character.casefold()
        for character in decomposed
        if character.isalnum() and not unicodedata.combining(character)
    )


def _markdown_anchor_pages(
    anchor: str,
    pages: dict[int, str],
) -> set[int]:
    normalized_anchor = _normalize_evidence(anchor)
    exact_pages = {
        number
        for number, text in pages.items()
        if normalized_anchor
        and normalized_anchor in _normalize_evidence(text)
    }
    if exact_pages:
        return exact_pages
    anchor_tokens = set(_compatible_tokens(anchor))
    if not anchor_tokens:
        return set()
    return {
        number
        for number, text in pages.items()
        if anchor_tokens <= set(_compatible_tokens(text))
    }


def _compatible_tokens(value: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", value)
    return [
        token.casefold()
        for token in re.findall(r"[^\W_]+", decomposed, flags=re.UNICODE)
        if len(token) >= 3
    ]


def _source_anchor_positions(anchor: str, page_text: str) -> list[int]:
    anchor_tokens = _source_tokens(anchor)
    page_tokens = _source_tokens(page_text)
    if not anchor_tokens or not page_tokens:
        return []
    positions: list[int] = []
    for start, token in enumerate(page_tokens):
        if token != anchor_tokens[0]:
            continue
        cursor = start
        matched = True
        for expected in anchor_tokens[1:]:
            upper_bound = min(len(page_tokens), cursor + 5)
            try:
                cursor = page_tokens.index(
                    expected,
                    cursor + 1,
                    upper_bound,
                )
            except ValueError:
                matched = False
                break
        if matched:
            positions.append(start)
    return positions


def _has_increasing_positions(
    candidate_groups: list[list[int]],
) -> bool:
    previous = -1
    for candidates in candidate_groups:
        next_position = next(
            (position for position in candidates if position >= previous),
            None,
        )
        if next_position is None:
            return False
        previous = next_position
    return True


def _source_tokens(value: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
