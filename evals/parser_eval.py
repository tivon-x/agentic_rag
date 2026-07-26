"""Evaluate parser page evidence, boundary detection, metadata, and latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
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
            page_exists = page_number in parsed_pages
            page_correct += int(page_exists)
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
                    "page_correct": page_exists,
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


if __name__ == "__main__":
    raise SystemExit(main())
