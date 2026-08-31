"""Timeout-bounded paper parsing, quality fallback, and artifact persistence."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
from pathlib import Path
from time import monotonic
from typing import Any

import pymupdf

from core.settings import AppSettings
from indexing.parsers.legacy_paper_parser import LegacyPaperParser
from indexing.parsers.paper_parser import ParsedPaper
from indexing.parsers.paper_parser import (
    MetadataField,
    NORMALIZATION_VERSION,
    PaperMetadata,
    ParsedPage,
    ParserQuality,
)
from indexing.parsers.parser_quality import assess_parser_quality
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser
from indexing.parsers.structure_normalizer import normalize_structure


class ParserTimeoutError(TimeoutError):
    """Raised when a parser exceeds the configured hard deadline."""


def parse_source(file_path: str, settings: AppSettings) -> ParsedPaper:
    if Path(file_path).suffix.lower() == ".pdf":
        return parse_paper_with_fallback(file_path, settings)
    return _parse_text_source(file_path)


def parse_paper_with_fallback(
    file_path: str,
    settings: AppSettings,
) -> ParsedPaper:
    if Path(file_path).suffix.lower() != ".pdf":
        raise ValueError("Structured paper parsing currently requires a PDF.")
    if settings.paper_parser == "legacy":
        parsed = _parse_pdf_with_timeout("legacy", file_path, settings)
        quality = assess_parser_quality(parsed, parsed)
        parsed.quality = quality
        if quality.needs_ocr:
            parsed.status = "needs_ocr"
            parsed.fallback_reason = "configured_legacy_parser; needs_ocr"
        else:
            parsed.status = "degraded"
            parsed.fallback_reason = "configured_legacy_parser"
        return parsed

    try:
        primary = _parse_pdf_with_timeout("pymupdf4llm", file_path, settings)
    except (ParserTimeoutError, RuntimeError, ValueError) as exc:
        legacy = _parse_pdf_with_timeout("legacy", file_path, settings)
        quality = assess_parser_quality(legacy, legacy)
        legacy.quality = quality
        reason = f"primary_parser_failed: {exc}"
        if quality.needs_ocr:
            legacy.status = "needs_ocr"
            legacy.fallback_reason = f"{reason}; needs_ocr"
        else:
            legacy.status = "degraded"
            legacy.fallback_reason = reason
        return legacy

    legacy = _parse_pdf_with_timeout("legacy", file_path, settings)
    quality = assess_parser_quality(primary, legacy)
    primary.quality = quality
    if quality.needs_ocr:
        legacy_quality = assess_parser_quality(legacy, legacy)
        legacy.quality = legacy_quality
        if not legacy_quality.needs_ocr:
            legacy.status = "degraded"
            reasons = ["primary_needs_ocr", *quality.reasons]
            legacy.fallback_reason = ", ".join(dict.fromkeys(reasons))
            return legacy
        selected = (
            legacy
            if _text_character_count(legacy) > _text_character_count(primary)
            else primary
        )
        selected.status = "needs_ocr"
        selected.fallback_reason = "needs_ocr"
        return selected
    if quality.passed:
        primary.status = "parsed"
        return primary

    legacy.status = "degraded"
    legacy.fallback_reason = ", ".join(quality.reasons)
    legacy.quality = assess_parser_quality(legacy, legacy)
    return legacy


def write_parsed_artifact(
    parsed: ParsedPaper,
    *,
    settings: AppSettings,
    paper_id: str,
    paper_version_id: str,
) -> Path:
    root = (
        settings.parsed_artifact_root or settings.data_dir / "parsed"
    ).resolve()
    target_dir = (root / paper_id).resolve()
    if not target_dir.is_relative_to(root):
        raise ValueError("Parsed artifact path escaped PARSED_ARTIFACT_ROOT.")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{paper_version_id}.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _parse_pdf_with_timeout(
    parser_name: str,
    file_path: str,
    settings: AppSettings,
) -> ParsedPaper:
    return _parse_with_timeout(
        parser_name,
        file_path,
        settings.parser_timeout_seconds,
        long_document_timeout_seconds=settings.long_document_timeout_seconds,
    )


def _parse_with_timeout(
    parser_name: str,
    file_path: str,
    timeout_seconds: int,
    *,
    long_document_timeout_seconds: int | None = None,
) -> ParsedPaper:
    context = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[Any] = context.Queue(maxsize=1)
    process = context.Process(
        target=_parser_process_entry,
        args=(parser_name, file_path, result_queue),
        daemon=True,
    )
    process.start()
    try:
        kind, payload = _get_process_result(
            process,
            result_queue,
            parser_name=parser_name,
            timeout_seconds=timeout_seconds,
            long_document_timeout_seconds=long_document_timeout_seconds,
        )
    finally:
        result_queue.close()
        result_queue.join_thread()
    process.join(5)
    if process.is_alive():
        _terminate_process(process)
        raise RuntimeError(f"{parser_name} did not exit after returning a result.")
    if kind == "error":
        raise RuntimeError(str(payload))
    if not isinstance(payload, ParsedPaper):
        raise RuntimeError(f"{parser_name} returned an invalid parser result.")
    return payload


def _get_process_result(
    process: multiprocessing.Process,
    result_queue: multiprocessing.Queue[Any],
    *,
    parser_name: str,
    timeout_seconds: int,
    long_document_timeout_seconds: int | None = None,
) -> tuple[str, Any]:
    started = monotonic()
    deadline = started + timeout_seconds
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            _terminate_process(process)
            active_timeout = max(1, int(round(deadline - started)))
            raise ParserTimeoutError(
                f"{parser_name} exceeded {active_timeout} seconds."
            )
        try:
            kind, payload = result_queue.get(timeout=min(0.25, remaining))
            if kind == "page_count":
                if (
                    isinstance(payload, int)
                    and payload > 100
                    and long_document_timeout_seconds is not None
                ):
                    deadline = max(
                        deadline,
                        started + long_document_timeout_seconds,
                    )
                continue
            return kind, payload
        except queue.Empty as exc:
            if process.is_alive():
                continue
            process.join()
            raise RuntimeError(
                f"{parser_name} exited without a parser result."
            ) from exc


def _terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def _parser_process_entry(
    parser_name: str,
    file_path: str,
    result_queue: multiprocessing.Queue[Any],
) -> None:
    try:
        with pymupdf.open(file_path) as document:
            page_count = document.page_count
        result_queue.put(("page_count", page_count))
        parser = (
            PyMuPDF4LLMPaperParser()
            if parser_name == "pymupdf4llm"
            else LegacyPaperParser()
        )
        result_queue.put(("ok", parser.parse(file_path)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _text_character_count(parsed: ParsedPaper) -> int:
    return sum(len(page.text.strip()) for page in parsed.pages)


def _parse_text_source(file_path: str) -> ParsedPaper:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("Text source is empty; upload a non-empty .txt or .md file.")
    page = ParsedPage(page_number=1, text=text, source_text=text)
    title = path.stem
    def unknown() -> MetadataField:
        return MetadataField(None, "unknown", 0.0)
    return ParsedPaper(
        source_path=file_path,
        page_count=1,
        parser_name="legacy_text",
        parser_version="text-v1",
        normalization_version=NORMALIZATION_VERSION,
        metadata=PaperMetadata(
            title=MetadataField(title, "filename", 0.45),
            authors=unknown(),
            year=unknown(),
            venue=unknown(),
            doi=unknown(),
            arxiv_id=unknown(),
        ),
        pages=[page],
        sections=normalize_structure([page]),
        quality=ParserQuality(
            passed=True,
            page_coverage=1.0,
            nonempty_page_ratio=1.0 if text else 0.0,
            character_ratio_vs_legacy=1.0,
            page_numbers_monotonic=True,
            needs_ocr=False,
        ),
        status="parsed",
    )
