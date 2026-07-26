from __future__ import annotations

from copy import deepcopy
import multiprocessing
from queue import Queue
import threading
import time

import pymupdf
import pytest

from core.settings import load_settings
from indexing.paper_ingestion import (
    ParserTimeoutError,
    _get_process_result,
    parse_paper_with_fallback,
)
from indexing.parsers.legacy_paper_parser import LegacyPaperParser
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser
from indexing.passages import build_catalog_records


def _write_structured_pdf(path) -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Deterministic Evidence Parsing", fontsize=22)
    first.insert_text((72, 108), "Ada Researcher", fontsize=12)
    first.insert_text((72, 145), "Abstract", fontsize=16)
    first.insert_text(
        (72, 170),
        "This paper tests stable page evidence and metadata.",
        fontsize=11,
    )
    second = document.new_page()
    second.insert_text((72, 72), "1 Introduction", fontsize=17)
    second.insert_text(
        (72, 105),
        "The retrieval marker is amber-eigenvector.",
        fontsize=11,
    )
    document.set_metadata({"title": "", "author": ""})
    document.save(path)
    document.close()


def _never_returns(result_queue) -> None:
    time.sleep(10)
    result_queue.put(("ok", None))


def _returns_large_result(result_queue) -> None:
    result_queue.put(("ok", "x" * 1_000_000))


def _returns_after_long_page_signal(result_queue) -> None:
    result_queue.put(("page_count", 101))
    time.sleep(1.2)
    result_queue.put(("ok", "extended"))


def test_pymupdf4llm_parser_preserves_page_numbers_and_sections(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    _write_structured_pdf(path)

    parsed = PyMuPDF4LLMPaperParser().parse(str(path))

    assert parsed.page_count == 2
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert any(
        "Introduction" in section.title for section in parsed.sections
    )
    assert parsed.metadata.title.value == "Deterministic Evidence Parsing"
    assert parsed.metadata.title.source == "first_page_heuristic"


def test_stable_section_and_passage_ids_survive_reparse(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    _write_structured_pdf(path)
    parser = PyMuPDF4LLMPaperParser()
    first = parser.parse(str(path))
    second = parser.parse(str(path))
    metadata_values = first.metadata.values()
    metadata_evidence = first.metadata.evidence()

    first_records = build_catalog_records(
        first,
        paper_id="a" * 64,
        metadata_values=metadata_values,
        metadata_evidence=metadata_evidence,
        max_input_chars=6000,
    )
    second_records = build_catalog_records(
        second,
        paper_id="a" * 64,
        metadata_values=metadata_values,
        metadata_evidence=metadata_evidence,
        max_input_chars=6000,
    )

    assert first_records[0] == second_records[0]
    assert [section.id for section in first_records[1]] == [
        section.id for section in second_records[1]
    ]
    assert [passage.id for passage in first_records[2]] == [
        passage.id for passage in second_records[2]
    ]
    assert all(passage.page_start >= 1 for passage in first_records[2])


def test_parser_falls_back_to_legacy_when_primary_fails(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "fallback.pdf"
    _write_structured_pdf(path)
    legacy = LegacyPaperParser().parse(str(path))

    def parse_with_failure(parser_name, _file_path, _timeout, **_kwargs):
        if parser_name == "pymupdf4llm":
            raise RuntimeError("synthetic primary failure")
        return legacy

    monkeypatch.setattr(
        "indexing.paper_ingestion._parse_with_timeout",
        parse_with_failure,
    )
    parsed = parse_paper_with_fallback(
        str(path),
        load_settings(base_dir=tmp_path),
    )

    assert parsed.status == "degraded"
    assert parsed.parser_name == "legacy"
    assert parsed.fallback_reason == (
        "primary_parser_failed: synthetic primary failure"
    )


def test_parser_uses_legacy_when_primary_needs_ocr_but_legacy_has_text(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "fallback.pdf"
    _write_structured_pdf(path)
    legacy = LegacyPaperParser().parse(str(path))
    primary = deepcopy(legacy)
    primary.parser_name = "pymupdf4llm"
    for page in primary.pages:
        page.text = ""

    def parse_with_low_text_primary(
        parser_name,
        _file_path,
        _timeout,
        **_kwargs,
    ):
        return primary if parser_name == "pymupdf4llm" else legacy

    monkeypatch.setattr(
        "indexing.paper_ingestion._parse_with_timeout",
        parse_with_low_text_primary,
    )

    parsed = parse_paper_with_fallback(
        str(path),
        load_settings(base_dir=tmp_path),
    )

    assert parsed.status == "degraded"
    assert parsed.parser_name == "legacy"
    assert "primary_needs_ocr" in str(parsed.fallback_reason)


def test_parser_marks_legacy_needs_ocr_after_primary_failure(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "fallback.pdf"
    _write_structured_pdf(path)
    legacy = LegacyPaperParser().parse(str(path))
    for page in legacy.pages:
        page.text = ""

    def parse_with_failed_primary(
        parser_name,
        _file_path,
        _timeout,
        **_kwargs,
    ):
        if parser_name == "pymupdf4llm":
            raise RuntimeError("synthetic primary failure")
        return legacy

    monkeypatch.setattr(
        "indexing.paper_ingestion._parse_with_timeout",
        parse_with_failed_primary,
    )

    parsed = parse_paper_with_fallback(
        str(path),
        load_settings(base_dir=tmp_path),
    )

    assert parsed.status == "needs_ocr"
    assert parsed.parser_name == "legacy"
    assert "primary_parser_failed" in str(parsed.fallback_reason)
    assert "needs_ocr" in str(parsed.fallback_reason)


def test_parent_process_does_not_open_pdf_before_timeout_guard(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "guarded.pdf"
    _write_structured_pdf(path)
    parsed = LegacyPaperParser().parse(str(path))

    def forbidden_parent_open(*_args, **_kwargs):
        raise AssertionError("PDF opened outside the supervised parser process")

    def parse_without_child(_parser_name, _file_path, _timeout, **_kwargs):
        return deepcopy(parsed)

    monkeypatch.setattr(
        "indexing.paper_ingestion._parse_with_timeout",
        parse_without_child,
    )
    monkeypatch.setattr(
        "indexing.paper_ingestion.pymupdf.open",
        forbidden_parent_open,
    )

    result = parse_paper_with_fallback(
        str(path),
        load_settings(base_dir=tmp_path),
    )

    assert result.status in {"parsed", "degraded"}


def test_parser_timeout_terminates_child_process() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_never_returns, args=(result_queue,))
    process.start()

    with pytest.raises(ParserTimeoutError, match="exceeded 1 seconds"):
        _get_process_result(
            process,
            result_queue,
            parser_name="slow",
            timeout_seconds=1,
        )

    result_queue.close()
    assert not process.is_alive()


def test_parser_result_reader_drains_large_payload_before_join() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_returns_large_result, args=(result_queue,))
    process.start()

    kind, payload = _get_process_result(
        process,
        result_queue,
        parser_name="large",
        timeout_seconds=5,
    )
    process.join(5)

    result_queue.close()
    result_queue.join_thread()
    assert kind == "ok"
    assert len(payload) == 1_000_000
    assert not process.is_alive()


def test_long_document_signal_extends_total_parser_deadline() -> None:
    result_queue = Queue(maxsize=1)
    process = threading.Thread(
        target=_returns_after_long_page_signal,
        args=(result_queue,),
    )
    process.start()

    kind, payload = _get_process_result(
        process,
        result_queue,
        parser_name="long",
        timeout_seconds=1,
        long_document_timeout_seconds=3,
    )
    process.join(3)

    assert (kind, payload) == ("ok", "extended")
    assert not process.is_alive()
