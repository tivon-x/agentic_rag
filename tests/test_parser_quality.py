from __future__ import annotations

from evals.parser_eval import evaluate_page_evidence
from indexing.parsers.paper_parser import (
    MetadataField,
    PaperMetadata,
    ParsedBlock,
    ParsedPage,
    ParsedPaper,
    ParsedSection,
)
from indexing.parsers.parser_quality import assess_parser_quality
from indexing.passages import build_catalog_records, split_quote_text


def _metadata() -> PaperMetadata:
    empty = MetadataField(None, "unknown", 0.0)
    return PaperMetadata(
        title=MetadataField("Parser quality", "filename", 0.45),
        authors=empty,
        year=empty,
        venue=empty,
        doi=empty,
        arxiv_id=empty,
    )


def _paper(pages: list[ParsedPage], page_count: int) -> ParsedPaper:
    return ParsedPaper(
        source_path="paper.pdf",
        page_count=page_count,
        parser_name="test",
        parser_version="1",
        normalization_version="1",
        metadata=_metadata(),
        pages=pages,
        sections=[],
    )


def test_quality_gate_rejects_missing_pages_and_low_character_recall() -> None:
    parsed = _paper([ParsedPage(1, "short")], page_count=3)
    legacy = _paper(
        [
            ParsedPage(1, "a" * 100),
            ParsedPage(2, "b" * 100),
            ParsedPage(3, "c" * 100),
        ],
        page_count=3,
    )

    quality = assess_parser_quality(parsed, legacy)

    assert not quality.passed
    assert "page_coverage_below_95_percent" in quality.reasons
    assert "character_count_below_60_percent_of_legacy" in quality.reasons


def test_quality_gate_marks_very_low_text_document_needs_ocr() -> None:
    pages = [ParsedPage(1, ""), ParsedPage(2, ""), ParsedPage(3, "x")]
    quality = assess_parser_quality(
        _paper(pages, page_count=3),
        _paper(pages, page_count=3),
    )

    assert quality.needs_ocr
    assert "needs_ocr" in quality.reasons


def test_passage_splitter_enforces_deterministic_character_boundary() -> None:
    text = "Sentence one. " * 900

    first = split_quote_text(text, max_chars=700)
    second = split_quote_text(text, max_chars=700)

    assert first == second
    assert "".join(first).replace(" ", "") == text.strip().replace(" ", "")
    assert max(map(len, first)) <= 700


def test_catalog_passages_include_prefix_within_embedding_hard_limit() -> None:
    paper = _paper([ParsedPage(1, "evidence")], page_count=1)
    paper.sections = [
        ParsedSection(
            title="1 Evidence",
            level=1,
            ordinal=0,
            page_start=1,
            page_end=1,
            heading_path=["1 Evidence"],
            blocks=[
                ParsedBlock(
                    page_number=1,
                    block_type="paragraph",
                    text="Sentence with stable evidence. " * 700,
                )
            ],
        )
    ]
    values = {
        "title": "T" * 300,
        "authors": ["A" * 100 for _ in range(30)],
        "year": 2026,
    }
    evidence = {
        name: {"value": value, "source": "user", "confidence": 1.0}
        for name, value in values.items()
    }

    _, _, passages = build_catalog_records(
        paper,
        paper_id="a" * 64,
        metadata_values=values,
        metadata_evidence=evidence,
        max_input_chars=6000,
    )

    assert len(passages) > 1
    assert max(len(passage.retrieval_text) for passage in passages) <= 6000
    assert all(
        passage.quote_text in passage.retrieval_text for passage in passages
    )


def test_catalog_fails_when_metadata_prefix_exceeds_embedding_limit() -> None:
    paper = _paper([ParsedPage(1, "evidence")], page_count=1)
    paper.sections = [
        ParsedSection(
            title="Evidence",
            level=1,
            ordinal=0,
            page_start=1,
            page_end=1,
            heading_path=["Evidence"],
            blocks=[ParsedBlock(1, "paragraph", "evidence")],
        )
    ]
    values = {"title": "T" * 7000}
    evidence = {
        "title": {
            "value": values["title"],
            "source": "user",
            "confidence": 1.0,
        }
    }

    try:
        build_catalog_records(
            paper,
            paper_id="a" * 64,
            metadata_values=values,
            metadata_evidence=evidence,
            max_input_chars=6000,
        )
    except ValueError as exc:
        assert "Metadata prefix exceeds" in str(exc)
    else:
        raise AssertionError("Expected an oversized metadata prefix failure.")


def test_page_evidence_rejects_expected_page_with_wrong_content() -> None:
    pages = {
        1: ParsedPage(
            page_number=1,
            text="Content copied from another page.",
            source_fingerprint="page-one",
        ),
        2: ParsedPage(
            page_number=2,
            text="The unique amber evidence belongs on page one.",
            source_fingerprint="page-two",
        ),
    }

    evidence = evaluate_page_evidence(
        {
            "page": 1,
            "source_fingerprint": "page-one",
            "text_anchors": ["unique amber evidence"],
        },
        pages,
    )

    assert evidence["page_exists"]
    assert evidence["fingerprint_correct"]
    assert not evidence["anchors_correct"]
    assert not evidence["markdown_evidence_correct"]
    assert not evidence["page_correct"]


def test_page_evidence_rejects_reversed_reading_order() -> None:
    pages = {
        1: ParsedPage(
            page_number=1,
            text="right column evidence then left column evidence",
            source_fingerprint="page-one",
        )
    }

    evidence = evaluate_page_evidence(
        {
            "page": 1,
            "source_fingerprint": "page-one",
            "text_anchors": ["left column evidence", "right column evidence"],
            "ordered_anchors": [
                "left column evidence",
                "right column evidence",
            ],
        },
        pages,
    )

    assert evidence["anchors_correct"]
    assert not evidence["order_correct"]
    assert not evidence["page_correct"]


def test_source_evidence_keeps_ligature_identity_for_page_uniqueness() -> None:
    pages = {
        1: ParsedPage(
            page_number=1,
            text="three challenging geometric problems finding",
            source_fingerprint="page-one",
            source_text="three challenging geometric problems ﬁnding",
        ),
        2: ParsedPage(
            page_number=2,
            text="different geometric evidence",
            source_fingerprint="page-two",
            source_text="three challenging geometric problems finding",
        ),
    }

    evidence = evaluate_page_evidence(
        {
            "page": 1,
            "source_fingerprint": "page-one",
            "text_anchors": [
                "three challenging geometric problems ﬁnding"
            ],
        },
        pages,
    )

    assert evidence["anchors_correct"]
    assert evidence["markdown_evidence_correct"]
    assert evidence["page_correct"]
