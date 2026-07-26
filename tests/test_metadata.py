from __future__ import annotations

import asyncio
import json
import sqlite3

import pymupdf

from api.db.database import create_indexing_job
from api.db.papers import (
    resolve_effective_metadata,
    save_parsed_catalog,
    update_paper_metadata,
)
from core.settings import load_settings
from indexing.parsers.metadata import extract_pdf_metadata
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser
from indexing.passages import build_catalog_records


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "APP_DB_PATH",
        str(tmp_path / "data" / "api" / "sessions.db"),
    )
    monkeypatch.setenv("EMBEDDING_MAX_INPUT_CHARS", "6000")
    return load_settings(
        base_dir=tmp_path,
        env_file=tmp_path / "missing.env",
    )


def _write_bad_metadata_pdf(path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Trust the Visible Paper Title", fontsize=24)
    page.insert_text((72, 125), "Abstract", fontsize=15)
    page.insert_text(
        (72, 155),
        "The paper deliberately has no identifiable author line.",
        fontsize=11,
    )
    document.set_metadata({"title": "Microsoft Word", "author": ""})
    document.save(path)
    document.close()


def _write_arxiv_year_pdf(path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Reliable Publication Year", fontsize=24)
    page.insert_text((72, 112), "Ada Researcher", fontsize=12)
    page.insert_text((72, 150), "arXiv:1706.03762v7", fontsize=10)
    page.insert_text((72, 180), "ImageNet 2014 is discussed below.", fontsize=10)
    document.set_metadata({"title": "", "author": ""})
    document.save(path)
    document.close()


def test_suspicious_pdf_title_does_not_override_first_page_title(tmp_path) -> None:
    path = tmp_path / "wrong-title.pdf"
    _write_bad_metadata_pdf(path)

    metadata = extract_pdf_metadata(str(path))

    assert metadata.title.value == "Trust the Visible Paper Title"
    assert metadata.title.source == "first_page_heuristic"
    assert metadata.authors.value == []


def test_arxiv_identifier_beats_unrelated_body_year(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    _write_arxiv_year_pdf(path)

    metadata = extract_pdf_metadata(str(path))

    assert metadata.year.value == 2017
    assert metadata.year.source == "first_page_heuristic"


def test_user_title_refreshes_retrieval_text_without_changing_quote(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    source = settings.upload_root / "jobs" / "job" / "paper.pdf"
    source.parent.mkdir(parents=True)
    _write_bad_metadata_pdf(source)
    paper_id = "b" * 64

    asyncio.run(
        create_indexing_job(
            settings,
            job_id="job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
            items=[
                {
                    "filename": "paper.pdf",
                    "source_path": str(source),
                    "paper_id": paper_id,
                    "content_hash": paper_id,
                    "size_bytes": source.stat().st_size,
                    "source_type": "application/pdf",
                }
            ],
        )
    )
    parsed = PyMuPDF4LLMPaperParser().parse(str(source))
    values, evidence, _ = asyncio.run(
        resolve_effective_metadata(
            settings,
            paper_id=paper_id,
            parsed=parsed,
        )
    )
    version_id, sections, passages = build_catalog_records(
        parsed,
        paper_id=paper_id,
        metadata_values=values,
        metadata_evidence=evidence,
        max_input_chars=6000,
    )
    asyncio.run(
        save_parsed_catalog(
            settings,
            paper_id=paper_id,
            version_id=version_id,
            parsed=parsed,
            artifact_path=str(tmp_path / "artifact.json"),
            sections=sections,
            passages=passages,
        )
    )
    with sqlite3.connect(settings.app_db_path) as db:
        before = db.execute(
            """
            SELECT quote_text, retrieval_text
            FROM passages
            WHERE paper_version_id = ?
            ORDER BY ordinal
            LIMIT 1
            """,
            (version_id,),
        ).fetchone()

    asyncio.run(
        update_paper_metadata(
            settings,
            paper_id=paper_id,
            expected_version=1,
            updates={"title": "Corrected Research Title"},
        )
    )
    with sqlite3.connect(settings.app_db_path) as db:
        after = db.execute(
            """
            SELECT quote_text, retrieval_text
            FROM passages
            WHERE paper_version_id = ?
            ORDER BY ordinal
            LIMIT 1
            """,
            (version_id,),
        ).fetchone()
        metadata_json = json.loads(
            db.execute(
                "SELECT metadata_json FROM papers WHERE id = ?",
                (paper_id,),
            ).fetchone()[0]
        )

    assert before[0] == after[0]
    assert "Corrected Research Title" in after[1]
    assert "Corrected Research Title" not in after[0]
    assert metadata_json["title"]["source"] == "user"
