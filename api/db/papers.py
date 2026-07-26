"""Paper catalog repository and passage materialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from api.db.database import get_db
from core.settings import AppSettings
from indexing.parsers.paper_parser import ParsedPaper
from indexing.passages import PassageRecord, SectionRecord, build_retrieval_prefix


class PaperVersionConflictError(ValueError):
    """Raised when If-Match does not match the current metadata version."""


async def mark_paper_parsing(
    settings: AppSettings,
    *,
    paper_id: str,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            UPDATE papers
            SET parse_status = 'parsing', parse_error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (datetime.now(UTC).isoformat(), paper_id),
        )
        await db.commit()


async def resolve_effective_metadata(
    settings: AppSettings,
    *,
    paper_id: str,
    parsed: ParsedPaper,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    async with get_db(settings) as db:
        cursor = await db.execute(
            """
            SELECT title, authors_json, year, venue, doi, arxiv_id,
                   metadata_json, metadata_status
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"Paper {paper_id} does not exist.")
        if str(row["metadata_status"]) == "verified":
            values = _metadata_values_from_row(row)
            evidence = _json_object(row["metadata_json"])
            return values, evidence, "verified"

        values = parsed.metadata.values()
        evidence = parsed.metadata.evidence()
        status = (
            "needs_review"
            if any(
                not item.get("value") or float(item.get("confidence") or 0) < 0.8
                for item in evidence.values()
            )
            else "verified"
        )
        await db.execute(
            """
            UPDATE papers
            SET title = ?, authors_json = ?, year = ?, venue = ?, doi = ?,
                arxiv_id = ?, metadata_json = ?, metadata_status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                values.get("title"),
                json.dumps(values.get("authors") or [], ensure_ascii=False),
                values.get("year"),
                values.get("venue"),
                values.get("doi"),
                values.get("arxiv_id"),
                json.dumps(evidence, ensure_ascii=False),
                status,
                datetime.now(UTC).isoformat(),
                paper_id,
            ),
        )
        await db.commit()
    return values, evidence, status


async def save_parsed_catalog(
    settings: AppSettings,
    *,
    paper_id: str,
    version_id: str,
    parsed: ParsedPaper,
    artifact_path: str,
    sections: list[SectionRecord],
    passages: list[PassageRecord],
) -> None:
    now = datetime.now(UTC).isoformat()
    quality = parsed.quality
    quality_json = json.dumps(
        {
            "passed": quality.passed if quality else True,
            "page_coverage": quality.page_coverage if quality else 1.0,
            "nonempty_page_ratio": (
                quality.nonempty_page_ratio if quality else 1.0
            ),
            "character_ratio_vs_legacy": (
                quality.character_ratio_vs_legacy if quality else 1.0
            ),
            "page_numbers_monotonic": (
                quality.page_numbers_monotonic if quality else True
            ),
            "needs_ocr": quality.needs_ocr if quality else False,
            "reasons": quality.reasons if quality else [],
        },
        ensure_ascii=False,
    )
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            "DELETE FROM paper_versions WHERE id = ?",
            (version_id,),
        )
        await db.execute(
            """
            INSERT INTO paper_versions (
                id, paper_id, parser_name, parser_version, normalization_version,
                source_path, parsed_artifact_path, status, fallback_reason,
                quality_json, page_count, duration_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                paper_id,
                parsed.parser_name,
                parsed.parser_version,
                parsed.normalization_version,
                parsed.source_path,
                artifact_path,
                parsed.status,
                parsed.fallback_reason,
                quality_json,
                parsed.page_count,
                parsed.duration_ms,
                now,
            ),
        )
        for section in sections:
            await db.execute(
                """
                INSERT INTO sections (
                    id, paper_version_id, parent_id, title, level, ordinal,
                    page_start, page_end, heading_path_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section.id,
                    version_id,
                    section.parent_id,
                    section.title,
                    section.level,
                    section.ordinal,
                    section.page_start,
                    section.page_end,
                    json.dumps(section.heading_path, ensure_ascii=False),
                ),
            )
        for passage in passages:
            await db.execute(
                """
                INSERT INTO passages (
                    id, paper_version_id, section_id, page_start, page_end,
                    quote_text, retrieval_text, block_type, ordinal
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    passage.id,
                    version_id,
                    passage.section_id,
                    passage.page_start,
                    passage.page_end,
                    passage.quote_text,
                    passage.retrieval_text,
                    passage.block_type,
                    passage.ordinal,
                ),
            )
        await db.execute(
            """
            UPDATE papers
            SET parse_status = ?, parse_error = NULL, fallback_reason = ?,
                latest_version_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                parsed.status,
                parsed.fallback_reason,
                version_id,
                now,
                paper_id,
            ),
        )
        await db.commit()


async def mark_paper_failed(
    settings: AppSettings,
    *,
    paper_id: str,
    error: str,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            UPDATE papers
            SET parse_status = 'failed', parse_error = ?, fallback_reason = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (error, error, datetime.now(UTC).isoformat(), paper_id),
        )
        await db.commit()


async def list_papers(
    settings: AppSettings,
    *,
    query: str | None = None,
    parse_status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = ["archived_at IS NULL"]
    parameters: list[Any] = []
    if query:
        clauses.append(
            "(title LIKE ? OR file_name LIKE ? OR authors_json LIKE ?)"
        )
        pattern = f"%{query.strip()}%"
        parameters.extend([pattern, pattern, pattern])
    if parse_status:
        clauses.append("parse_status = ?")
        parameters.append(parse_status)
    where = " AND ".join(clauses)
    async with get_db(settings) as db:
        count_cursor = await db.execute(
            f"SELECT COUNT(*) AS count FROM papers WHERE {where}",
            parameters,
        )
        count_row = await count_cursor.fetchone()
        cursor = await db.execute(
            f"""
            SELECT *
            FROM papers
            WHERE {where}
            ORDER BY updated_at DESC, id
            LIMIT ? OFFSET ?
            """,
            [*parameters, limit, offset],
        )
        rows = await cursor.fetchall()
    return {
        "items": [_paper_from_row(row, include_sections=False) for row in rows],
        "total": int(count_row["count"]) if count_row else 0,
        "limit": limit,
        "offset": offset,
    }


async def get_paper(
    settings: AppSettings,
    *,
    paper_id: str,
) -> dict[str, Any] | None:
    async with get_db(settings) as db:
        cursor = await db.execute(
            "SELECT * FROM papers WHERE id = ? AND archived_at IS NULL",
            (paper_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        version = None
        sections: list[dict[str, Any]] = []
        if row["latest_version_id"]:
            version_cursor = await db.execute(
                "SELECT * FROM paper_versions WHERE id = ?",
                (row["latest_version_id"],),
            )
            version_row = await version_cursor.fetchone()
            if version_row is not None:
                version = _paper_version_from_row(version_row)
            section_cursor = await db.execute(
                """
                SELECT id, parent_id, title, level, ordinal, page_start, page_end,
                       heading_path_json
                FROM sections
                WHERE paper_version_id = ?
                ORDER BY ordinal
                """,
                (row["latest_version_id"],),
            )
            section_rows = await section_cursor.fetchall()
            sections = [
                {
                    "id": str(section["id"]),
                    "parent_id": (
                        str(section["parent_id"])
                        if section["parent_id"]
                        else None
                    ),
                    "title": str(section["title"]),
                    "level": int(section["level"]),
                    "ordinal": int(section["ordinal"]),
                    "page_start": int(section["page_start"]),
                    "page_end": int(section["page_end"]),
                    "heading_path": _json_list(section["heading_path_json"]),
                }
                for section in section_rows
            ]
    payload = _paper_from_row(row, include_sections=True)
    payload["paper_version"] = version
    payload["sections"] = sections
    return payload


async def update_paper_metadata(
    settings: AppSettings,
    *,
    paper_id: str,
    expected_version: int,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    now = datetime.now(UTC).isoformat()
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            "SELECT * FROM papers WHERE id = ? AND archived_at IS NULL",
            (paper_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.rollback()
            return None
        current_version = int(row["metadata_version"])
        if current_version != expected_version:
            await db.rollback()
            raise PaperVersionConflictError(
                f"Paper metadata version is {current_version}, not "
                f"{expected_version}."
            )
        values = _metadata_values_from_row(row)
        evidence = _json_object(row["metadata_json"])
        for name, value in updates.items():
            values[name] = value
            evidence[name] = {
                "value": value,
                "source": "user",
                "confidence": 1.0,
            }

        if row["latest_version_id"]:
            passage_cursor = await db.execute(
                """
                SELECT p.id, p.quote_text, p.block_type, s.heading_path_json
                FROM passages AS p
                JOIN sections AS s ON s.id = p.section_id
                WHERE p.paper_version_id = ?
                """,
                (row["latest_version_id"],),
            )
            passage_rows = await passage_cursor.fetchall()
            refreshed: list[tuple[str, str]] = []
            for passage in passage_rows:
                heading_path = _json_list(passage["heading_path_json"])
                prefix = build_retrieval_prefix(
                    values,
                    evidence,
                    heading_path=heading_path,
                )
                retrieval_text = (
                    f"{prefix}[BLOCK] {passage['block_type']}\n"
                    f"{passage['quote_text']}"
                )
                if len(retrieval_text) > settings.embedding_max_input_chars:
                    await db.rollback()
                    raise ValueError(
                        "Corrected metadata would make a passage exceed "
                        "EMBEDDING_MAX_INPUT_CHARS; shorten the metadata."
                    )
                refreshed.append((retrieval_text, str(passage["id"])))
            await db.executemany(
                "UPDATE passages SET retrieval_text = ? WHERE id = ?",
                refreshed,
            )

        await db.execute(
            """
            UPDATE papers
            SET title = ?, authors_json = ?, year = ?, venue = ?, doi = ?,
                arxiv_id = ?, metadata_json = ?, metadata_status = 'verified',
                metadata_version = metadata_version + 1, updated_at = ?
            WHERE id = ? AND metadata_version = ?
            """,
            (
                values.get("title"),
                json.dumps(values.get("authors") or [], ensure_ascii=False),
                values.get("year"),
                values.get("venue"),
                values.get("doi"),
                values.get("arxiv_id"),
                json.dumps(evidence, ensure_ascii=False),
                now,
                paper_id,
                expected_version,
            ),
        )
        await db.commit()
    return await get_paper(settings, paper_id=paper_id)


async def create_metadata_reindex_job(
    settings: AppSettings,
    *,
    paper_id: str,
) -> str:
    job_id = uuid4().hex
    now = datetime.now(UTC).isoformat()
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO indexing_jobs (
                id, status, created_at, updated_at, request_json, max_attempts
            )
            VALUES (?, 'queued', ?, ?, ?, ?)
            """,
            (
                job_id,
                now,
                now,
                json.dumps(
                    {"kind": "metadata_reindex", "paper_id": paper_id},
                    ensure_ascii=False,
                ),
                settings.index_worker_max_attempts,
            ),
        )
        await db.commit()
    return job_id


async def list_catalog_documents(settings: AppSettings) -> list[Document]:
    async with get_db(settings) as db:
        cursor = await db.execute(
            """
            SELECT
                p.id AS passage_id,
                p.page_start,
                p.page_end,
                p.quote_text,
                p.retrieval_text,
                p.block_type,
                p.ordinal,
                s.id AS section_id,
                s.title AS section_title,
                s.heading_path_json,
                papers.id AS paper_id,
                papers.title AS paper_title,
                papers.authors_json,
                papers.year,
                papers.file_name
            FROM papers
            JOIN passages AS p
                ON p.paper_version_id = papers.latest_version_id
            JOIN sections AS s ON s.id = p.section_id
            WHERE papers.archived_at IS NULL
              AND papers.parse_status IN ('parsed', 'degraded')
            ORDER BY papers.created_at, p.ordinal
            """
        )
        rows = await cursor.fetchall()
    return [
        Document(
            page_content=str(row["retrieval_text"]),
            metadata={
                "node_id": str(row["passage_id"]),
                "passage_id": str(row["passage_id"]),
                "paper_id": str(row["paper_id"]),
                "paper_title": row["paper_title"],
                "authors": _json_list(row["authors_json"]),
                "year": row["year"],
                "section_id": str(row["section_id"]),
                "section_title": str(row["section_title"]),
                "heading_path": _json_list(row["heading_path_json"]),
                "page": int(row["page_start"]),
                "page_start": int(row["page_start"]),
                "page_end": int(row["page_end"]),
                "quote_text": str(row["quote_text"]),
                "block_type": str(row["block_type"]),
                "order": int(row["ordinal"]),
                "node_type": "paragraph",
                "source": str(row["file_name"]),
            },
        )
        for row in rows
    ]


def _paper_from_row(row: Any, *, include_sections: bool) -> dict[str, Any]:
    del include_sections
    return {
        "id": str(row["id"]),
        "content_hash": str(row["content_hash"]),
        "file_name": str(row["file_name"]),
        "source_type": str(row["source_type"]),
        "size_bytes": int(row["size_bytes"]),
        "title": row["title"],
        "authors": _json_list(row["authors_json"]),
        "year": row["year"],
        "venue": row["venue"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
        "metadata": _json_object(row["metadata_json"]),
        "metadata_status": str(row["metadata_status"]),
        "metadata_version": int(row["metadata_version"]),
        "parse_status": str(row["parse_status"]),
        "parse_error": row["parse_error"],
        "fallback_reason": row["fallback_reason"],
        "latest_version_id": row["latest_version_id"],
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "file_url": f"/api/papers/{row['id']}/file",
    }


def _paper_version_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "parser_name": str(row["parser_name"]),
        "parser_version": str(row["parser_version"]),
        "normalization_version": str(row["normalization_version"]),
        "status": str(row["status"]),
        "fallback_reason": row["fallback_reason"],
        "quality": _json_object(row["quality_json"]),
        "page_count": int(row["page_count"]),
        "duration_ms": int(row["duration_ms"]),
        "created_at": str(row["created_at"]),
    }


def _metadata_values_from_row(row: Any) -> dict[str, Any]:
    return {
        "title": row["title"],
        "authors": _json_list(row["authors_json"]),
        "year": row["year"],
        "venue": row["venue"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
    }


def _json_object(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_list(value: Any) -> list[Any]:
    try:
        payload = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return payload if isinstance(payload, list) else []
