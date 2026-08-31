"""Deterministic adapter and corpus checks for the frozen KITE snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.v2_corpus import sha256_file
from indexing.parsers.legacy_paper_parser import LegacyPaperParser
from indexing.parsers.paper_parser import NORMALIZATION_VERSION
from indexing.parsers.pymupdf4llm_parser import PyMuPDF4LLMPaperParser
from indexing.passages import build_catalog_records


KITE_REPOSITORY = "https://github.com/D-Star-AI/KITE"
KITE_COMMIT = "85e71ad63db9ea410eccbb0158f94e7d72462b99"
KITE_QUERY_PATH = "queries/ai_papers.json"
KITE_CORPUS_PATH = "knowledge-base-content/ai-papers"
KITE_QUERY_SHA256 = "6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3"
KITE_CASE_COUNT = 15
KITE_EMPTY_RUBRIC_COUNT = 6
KITE_CORPUS_FILE_COUNT = 134
KITE_CORPUS_SHA256 = "f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8"
KITE_PARSER_NAME = PyMuPDF4LLMPaperParser.name
KITE_PARSER_VERSION = PyMuPDF4LLMPaperParser.version

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_SECRET_OR_ABSOLUTE_PATH = (
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "sk-",
)


class KiteDataError(ValueError):
    """Raised when a KITE query or corpus violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class KiteCase:
    id: str
    query: str
    reference_answer: str
    rubric: str
    source_index: int


def load_kite_cases(
    query_path: Path,
    *,
    expected_sha256: str | None = KITE_QUERY_SHA256,
    expected_case_count: int | None = KITE_CASE_COUNT,
    expected_empty_rubric_count: int | None = KITE_EMPTY_RUBRIC_COUNT,
) -> list[KiteCase]:
    """Load the frozen AI Papers query file without changing its contents."""
    _require_file(query_path, "KITE query file")
    actual_sha256 = sha256_file(query_path)
    if expected_sha256 and actual_sha256 != expected_sha256.casefold():
        raise KiteDataError(
            "KITE query SHA-256 mismatch "
            f"(expected={expected_sha256}, actual={actual_sha256})."
        )
    try:
        payload = json.loads(query_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KiteDataError(f"Invalid KITE query JSON: {query_path}") from exc
    if not isinstance(payload, list):
        raise KiteDataError("KITE query JSON must contain a list of cases.")

    cases: list[KiteCase] = []
    seen_source_ids: set[str] = set()
    empty_rubrics = 0
    for source_index, raw_case in enumerate(payload):
        if not isinstance(raw_case, dict):
            raise KiteDataError(
                f"KITE case at index {source_index} must be an object."
            )
        for field in ("query", "gt_answer", "rubric"):
            if field not in raw_case:
                raise KiteDataError(
                    f"KITE case at index {source_index} is missing {field}."
                )
        query = raw_case["query"]
        reference_answer = raw_case["gt_answer"]
        rubric = raw_case["rubric"]
        if not isinstance(query, str) or not query.strip():
            raise KiteDataError(
                f"KITE case at index {source_index} has an empty query."
            )
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise KiteDataError(
                f"KITE case at index {source_index} has an empty gt_answer."
            )
        if not isinstance(rubric, str):
            raise KiteDataError(
                f"KITE case at index {source_index} rubric must be a string."
            )
        source_id = raw_case.get("id", raw_case.get("case_id"))
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id.strip():
                raise KiteDataError(
                    f"KITE case at index {source_index} has an invalid id."
                )
            if source_id in seen_source_ids:
                raise KiteDataError(f"Duplicate KITE case id: {source_id}.")
            seen_source_ids.add(source_id)
        if not rubric:
            empty_rubrics += 1
        cases.append(
            KiteCase(
                id=f"ai-papers-{source_index + 1:03d}",
                query=query,
                reference_answer=reference_answer,
                rubric=rubric,
                source_index=source_index,
            )
        )

    if expected_case_count is not None and len(cases) != expected_case_count:
        raise KiteDataError(
            "Unexpected KITE case count "
            f"(expected={expected_case_count}, actual={len(cases)})."
        )
    if (
        expected_empty_rubric_count is not None
        and empty_rubrics != expected_empty_rubric_count
    ):
        raise KiteDataError(
            "Unexpected empty KITE rubric count "
            f"(expected={expected_empty_rubric_count}, actual={empty_rubrics})."
        )
    return cases


def validate_pdf_file(path: Path) -> None:
    """Reject missing, empty, LFS-pointer, and non-PDF files."""
    _require_file(path, "KITE corpus file")
    if path.stat().st_size == 0:
        raise KiteDataError(f"KITE PDF is empty: {path}")
    with path.open("rb") as source:
        header = source.read(256)
    if header.startswith(_LFS_POINTER_PREFIX):
        raise KiteDataError(
            "KITE PDFs are Git LFS objects. Run git lfs pull before "
            f"preparing the benchmark: {path}"
        )
    if not header.startswith(b"%PDF-"):
        raise KiteDataError(f"KITE corpus file is not a PDF: {path}")


def build_corpus_manifest(
    corpus_root: Path,
    *,
    expected_file_count: int | None = KITE_CORPUS_FILE_COUNT,
) -> tuple[list[dict[str, Any]], str]:
    """Validate every PDF and return a sorted per-file and aggregate hash."""
    if not corpus_root.is_dir():
        raise KiteDataError(f"KITE corpus directory does not exist: {corpus_root}")
    files = sorted(
        (
            path
            for path in corpus_root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.relative_to(corpus_root).as_posix(),
    )
    if expected_file_count is not None and len(files) != expected_file_count:
        raise KiteDataError(
            "Unexpected KITE PDF count "
            f"(expected={expected_file_count}, actual={len(files)})."
        )

    manifest: list[dict[str, Any]] = []
    for path in files:
        validate_pdf_file(path)
        relative_path = path.relative_to(corpus_root).as_posix()
        manifest.append(
            {
                "file_name": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    corpus_sha256 = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return manifest, corpus_sha256


def build_kite_manifest(
    *,
    query_path: Path,
    corpus_root: Path,
    upstream_commit: str = KITE_COMMIT,
    expected_query_sha256: str | None = KITE_QUERY_SHA256,
    expected_case_count: int | None = KITE_CASE_COUNT,
    expected_empty_rubric_count: int | None = KITE_EMPTY_RUBRIC_COUNT,
    expected_corpus_file_count: int | None = KITE_CORPUS_FILE_COUNT,
    expected_corpus_sha256: str | None = KITE_CORPUS_SHA256,
) -> dict[str, Any]:
    """Build a path-safe, provenance-only manifest for a KITE checkout."""
    if upstream_commit != KITE_COMMIT:
        raise KiteDataError(
            "This Goal is frozen to KITE commit "
            f"{KITE_COMMIT}; received {upstream_commit}."
        )
    cases = load_kite_cases(
        query_path,
        expected_sha256=expected_query_sha256,
        expected_case_count=expected_case_count,
        expected_empty_rubric_count=expected_empty_rubric_count,
    )
    corpus_manifest, corpus_sha256 = build_corpus_manifest(
        corpus_root,
        expected_file_count=expected_corpus_file_count,
    )
    if expected_corpus_sha256 and corpus_sha256 != expected_corpus_sha256.casefold():
        raise KiteDataError(
            "KITE corpus SHA-256 mismatch "
            f"(expected={expected_corpus_sha256}, actual={corpus_sha256})."
        )
    manifest = {
        "schema_version": 1,
        "benchmark_name": "kite-ai-papers",
        "upstream_repository": KITE_REPOSITORY,
        "upstream_commit": upstream_commit,
        "query_path": KITE_QUERY_PATH,
        "query_sha256": sha256_file(query_path),
        "case_count": len(cases),
        "empty_rubric_count": sum(not case.rubric for case in cases),
        "corpus_root": KITE_CORPUS_PATH,
        "corpus_file_count": len(corpus_manifest),
        "corpus_file_sha256": corpus_sha256,
        "corpus_manifest": corpus_manifest,
        "parser_name": KITE_PARSER_NAME,
        "parser_version": KITE_PARSER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
    }
    validate_manifest_payload(manifest)
    return manifest


def build_kite_parser_artifact(
    settings: Any,
    *,
    corpus_root: Path,
    output_path: Path,
    workers: int = 4,
) -> tuple[dict[str, Any], str]:
    """Parse KITE PDFs with the product parser and persist stable catalog data."""
    corpus_manifest, _ = build_corpus_manifest(
        corpus_root,
        expected_file_count=KITE_CORPUS_FILE_COUNT,
    )
    files = [corpus_root / row["file_name"] for row in corpus_manifest]

    def parse_one(path: Path) -> dict[str, Any]:
        try:
            parsed = PyMuPDF4LLMPaperParser().parse(str(path))
        except Exception as primary_error:
            parsed = LegacyPaperParser().parse(str(path))
            parsed.status = "degraded"
            parsed.fallback_reason = (
                f"primary_parser_failed: {type(primary_error).__name__}: "
                f"{primary_error}"
            )
        paper_id = hashlib.sha256(path.read_bytes()).hexdigest()
        values = parsed.metadata.values()
        evidence = parsed.metadata.evidence()
        version_id, sections, passages = build_catalog_records(
            parsed,
            paper_id=paper_id,
            metadata_values=values,
            metadata_evidence=evidence,
            max_input_chars=settings.embedding_max_input_chars,
        )
        if not passages:
            raise KiteDataError(f"KITE parser produced no passages: {path.name}")
        return {
            "file_name": path.relative_to(corpus_root).as_posix(),
            "file_sha256": sha256_file(path),
            "paper_id": paper_id,
            "paper_version_id": version_id,
            "parser_name": parsed.parser_name,
            "parser_version": parsed.parser_version,
            "normalization_version": parsed.normalization_version,
            "status": parsed.status,
            "fallback_reason": parsed.fallback_reason,
            "metadata_values": values,
            "metadata_evidence": evidence,
            "sections": [asdict(section) for section in sections],
            "passages": [asdict(passage) for passage in passages],
        }

    max_workers = max(1, min(int(workers), os.cpu_count() or 1, 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        papers = list(executor.map(parse_one, files))
    artifact = {
        "schema_version": 1,
        "kind": "kite-parser-artifact",
        "parser_name": KITE_PARSER_NAME,
        "parser_version": KITE_PARSER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "embedding_max_input_chars": settings.embedding_max_input_chars,
        "corpus_manifest": corpus_manifest,
        "papers": papers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return artifact, sha256_file(output_path)


def validate_manifest_payload(manifest: dict[str, Any]) -> None:
    """Check required metadata and reject secrets or local path leakage."""
    required = {
        "schema_version",
        "benchmark_name",
        "upstream_repository",
        "upstream_commit",
        "query_path",
        "query_sha256",
        "case_count",
        "empty_rubric_count",
        "corpus_root",
        "corpus_file_count",
        "corpus_file_sha256",
        "corpus_manifest",
        "parser_name",
        "parser_version",
        "normalization_version",
        "created_at",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise KiteDataError(f"KITE manifest is missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != 1:
        raise KiteDataError("Unsupported KITE manifest schema.")
    if not isinstance(manifest["corpus_manifest"], list):
        raise KiteDataError("KITE corpus_manifest must be a list.")
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).casefold()
    if any(marker in encoded for marker in _SECRET_OR_ABSOLUTE_PATH):
        raise KiteDataError("KITE manifest contains a secret-like value.")
    for key in ("query_path", "corpus_root"):
        value = manifest[key]
        if not isinstance(value, str) or Path(value).is_absolute():
            raise KiteDataError(f"KITE manifest {key} must be a logical path.")
        if "\\" in value or ":" in value:
            raise KiteDataError(f"KITE manifest {key} must use a relative POSIX path.")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise KiteDataError(f"{label} does not exist: {path}")


__all__ = [
    "KITE_CASE_COUNT",
    "KITE_COMMIT",
    "KITE_CORPUS_FILE_COUNT",
    "KITE_CORPUS_SHA256",
    "KITE_CORPUS_PATH",
    "KITE_EMPTY_RUBRIC_COUNT",
    "KITE_PARSER_NAME",
    "KITE_PARSER_VERSION",
    "KITE_QUERY_PATH",
    "KITE_QUERY_SHA256",
    "KITE_REPOSITORY",
    "KiteCase",
    "KiteDataError",
    "build_corpus_manifest",
    "build_kite_manifest",
    "build_kite_parser_artifact",
    "load_kite_cases",
    "validate_manifest_payload",
    "validate_pdf_file",
]
