"""Frozen parser artifact construction for V2 retrieval evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from core.settings import AppSettings, load_settings
from indexing.paper_ingestion import parse_source
from indexing.parsers.paper_parser import paper_id_for_file
from indexing.passages import build_catalog_records


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
ARTIFACT_SCHEMA_VERSION = 1


def build_parser_artifact(
    settings: AppSettings,
    *,
    corpus_dir: Path,
    output_path: Path,
    parser_gold_path: Path,
) -> tuple[dict[str, Any], str]:
    """Parse the frozen corpus once and persist deterministic passage records."""
    files = sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported evaluation files found in {corpus_dir}.")

    papers: list[dict[str, Any]] = []
    corpus_manifest: list[dict[str, str]] = []
    for path in files:
        parsed = parse_source(str(path), settings)
        if parsed.status == "needs_ocr":
            raise ValueError(
                f"Frozen parser artifact cannot include needs_ocr: {path.name}."
            )
        paper_id = paper_id_for_file(path)
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
            raise ValueError(f"Parser produced no passages for {path.name}.")
        file_sha256 = sha256_file(path)
        corpus_manifest.append(
            {"file_name": path.name, "sha256": file_sha256}
        )
        papers.append(
            {
                "file_name": path.name,
                "file_sha256": file_sha256,
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
        )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "parser_gold": {
            "path": parser_gold_path.as_posix(),
            "sha256": sha256_file(parser_gold_path),
        },
        "embedding_max_input_chars": settings.embedding_max_input_chars,
        "corpus_manifest": corpus_manifest,
        "papers": papers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact, sha256_file(output_path)


def load_parser_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    corpus_dir: Path | None = None,
) -> tuple[dict[str, Any], str]:
    artifact_sha256 = sha256_file(path)
    if expected_sha256 and artifact_sha256 != expected_sha256.casefold():
        raise ValueError(
            "Parser artifact checksum mismatch "
            f"(expected={expected_sha256}, actual={artifact_sha256})."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported parser artifact schema.")
    if corpus_dir is not None:
        _validate_corpus_manifest(payload, corpus_dir)
    return payload, artifact_sha256


def artifact_documents(artifact: dict[str, Any]) -> list[Document]:
    documents: list[Document] = []
    for paper in artifact.get("papers", []):
        sections = {
            str(section["id"]): section
            for section in paper.get("sections", [])
        }
        metadata_values = paper.get("metadata_values", {})
        for passage in paper.get("passages", []):
            section = sections[str(passage["section_id"])]
            documents.append(
                Document(
                    page_content=str(passage["retrieval_text"]),
                    metadata={
                        "node_id": str(passage["id"]),
                        "passage_id": str(passage["id"]),
                        "paper_id": str(paper["paper_id"]),
                        "paper_title": metadata_values.get("title"),
                        "authors": metadata_values.get("authors") or [],
                        "year": metadata_values.get("year"),
                        "section_id": str(section["id"]),
                        "section_title": str(section["title"]),
                        "heading_path": list(section["heading_path"]),
                        "section_path": list(section["heading_path"]),
                        "page": int(passage["page_start"]),
                        "page_start": int(passage["page_start"]),
                        "page_end": int(passage["page_end"]),
                        "quote_text": str(passage["quote_text"]),
                        "retrieval_text": str(passage["retrieval_text"]),
                        "block_type": str(passage["block_type"]),
                        "order": int(passage["ordinal"]),
                        "node_type": "paragraph",
                        "source": str(paper["file_name"]),
                    },
                )
            )
    return documents


def artifact_id_sets(
    artifact: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    paper_ids: set[str] = set()
    section_ids: set[str] = set()
    passage_ids: set[str] = set()
    for paper in artifact.get("papers", []):
        paper_ids.add(str(paper["paper_id"]))
        section_ids.update(
            str(section["id"]) for section in paper.get("sections", [])
        )
        passage_ids.update(
            str(passage["id"]) for passage in paper.get("passages", [])
        )
    return paper_ids, section_ids, passage_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_corpus_manifest(
    artifact: dict[str, Any],
    corpus_dir: Path,
) -> None:
    expected = {
        str(row["file_name"]): str(row["sha256"])
        for row in artifact.get("corpus_manifest", [])
    }
    actual_paths = sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )
    actual = {path.name: sha256_file(path) for path in actual_paths}
    if actual != expected:
        raise ValueError(
            "Evaluation corpus differs from the frozen parser artifact."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="evals")
    parser.add_argument(
        "--output",
        default="artifacts/evals/v2_core/parser_artifact.json",
    )
    parser.add_argument(
        "--parser-gold",
        default="evals/datasets/parser_v2.json",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    loaded_settings = load_settings(base_dir=repo_root)
    artifact, artifact_sha256 = build_parser_artifact(
        loaded_settings,
        corpus_dir=(repo_root / args.corpus).resolve(),
        output_path=(repo_root / args.output).resolve(),
        parser_gold_path=(repo_root / args.parser_gold).resolve(),
    )
    print(
        json.dumps(
            {
                "papers": len(artifact["papers"]),
                "passages": sum(
                    len(paper["passages"])
                    for paper in artifact["papers"]
                ),
                "sha256": artifact_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
