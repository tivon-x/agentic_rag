from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_PROFILE = {
    "name": "",
    "summary": "",
    "coverage": "",
    "usage_notes": "",
    "source_examples": [],
}



def corpus_profile_path(index_dir: Path) -> Path:
    return index_dir / "corpus_profile.json"



def load_corpus_profile(index_dir: Path) -> dict[str, Any]:
    path = corpus_profile_path(index_dir)
    if not path.exists():
        return dict(DEFAULT_CORPUS_PROFILE)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_CORPUS_PROFILE)

    merged = dict(DEFAULT_CORPUS_PROFILE)
    if isinstance(data, dict):
        merged.update(data)
    return merged



def save_corpus_profile(
    index_dir: Path,
    *,
    name: str,
    summary: str,
    coverage: str,
    usage_notes: str,
    source_examples: list[str],
) -> Path:
    path = corpus_profile_path(index_dir)
    payload = {
        "name": name.strip(),
        "summary": summary.strip(),
        "coverage": coverage.strip(),
        "usage_notes": usage_notes.strip(),
        "source_examples": source_examples[:10],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path



def format_corpus_profile(profile: dict[str, Any]) -> str:
    name = str(profile.get("name", "")).strip() or "未命名知识库"
    summary = str(profile.get("summary", "")).strip() or "未填写"
    coverage = str(profile.get("coverage", "")).strip() or "未填写"
    usage_notes = str(profile.get("usage_notes", "")).strip() or "未填写"
    source_examples = profile.get("source_examples", []) or []

    lines = [
        f"知识库名称: {name}",
        f"内容摘要: {summary}",
        f"覆盖范围: {coverage}",
        f"使用说明: {usage_notes}",
    ]
    if source_examples:
        lines.append("代表性文件: " + ", ".join(source_examples))
    else:
        lines.append("代表性文件: 暂无")
    return "\n".join(lines)



def build_corpus_profile_context(profile: dict[str, Any]) -> str:
    name = str(profile.get("name", "")).strip()
    summary = str(profile.get("summary", "")).strip()
    coverage = str(profile.get("coverage", "")).strip()
    usage_notes = str(profile.get("usage_notes", "")).strip()
    source_examples = [str(item).strip() for item in profile.get("source_examples", []) or []]
    source_examples = [item for item in source_examples if item]

    if not any([name, summary, coverage, usage_notes, source_examples]):
        return "No corpus profile is available."

    lines = [
        f"Knowledge Base Name: {name or 'Unnamed knowledge base'}",
        f"Summary: {summary or 'Not provided'}",
        f"Coverage: {coverage or 'Not provided'}",
        f"Usage Notes: {usage_notes or 'Not provided'}",
        "Representative Sources: " + (", ".join(source_examples[:10]) if source_examples else "None listed"),
    ]
    return "\n".join(lines)
