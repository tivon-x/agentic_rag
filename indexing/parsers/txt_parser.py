from __future__ import annotations

import re
from pathlib import Path

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.parsers.common import (
    build_doc_id,
    build_tree,
    make_node_id,
    normalize_text,
    split_paragraphs,
)


def _looks_like_heading(
    line: str, previous_line: str | None = None, next_line: str | None = None
) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 70:
        return False
    if stripped.endswith((".", "!", "?", "。", "！", "？", ";", "；")):
        return False
    previous_blank = previous_line is None or not previous_line.strip()
    next_blank = next_line is None or not next_line.strip()
    if not (previous_blank or next_blank):
        return False
    if re.match(r"^\d+(\.\d+)*\s+", stripped):
        return True
    if stripped.endswith(":") or stripped.endswith("："):
        return next_blank and bool(re.match(r"^[A-Z][^.!?。！？]{0,60}[:：]$", stripped))
    if stripped.isupper():
        alpha_chars = [char for char in stripped if char.isalpha()]
        return len(alpha_chars) >= 3 and len(stripped.split()) <= 6
    return bool(re.match(r"^[A-Z][A-Za-z0-9 /_-]{0,50}$", stripped))


class TxtHierarchicalParser:
    def parse(self, file_path: str) -> ParsedDocumentTree:
        content = Path(file_path).read_text(encoding="utf-8")
        doc_id = build_doc_id(file_path)
        source_name = Path(file_path).name
        nodes: list[Node] = []
        order = 0

        root_id = make_node_id(doc_id, "document", order)
        root = Node(
            node_id=root_id,
            parent_id=None,
            doc_id=doc_id,
            node_type="document",
            title=Path(file_path).stem,
            text=normalize_text(content),
            order=order,
            level=0,
            metadata={
                "source": file_path,
                "file_name": source_name,
                "source_type": "txt",
                "title_path": [Path(file_path).stem],
                "section_path": [],
            },
        )
        nodes.append(root)
        order += 1

        current_parent = root
        paragraph_buffer: list[str] = []

        def flush_paragraphs() -> None:
            nonlocal order
            for paragraph in split_paragraphs("\n".join(paragraph_buffer)):
                section_path = current_parent.metadata.get("section_path", [])
                title_path = root.metadata["title_path"] + list(section_path)
                nodes.append(
                    Node(
                        node_id=make_node_id(doc_id, "paragraph", order),
                        parent_id=current_parent.node_id,
                        doc_id=doc_id,
                        node_type="paragraph",
                        title=None,
                        text=paragraph,
                        order=order,
                        level=current_parent.level + 1,
                        metadata={
                            "source": file_path,
                            "file_name": source_name,
                            "source_type": "txt",
                            "title_path": title_path,
                            "section_path": list(section_path),
                        },
                    )
                )
                order += 1
            paragraph_buffer.clear()

        lines = content.splitlines()
        for index, line in enumerate(lines):
            previous_line = lines[index - 1] if index > 0 else None
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if _looks_like_heading(line, previous_line, next_line):
                flush_paragraphs()
                heading = line.strip().rstrip(":：")
                section_path = [heading]
                section = Node(
                    node_id=make_node_id(doc_id, "section", order),
                    parent_id=root.node_id,
                    doc_id=doc_id,
                    node_type="section",
                    title=heading,
                    text="",
                    order=order,
                    level=1,
                    metadata={
                        "source": file_path,
                        "file_name": source_name,
                        "source_type": "txt",
                        "title_path": [Path(file_path).stem, heading],
                        "section_path": section_path,
                    },
                )
                nodes.append(section)
                order += 1
                current_parent = section
                continue
            paragraph_buffer.append(line)

        flush_paragraphs()
        return build_tree(doc_id, root_id, nodes)
