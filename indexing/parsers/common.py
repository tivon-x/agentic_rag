from __future__ import annotations

import hashlib
import re
from pathlib import Path

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node


def build_doc_id(file_path: str) -> str:
    normalized = str(Path(file_path).resolve()).lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def make_node_id(doc_id: str, node_type: str, order: int) -> str:
    return f"{doc_id}:{node_type}:{order}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    parts = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if parts:
        return parts

    return [normalized]


def section_path_for(node: Node, nodes_by_id: dict[str, Node]) -> list[str]:
    path: list[str] = []
    current = node
    while current.parent_id:
        parent = nodes_by_id.get(current.parent_id)
        if parent is None:
            break
        if parent.node_type == "section" and parent.title:
            path.append(parent.title)
        current = parent
    path.reverse()
    return path


def build_tree(doc_id: str, root_id: str, nodes: list[Node]) -> ParsedDocumentTree:
    children_by_parent: dict[str, list[str]] = {}
    for node in nodes:
        if node.parent_id is None:
            continue
        children_by_parent.setdefault(node.parent_id, []).append(node.node_id)
    return ParsedDocumentTree(
        doc_id=doc_id,
        root_id=root_id,
        nodes=nodes,
        children_by_parent=children_by_parent,
    )
