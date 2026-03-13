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


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownHierarchicalParser:
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
                "source_type": "md",
                "title_path": [Path(file_path).stem],
                "section_path": [],
            },
        )
        nodes.append(root)
        order += 1

        section_stack: list[Node] = [root]
        buffer: list[str] = []

        def flush_paragraphs() -> None:
            nonlocal order
            paragraph_parent = section_stack[-1]
            for paragraph in split_paragraphs("\n".join(buffer)):
                section_path = paragraph_parent.metadata.get("section_path", [])
                title_path = root.metadata["title_path"] + list(section_path)
                nodes.append(
                    Node(
                        node_id=make_node_id(doc_id, "paragraph", order),
                        parent_id=paragraph_parent.node_id,
                        doc_id=doc_id,
                        node_type="paragraph",
                        title=None,
                        text=paragraph,
                        order=order,
                        level=paragraph_parent.level + 1,
                        metadata={
                            "source": file_path,
                            "file_name": source_name,
                            "source_type": "md",
                            "title_path": title_path,
                            "section_path": list(section_path),
                        },
                    )
                )
                order += 1
            buffer.clear()

        for line in content.splitlines():
            match = _HEADING_RE.match(line.strip())
            if match:
                flush_paragraphs()
                hashes, heading_text = match.groups()
                level = len(hashes)
                while len(section_stack) > 1 and section_stack[-1].level >= level:
                    section_stack.pop()
                parent = section_stack[-1]
                section_path = list(parent.metadata.get("section_path", []))
                section_path.append(heading_text.strip())
                title_path = root.metadata["title_path"] + section_path
                section = Node(
                    node_id=make_node_id(doc_id, "section", order),
                    parent_id=parent.node_id,
                    doc_id=doc_id,
                    node_type="section",
                    title=heading_text.strip(),
                    text="",
                    order=order,
                    level=level,
                    metadata={
                        "source": file_path,
                        "file_name": source_name,
                        "source_type": "md",
                        "title_path": title_path,
                        "section_path": section_path,
                    },
                )
                nodes.append(section)
                order += 1
                section_stack.append(section)
                continue
            buffer.append(line)

        flush_paragraphs()
        return build_tree(doc_id, root_id, nodes)
