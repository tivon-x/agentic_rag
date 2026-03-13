from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.parsers.common import build_doc_id, build_tree, make_node_id, split_paragraphs


def _clean_pdf_page(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PdfHierarchicalParser:
    def parse(self, file_path: str) -> ParsedDocumentTree:
        loader = PyPDFLoader(file_path)
        pages = list(loader.lazy_load())
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
            text="\n\n".join(_clean_pdf_page(page.page_content) for page in pages),
            order=order,
            level=0,
            metadata={
                "source": file_path,
                "file_name": source_name,
                "source_type": "pdf",
                "title_path": [Path(file_path).stem],
                "section_path": [],
            },
        )
        nodes.append(root)
        order += 1

        for page_index, page in enumerate(pages, start=1):
            page_title = f"Page {page_index}"
            section = Node(
                node_id=make_node_id(doc_id, "section", order),
                parent_id=root.node_id,
                doc_id=doc_id,
                node_type="section",
                title=page_title,
                text="",
                order=order,
                level=1,
                metadata={
                    "source": file_path,
                    "file_name": source_name,
                    "source_type": "pdf",
                    "page": page_index,
                    "title_path": [Path(file_path).stem, page_title],
                    "section_path": [page_title],
                },
            )
            nodes.append(section)
            order += 1

            for paragraph in split_paragraphs(_clean_pdf_page(page.page_content)):
                nodes.append(
                    Node(
                        node_id=make_node_id(doc_id, "paragraph", order),
                        parent_id=section.node_id,
                        doc_id=doc_id,
                        node_type="paragraph",
                        title=None,
                        text=paragraph,
                        order=order,
                        level=2,
                        metadata={
                            "source": file_path,
                            "file_name": source_name,
                            "source_type": "pdf",
                            "page": page_index,
                            "title_path": [Path(file_path).stem, page_title],
                            "section_path": [page_title],
                        },
                    )
                )
                order += 1

        return build_tree(doc_id, root_id, nodes)
