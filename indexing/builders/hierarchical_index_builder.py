from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.token_count import estimate_token_count


class HierarchicalIndexBuilder:
    def __init__(
        self,
        embeddings: Embeddings,
        leaf_node_type: str = "paragraph",
        parent_embed_pooling: str = "mean",
    ):
        self.embeddings = embeddings
        self.leaf_node_type = leaf_node_type
        self.parent_embed_pooling = parent_embed_pooling
        self._supported_leaf_types = {"paragraph", "section", "document"}
        self._supported_pooling = {"mean", "none"}
        if self.leaf_node_type not in self._supported_leaf_types:
            raise ValueError(f"Unsupported leaf node type: {self.leaf_node_type}")
        if self.parent_embed_pooling not in self._supported_pooling:
            raise ValueError(
                f"Unsupported parent embedding pooling strategy: {self.parent_embed_pooling}"
            )

    def enrich_trees(self, trees: list[ParsedDocumentTree]) -> list[ParsedDocumentTree]:
        for tree in trees:
            self._embed_tree(tree)
        return trees

    def to_documents(self, trees: list[ParsedDocumentTree]) -> list[Document]:
        documents: list[Document] = []
        for tree in trees:
            nodes_by_id = {node.node_id: node for node in tree.nodes}
            for node in tree.nodes:
                if node.node_type != self.leaf_node_type:
                    continue
                metadata = {
                    **node.metadata,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "doc_id": node.doc_id,
                    "node_type": node.node_type,
                    "title": node.title,
                    "order": node.order,
                    "level": node.level,
                    "token_count": node.token_count,
                }
                if node.parent_id and node.parent_id in nodes_by_id:
                    metadata["parent_title"] = nodes_by_id[node.parent_id].title
                documents.append(Document(page_content=node.text, metadata=metadata))
        return documents

    def _embed_tree(self, tree: ParsedDocumentTree) -> None:
        self._hydrate_parent_text(tree)

        leaf_nodes = [
            node
            for node in tree.nodes
            if node.node_type == self.leaf_node_type and node.text.strip()
        ]
        if leaf_nodes:
            vectors = self.embeddings.embed_documents([node.text for node in leaf_nodes])
            for node, vector in zip(leaf_nodes, vectors, strict=False):
                node.embedding = vector
                node.token_count = estimate_token_count(node.text)

        children_by_parent: dict[str, list[Node]] = defaultdict(list)
        for node in tree.nodes:
            if node.parent_id:
                children_by_parent[node.parent_id].append(node)

        for node in sorted(tree.nodes, key=lambda item: item.level, reverse=True):
            if node.embedding is not None:
                continue
            children = children_by_parent.get(node.node_id, [])
            child_embeddings = [child.embedding for child in children if child.embedding]
            if not child_embeddings:
                node.token_count = estimate_token_count(node.text)
                continue
            if self.parent_embed_pooling == "mean":
                dimensions = len(child_embeddings[0])
                node.embedding = [
                    sum(embedding[index] for embedding in child_embeddings)
                    / len(child_embeddings)
                    for index in range(dimensions)
                ]
            node.token_count = sum(child.token_count or 0 for child in children)

    def _hydrate_parent_text(self, tree: ParsedDocumentTree) -> None:
        children_by_parent: dict[str, list[Node]] = defaultdict(list)
        for node in tree.nodes:
            if node.parent_id:
                children_by_parent[node.parent_id].append(node)

        for node in sorted(tree.nodes, key=lambda item: item.level, reverse=True):
            if node.text.strip():
                continue
            children = children_by_parent.get(node.node_id, [])
            child_texts = [child.text.strip() for child in children if child.text.strip()]
            if child_texts:
                node.text = "\n\n".join(child_texts)
