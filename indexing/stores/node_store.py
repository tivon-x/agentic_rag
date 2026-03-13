from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node


class NodeStore(Protocol):
    def save_trees(self, trees: list[ParsedDocumentTree]) -> None: ...
    def load_nodes(self) -> list[Node]: ...
    def load_trees(self) -> dict[str, ParsedDocumentTree]: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def get_children(self, node_id: str) -> list[Node]: ...
    def get_parent(self, node_id: str) -> Node | None: ...


class JsonNodeStore:
    def __init__(self, nodes_path: str | Path, doc_trees_path: str | Path):
        self.nodes_path = Path(nodes_path)
        self.doc_trees_path = Path(doc_trees_path)
        self._nodes_cache: list[Node] | None = None
        self._nodes_cache_mtime_ns: int | None = None
        self._trees_cache: dict[str, ParsedDocumentTree] | None = None
        self._trees_cache_mtime_ns: int | None = None

    def save_trees(self, trees: list[ParsedDocumentTree]) -> None:
        existing_trees = self.load_trees()
        for tree in trees:
            existing_trees[tree.doc_id] = tree

        all_nodes: dict[str, Node] = {}
        for tree in existing_trees.values():
            for node in tree.nodes:
                all_nodes[node.node_id] = node

        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_trees_path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes_path.write_text(
            "".join(
                json.dumps(node.to_dict(include_embedding=False), ensure_ascii=False) + "\n"
                for node in all_nodes.values()
            ),
            encoding="utf-8",
        )
        self.doc_trees_path.write_text(
            json.dumps(
                {doc_id: tree.to_dict() for doc_id, tree in existing_trees.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._nodes_cache = list(all_nodes.values())
        self._nodes_cache_mtime_ns = self.nodes_path.stat().st_mtime_ns
        self._trees_cache = dict(existing_trees)
        self._trees_cache_mtime_ns = self.doc_trees_path.stat().st_mtime_ns

    def load_nodes(self) -> list[Node]:
        if not self.nodes_path.exists():
            return []
        mtime_ns = self.nodes_path.stat().st_mtime_ns
        if self._nodes_cache is not None and self._nodes_cache_mtime_ns == mtime_ns:
            return list(self._nodes_cache)
        nodes: list[Node] = []
        for line in self.nodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            nodes.append(Node.from_dict(json.loads(line)))
        self._nodes_cache = list(nodes)
        self._nodes_cache_mtime_ns = mtime_ns
        return nodes

    def load_trees(self) -> dict[str, ParsedDocumentTree]:
        if not self.doc_trees_path.exists():
            return {}
        mtime_ns = self.doc_trees_path.stat().st_mtime_ns
        if self._trees_cache is not None and self._trees_cache_mtime_ns == mtime_ns:
            return {
                doc_id: ParsedDocumentTree.from_dict(
                    tree.to_dict(),
                    nodes=[Node.from_dict(node.to_dict()) for node in tree.nodes],
                )
                for doc_id, tree in self._trees_cache.items()
            }
        payload = json.loads(self.doc_trees_path.read_text(encoding="utf-8"))
        nodes_by_doc_id: dict[str, list[Node]] = {}
        for node in self.load_nodes():
            nodes_by_doc_id.setdefault(node.doc_id, []).append(node)
        trees = {
            str(doc_id): ParsedDocumentTree.from_dict(
                tree_payload,
                nodes=nodes_by_doc_id.get(str(doc_id), []),
            )
            for doc_id, tree_payload in payload.items()
        }
        self._trees_cache = {
            doc_id: ParsedDocumentTree.from_dict(
                tree.to_dict(),
                nodes=[Node.from_dict(node.to_dict()) for node in tree.nodes],
            )
            for doc_id, tree in trees.items()
        }
        self._trees_cache_mtime_ns = mtime_ns
        return trees

    def get_node(self, node_id: str) -> Node | None:
        for node in self.load_nodes():
            if node.node_id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> list[Node]:
        for tree in self.load_trees().values():
            children = tree.get_children(node_id)
            if children:
                return children
        return []

    def get_parent(self, node_id: str) -> Node | None:
        node = self.get_node(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.get_node(node.parent_id)
