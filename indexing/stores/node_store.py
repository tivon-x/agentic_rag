from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.stores.sqlite_node_store import SqliteNodeStore


@runtime_checkable
class NodeStore(Protocol):
    def save_nodes(self, nodes: list[Node]) -> None: ...
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
        self._node_index_cache: dict[str, Node] | None = None
        self._children_index_cache: dict[str, list[Node]] | None = None

    def save_nodes(self, nodes: list[Node]) -> None:
        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes_path.write_text(
            "".join(
                json.dumps(node.to_dict(include_embedding=False), ensure_ascii=False) + "\n"
                for node in nodes
            ),
            encoding="utf-8",
        )
        self._nodes_cache = list(nodes)
        self._nodes_cache_mtime_ns = self.nodes_path.stat().st_mtime_ns
        self._rebuild_node_indexes(nodes)

    def save_trees(self, trees: list[ParsedDocumentTree]) -> None:
        existing_trees = self.load_trees()
        for tree in trees:
            existing_trees[tree.doc_id] = tree

        all_nodes: dict[str, Node] = {}
        for tree in existing_trees.values():
            for node in tree.nodes:
                all_nodes[node.node_id] = node

        self.doc_trees_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_nodes(list(all_nodes.values()))
        self.doc_trees_path.write_text(
            json.dumps(
                {doc_id: tree.to_dict() for doc_id, tree in existing_trees.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
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
        self._rebuild_node_indexes(nodes)
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
        if self._node_index_cache is None:
            self._rebuild_node_indexes(self.load_nodes())
        assert self._node_index_cache is not None
        return self._node_index_cache.get(node_id)

    def get_children(self, node_id: str) -> list[Node]:
        if self._children_index_cache is None:
            self._rebuild_node_indexes(self.load_nodes())
        assert self._children_index_cache is not None
        return list(self._children_index_cache.get(node_id, []))

    def get_parent(self, node_id: str) -> Node | None:
        node = self.get_node(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.get_node(node.parent_id)

    def _rebuild_node_indexes(self, nodes: list[Node]) -> None:
        self._node_index_cache = {node.node_id: node for node in nodes}
        children_by_parent: dict[str, list[Node]] = {}
        for node in nodes:
            if node.parent_id is None:
                continue
            children_by_parent.setdefault(node.parent_id, []).append(node)
        for siblings in children_by_parent.values():
            siblings.sort(key=lambda item: item.order)
        self._children_index_cache = children_by_parent


def create_node_store(
    backend: str,
    *,
    nodes_path: str | Path,
    doc_trees_path: str | Path,
) -> NodeStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "json":
        return JsonNodeStore(nodes_path, doc_trees_path)
    if normalized_backend == "sqlite":
        return SqliteNodeStore(nodes_path=nodes_path, doc_trees_path=doc_trees_path)
    raise ValueError(f"Unsupported node backend: {backend}")
