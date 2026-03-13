from __future__ import annotations

from dataclasses import dataclass, field

from indexing.models.node import Node, NodeType


@dataclass
class ParsedDocumentTree:
    doc_id: str
    root_id: str
    nodes: list[Node]
    children_by_parent: dict[str, list[str]] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Node | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> list[Node]:
        child_ids = self.children_by_parent.get(node_id, [])
        node_by_id = {node.node_id: node for node in self.nodes}
        return [node_by_id[child_id] for child_id in child_ids if child_id in node_by_id]

    def iter_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        return [node for node in self.nodes if node.node_type == node_type]

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "root_id": self.root_id,
            "children_by_parent": self.children_by_parent,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object], *, nodes: list[Node] | None = None
    ) -> ParsedDocumentTree:
        return cls(
            doc_id=str(payload["doc_id"]),
            root_id=str(payload["root_id"]),
            nodes=nodes or [],
            children_by_parent={
                str(key): [str(child) for child in value]
                for key, value in dict(payload.get("children_by_parent", {})).items()
            },
        )
