from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


NodeType = Literal["document", "section", "paragraph", "sentence", "chunk"]


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    doc_id: str
    node_type: NodeType
    title: str | None
    text: str
    order: int
    level: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    token_count: int | None = None

    def to_dict(self, *, include_embedding: bool = True) -> dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "node_type": self.node_type,
            "title": self.title,
            "text": self.text,
            "order": self.order,
            "level": self.level,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }
        if include_embedding:
            payload["embedding"] = self.embedding
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Node:
        return cls(
            node_id=str(payload["node_id"]),
            parent_id=payload.get("parent_id"),
            doc_id=str(payload["doc_id"]),
            node_type=payload["node_type"],
            title=payload.get("title"),
            text=str(payload.get("text", "")),
            order=int(payload.get("order", 0)),
            level=int(payload.get("level", 0)),
            metadata=dict(payload.get("metadata", {})),
            embedding=payload.get("embedding"),
            token_count=payload.get("token_count"),
        )
