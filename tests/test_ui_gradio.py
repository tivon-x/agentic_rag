from __future__ import annotations

from core.settings import load_settings
from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.stores.node_store import JsonNodeStore
from ui.gradio import (
    _extract_debug_payload,
    _load_index_stats,
    _render_tree_hits,
)


def test_load_index_stats_reports_hierarchical_counts(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")

    node_store = JsonNodeStore(settings.nodes_path, settings.doc_trees_path)
    nodes = [
        Node(
            node_id="doc-1",
            parent_id=None,
            doc_id="doc-1",
            node_type="document",
            title="Guide",
            text="Guide body",
            order=0,
            level=0,
            metadata={"source": "guide.md"},
            token_count=20,
        ),
        Node(
            node_id="sec-1",
            parent_id="doc-1",
            doc_id="doc-1",
            node_type="section",
            title="Intro",
            text="Intro body",
            order=0,
            level=1,
            metadata={"source": "guide.md"},
            token_count=12,
        ),
        Node(
            node_id="para-1",
            parent_id="sec-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Alpha paragraph",
            order=0,
            level=2,
            metadata={"source": "guide.md"},
            token_count=8,
        ),
    ]
    tree = ParsedDocumentTree(
        doc_id="doc-1",
        root_id="doc-1",
        nodes=nodes,
        children_by_parent={"doc-1": ["sec-1"], "sec-1": ["para-1"]},
    )
    node_store.save_trees([tree])

    stats = _load_index_stats(settings)

    assert "Hierarchical Mode" in stats
    assert "文档数: `1`" in stats
    assert "Section 数: `1`" in stats
    assert "Paragraph 数: `1`" in stats
    assert "叶子节点数: `1`" in stats


def test_extract_debug_payload_uses_first_evidence_group_debug():
    payload = _extract_debug_payload(
        {
            "routingDecision": "retrieve",
            "routingReason": "The corpus covers this question.",
            "queryPlan": {"intent": "summary"},
            "rewrittenQuestions": ["what changed in retrieval"],
            "packedContexts": [{"subquery": "what changed", "passage_count": 2}],
            "evidenceGroups": [
                {
                    "debug": {
                        "query_plan": {"intent": "summary"},
                        "raw_candidates": 6,
                        "structured_candidates": {"what changed": 2},
                        "dedupe": {"raw_count": 6, "deduped_count": 4},
                        "rerank": {
                            "top_candidates": [{"node_id": "para-1", "final_score": 0.9}],
                            "flashrank": {"enabled": False},
                        },
                        "packed_count": 2,
                        "total_tokens": 320,
                    }
                }
            ],
        }
    )

    assert payload["route_decision"]["decision"] == "retrieve"
    assert payload["retrieved_candidates"]["raw_candidates"] == 6
    assert payload["retrieved_candidates"]["dedupe"]["deduped_count"] == 4
    assert payload["reranked_top_passages"]["top_candidates"][0]["node_id"] == "para-1"
    assert payload["packed_context"]["total_tokens"] == 320


def test_load_index_stats_handles_unreadable_faiss_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)

    path_cls = type(settings.faiss_dir)
    original_iterdir = path_cls.iterdir

    def patched_iterdir(path_obj):
        if path_obj == settings.faiss_dir:
            raise OSError("access denied")
        return original_iterdir(path_obj)

    monkeypatch.setattr(path_cls, "iterdir", patched_iterdir)

    stats = _load_index_stats(settings)

    assert "当前索引概览" in stats
    assert "Flat Chunk Mode" in stats


def test_render_tree_hits_groups_by_source_and_section_path():
    markdown = _render_tree_hits(
        {
            "groundedAnswer": {
                "evidence": [
                    {
                        "source": "README.md",
                        "section_path": ["检索路由说明", "rewrite_query"],
                        "node_id": "para-1",
                    },
                    {
                        "source": "README.md",
                        "section_path": ["检索路由说明", "aggregate_answers"],
                        "node_id": "para-2",
                    },
                ]
            }
        }
    )

    assert "## 命中的文档树位置" in markdown
    assert "### README.md" in markdown
    assert "- 检索路由说明" in markdown
    assert "> rewrite_query" in markdown
    assert "> aggregate_answers" in markdown
