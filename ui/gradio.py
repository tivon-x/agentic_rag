from __future__ import annotations

from collections import defaultdict
import logging
import uuid
from pathlib import Path
from typing import Any, TypeAlias, cast

import gradio as gr
import gradio.themes as themes
from langchain_core.messages import HumanMessage

from agent.states import GraphState
from core.corpus_profile import (
    format_corpus_profile,
    load_corpus_profile,
    save_corpus_profile,
)
from core.factory import build_graph, build_retriever
from core.rag_answer import (
    format_retrieval_only_answer,
    render_grounded_citations,
)
from core.settings import AppSettings
from indexing.indexer import Indexer
from indexing.stores.node_store import create_node_store


logger = logging.getLogger(__name__)


_CACHE: dict[str, object] = {"graph": None, "fingerprint": None}


SUPPORTED_SOURCE_TYPES = [".pdf", ".md", ".txt"]

DebugTuple: TypeAlias = tuple[
    dict[str, Any],
    dict[str, Any],
    list[Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]
DebugWithTreeTuple: TypeAlias = tuple[
    dict[str, Any],
    dict[str, Any],
    list[Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]
ChatYieldTuple: TypeAlias = tuple[Any, ...]


def _split_profile_values(text: str) -> list[str]:
    items = [part.strip() for part in text.replace(";", "\n").splitlines()]
    return [item for item in items if item]


def _humanize_index_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "hierarchical":
        return "Hierarchical Mode"
    return "Flat Chunk Mode"


def _detect_current_index_mode(settings: AppSettings) -> str:
    if settings.nodes_path.exists() and settings.doc_trees_path.exists():
        return "hierarchical"
    return "flat"


def _fingerprint(settings: AppSettings) -> str:
    return (
        f"{settings.faiss_dir}|{settings.bm25_path}|{settings.llm_model}|{settings.llm_api_base}|"
        f"{settings.embedding_model}|{settings.embedding_api_base}"
    )


def _invalidate_cache() -> None:
    _CACHE["graph"] = None
    _CACHE["fingerprint"] = None


def _get_graph(settings: AppSettings):
    fp = _fingerprint(settings)
    cached = _CACHE.get("graph")
    if cached is not None and _CACHE.get("fingerprint") == fp:
        return cached

    if settings.offline_mode:
        return None

    try:
        graph = build_graph(settings)
        _CACHE["graph"] = graph
        _CACHE["fingerprint"] = fp
        return graph
    except RuntimeError:
        return None


def _load_index_stats(settings: AppSettings) -> str:
    mode = _detect_current_index_mode(settings)
    lines = [
        "### 当前索引概览",
        "",
        f"- 当前可检测索引模式: `{_humanize_index_mode(mode)}`",
    ]

    if mode == "hierarchical":
        node_store = create_node_store(
            settings.node_backend,
            nodes_path=settings.nodes_path,
            doc_trees_path=settings.doc_trees_path,
        )
        nodes = node_store.load_nodes()
        trees = node_store.load_trees()
        counts: dict[str, int] = defaultdict(int)
        token_values: list[int] = []
        parent_ids = {
            node.parent_id for node in nodes if isinstance(node.parent_id, str) and node.parent_id
        }

        for node in nodes:
            counts[node.node_type] += 1
            if isinstance(node.token_count, int) and node.token_count > 0:
                token_values.append(node.token_count)
        leaf_count = sum(
            1 for node in nodes if node.node_id not in parent_ids and node.text.strip()
        )

        avg_tokens = round(sum(token_values) / len(token_values), 1) if token_values else 0
        lines.extend(
            [
                f"- 文档数: `{len(trees)}`",
                f"- Section 数: `{counts.get('section', 0)}`",
                f"- Paragraph 数: `{counts.get('paragraph', 0)}`",
                f"- 叶子节点数: `{leaf_count}`",
                f"- 平均 tokens: `{avg_tokens}`",
            ]
        )
        return "\n".join(lines)

    try:
        faiss_ready = settings.faiss_dir.exists() and any(settings.faiss_dir.iterdir())
    except OSError:
        faiss_ready = False
    bm25_ready = settings.bm25_path.exists()
    lines.extend(
        [
            f"- 文档数: `{'已构建' if faiss_ready or bm25_ready else 0}`",
            "- Section 数: `N/A（Flat 模式不保留层级节点）`",
            "- Paragraph 数: `N/A（Flat 模式不保留层级节点）`",
            "- 叶子节点数: `N/A（Flat 模式不保留层级节点）`",
            "- 平均 tokens: `N/A`",
        ]
    )
    return "\n".join(lines)


def _extract_citations(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return "当前回答没有可展示的结构化引用。"
    grounded = result.get("groundedAnswer", {})
    if isinstance(grounded, dict) and grounded.get("evidence"):
        return render_grounded_citations(grounded)
    return "当前回答没有可展示的结构化引用。"


def _render_tree_hits(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return "当前没有可展示的命中文档树位置。"

    grounded = result.get("groundedAnswer", {})
    evidence = grounded.get("evidence", []) if isinstance(grounded, dict) else []
    if not evidence:
        evidence_groups = result.get("evidenceGroups", [])
        for group in evidence_groups if isinstance(evidence_groups, list) else []:
            if isinstance(group, dict):
                evidence.extend(group.get("evidence", []) or [])

    if not evidence:
        return "当前没有可展示的命中文档树位置。"

    grouped: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "unknown")).strip() or "unknown"
        section_path = item.get("section_path", []) or []
        if isinstance(section_path, str):
            section_path = [section_path]
        if section_path:
            grouped[source].add(
                " > ".join(str(part).strip() for part in section_path if str(part).strip())
            )
        else:
            node_id = str(item.get("node_id", "")).strip()
            grouped[source].add(node_id or "未标注 section path")

    lines = ["## 命中的文档树位置"]
    for source, paths in sorted(grouped.items()):
        lines.extend(["", f"### {source}"])
        for path in sorted(path for path in paths if path):
            segments = [segment.strip() for segment in path.split(">") if segment.strip()]
            if not segments:
                lines.append("- 未标注层级位置")
                continue
            lines.append(f"- {segments[0]}")
            for depth, segment in enumerate(segments[1:], start=1):
                lines.append(f"{'  ' * depth}> {segment}")
    return "\n".join(lines)


def _extract_debug_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "route_decision": {},
            "query_plan": {},
            "rewritten_queries": [],
            "retrieved_candidates": {},
            "reranked_top_passages": {},
            "packed_context": {},
        }

    evidence_groups = result.get("evidenceGroups", [])
    first_group = (
        evidence_groups[0]
        if isinstance(evidence_groups, list)
        and evidence_groups
        and isinstance(evidence_groups[0], dict)
        else {}
    )
    debug = first_group.get("debug", {}) if isinstance(first_group, dict) else {}
    rerank = debug.get("rerank", {}) if isinstance(debug, dict) else {}
    dedupe = debug.get("dedupe", {}) if isinstance(debug, dict) else {}

    return {
        "route_decision": {
            "decision": result.get("routingDecision"),
            "reason": result.get("routingReason"),
        },
        "query_plan": result.get("queryPlan", {}),
        "rewritten_queries": result.get("rewrittenQuestions", []),
        "retrieved_candidates": {
            "query_plan": debug.get("query_plan"),
            "raw_candidates": debug.get("raw_candidates"),
            "structured_candidates": debug.get("structured_candidates"),
            "dedupe": dedupe,
        },
        "reranked_top_passages": {
            "top_candidates": rerank.get("top_candidates", []),
            "flashrank": rerank.get("flashrank", {}),
        },
        "packed_context": {
            "packed_count": debug.get("packed_count"),
            "total_tokens": debug.get("total_tokens"),
            "packing_strategy": debug.get("packing_strategy", "score_then_contiguity"),
            "packed_contexts": result.get("packedContexts", []),
        },
    }


def _default_debug_outputs() -> DebugTuple:
    payload = _extract_debug_payload(None)
    return (
        payload["route_decision"],
        payload["query_plan"],
        payload["rewritten_queries"],
        payload["retrieved_candidates"],
        payload["reranked_top_passages"],
        payload["packed_context"],
    )


def _extract_debug_outputs(
    result: dict[str, Any] | None,
) -> DebugWithTreeTuple:
    payload = _extract_debug_payload(result)
    return (
        payload["route_decision"],
        payload["query_plan"],
        payload["rewritten_queries"],
        payload["retrieved_candidates"],
        payload["reranked_top_passages"],
        payload["packed_context"],
        _render_tree_hits(result),
    )


def _pending_debug_response(history: list[dict[str, str]]) -> ChatYieldTuple:
    return (
        history,
        "正在整理本次回答的证据引用…",
        *_default_debug_outputs(),
        "当前没有可展示的命中文档树位置。",
    )


def _empty_debug_response(
    history: list[dict[str, str]],
    citation_message: str,
) -> ChatYieldTuple:
    return (
        history,
        citation_message,
        *_default_debug_outputs(),
        "当前没有可展示的命中文档树位置。",
    )


def build_ui(settings: AppSettings) -> gr.Blocks:
    css = """
    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(202, 138, 4, 0.10), transparent 24%),
            linear-gradient(180deg, #f6f4ee 0%, #fffdf8 100%);
        color: #1f2937;
        font-family: "Source Han Sans SC", "Noto Sans SC", "Segoe UI", sans-serif;
    }
    .panel {
        background: rgba(255, 252, 245, 0.88);
        border: 1px solid rgba(30, 41, 59, 0.08);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
    }
    .hero {
        padding: 8px 4px 18px 4px;
    }
    .eyebrow {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #e6f4ef;
        color: #0f766e;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #111827;
        letter-spacing: -0.04em;
        font-weight: 800;
        font-size: 2.35rem;
        margin: 12px 0 8px 0;
    }
    .subtle {
        color: #6b7280;
        font-size: 0.98rem;
        line-height: 1.7;
    }
    .section-title {
        margin: 0 0 8px 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #172033;
    }
    .hint-card {
        background: linear-gradient(135deg, #fdf6e8 0%, #fffdfa 100%);
        border: 1px solid #f3e3b4;
        border-radius: 18px;
        padding: 16px 18px;
    }
    .profile-card {
        background: linear-gradient(135deg, #eef7f4 0%, #fffdfa 100%);
        border: 1px solid #cfe4db;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .gradio-tabs {
        background: transparent !important;
    }
    .tab-nav button {
        background: transparent !important;
        color: #667085 !important;
        border-bottom: 2px solid transparent !important;
    }
    .tab-nav button.selected {
        color: #0f766e !important;
        border-bottom: 2px solid #0f766e !important;
    }
    .input textarea, .input input {
        background: #fffdf8 !important;
        border: 1px solid #e7dcc4 !important;
        color: #1f2937 !important;
    }
    .input textarea:focus, .input input:focus {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 3px rgba(15,118,110,0.10) !important;
    }
    button.primary {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
        color: white !important;
        border: none !important;
    }
    button.primary:hover {
        filter: brightness(1.04);
    }
    button.secondary {
        background: #fffaf0 !important;
        color: #1f2937 !important;
        border: 1px solid #eadfca !important;
    }
    button.secondary:hover {
        background: #f7f0df !important;
    }
    .chatbot {
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid #eadfca !important;
    }
    .upload-button {
        background: linear-gradient(180deg, #fffaf0 0%, #fffdfa 100%) !important;
        border: 2px dashed #d8c7a4 !important;
    }
    .upload-button:hover {
        border-color: #0f766e !important;
        background: #f0faf7 !important;
    }
    .prose {
        color: #1f2937 !important;
    }
    """

    existing_profile = load_corpus_profile(settings.index_dir)
    initial_profile_text = format_corpus_profile(existing_profile)
    initial_index_stats = _load_index_stats(settings)
    initial_mode = _detect_current_index_mode(settings)
    (
        default_route,
        default_plan,
        default_queries,
        default_retrieved,
        default_rerank,
        default_packed,
    ) = _default_debug_outputs()

    with gr.Blocks(theme=themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            <div class='hero'>
              <div class='eyebrow'>Knowledge Workspace</div>
              <h1>企业知识库 RAG 工作台</h1>
              <div class='subtle'>
                先定义知识库要覆盖什么，再导入资料建立索引，最后基于这套语料进行问答。
                这不是“和单个 PDF 聊天”，而是“围绕一组有边界的知识源构建可用语义检索”。
              </div>
            </div>
            """,
        )

        with gr.Tabs():
            with gr.Tab("知识库构建"):
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("<div class='section-title'>1. 定义知识库边界</div>")
                    kb_name = gr.Textbox(
                        label="知识库名称",
                        placeholder="例如：企业内部产品与研发文档库",
                        value=str(existing_profile.get("name", "")),
                    )
                    kb_summary = gr.Textbox(
                        label="内容摘要",
                        lines=3,
                        placeholder="用 1 到 3 句话说明这批资料主要讲什么。",
                        value=str(existing_profile.get("summary", "")),
                    )
                    kb_coverage = gr.Textbox(
                        label="覆盖范围",
                        lines=3,
                        placeholder="说明适合回答哪些问题，不适合回答哪些问题。",
                        value=str(existing_profile.get("coverage", "")),
                    )
                    kb_non_coverage = gr.Textbox(
                        label="不覆盖范围",
                        lines=2,
                        placeholder="例如：通用百科、财务数据、未上传资料对应的问题。",
                        value=str(existing_profile.get("non_coverage", "")),
                    )
                    kb_usage_notes = gr.Textbox(
                        label="使用说明",
                        lines=2,
                        placeholder="例如：优先回答产品实现、架构设计和 API 细节，不回答通用百科问题。",
                        value=str(existing_profile.get("usage_notes", "")),
                    )
                    kb_domain_keywords = gr.Textbox(
                        label="领域关键词",
                        lines=2,
                        placeholder="每行一个，或用分号分隔。",
                        value="\n".join(existing_profile.get("domain_keywords", [])),
                    )
                    kb_primary_entities = gr.Textbox(
                        label="核心实体",
                        lines=2,
                        placeholder="每行一个，或用分号分隔。",
                        value="\n".join(existing_profile.get("primary_entities", [])),
                    )
                    kb_recommended_questions = gr.Textbox(
                        label="推荐提问",
                        lines=2,
                        placeholder="每行一个，列出这套知识库最适合回答的问题。",
                        value="\n".join(existing_profile.get("recommended_questions", [])),
                    )
                    kb_forbidden_questions = gr.Textbox(
                        label="禁止/不建议问题",
                        lines=2,
                        placeholder="每行一个，列出明确超范围的问题类型。",
                        value="\n".join(existing_profile.get("forbidden_questions", [])),
                    )
                    kb_answer_style = gr.Textbox(
                        label="偏好回答风格",
                        lines=2,
                        placeholder="例如：先给结论，再列证据，保持实现导向。",
                        value=str(existing_profile.get("preferred_answer_style", "")),
                    )

                    gr.Markdown("<div class='section-title'>2. 导入知识源</div>")
                    index_mode = gr.Radio(
                        label="索引模式",
                        choices=["flat", "hierarchical"],
                        value=initial_mode,
                        info="Flat Chunk Mode 适合快速构建；Hierarchical Mode 会保留文档树，便于调试、引用和命中路径展示。",
                    )
                    files = gr.File(
                        label="上传知识源文件",
                        file_count="multiple",
                        file_types=SUPPORTED_SOURCE_TYPES,
                        type="filepath",
                    )
                    gr.Markdown(
                        "<div class='subtle'>支持文件类型：<code>.pdf</code>、<code>.md</code>、<code>.txt</code>。建议按同一主题或同一业务域分批构建。</div>"
                    )

                    with gr.Row():
                        index_btn = gr.Button("保存并构建索引", variant="primary")
                        refresh_profile_btn = gr.Button("刷新知识库画像", variant="secondary")

                    status = gr.Textbox(label="构建状态", lines=8, interactive=False)
                    corpus_profile_box = gr.Textbox(
                        label="当前知识库画像",
                        lines=8,
                        interactive=False,
                        value=initial_profile_text,
                    )
                    index_stats_box = gr.Markdown(value=initial_index_stats)

                    def do_index(
                        corpus_name: str,
                        corpus_summary: str,
                        corpus_coverage: str,
                        corpus_non_coverage: str,
                        corpus_usage_notes: str,
                        corpus_domain_keywords: str,
                        corpus_primary_entities: str,
                        corpus_recommended_questions: str,
                        corpus_forbidden_questions: str,
                        corpus_answer_style: str,
                        selected_index_mode: str,
                        file_paths: list[str] | None,
                        progress=gr.Progress(),
                    ):
                        if not corpus_name.strip() and not corpus_summary.strip():
                            return (
                                "请至少填写“知识库名称”或“内容摘要”，让系统知道这批语料大致是什么。",
                                initial_profile_text,
                                initial_index_stats,
                            )

                        source_examples = [Path(p).name for p in (file_paths or [])][:10]
                        profile_path = save_corpus_profile(
                            settings.index_dir,
                            name=corpus_name,
                            summary=corpus_summary,
                            coverage=corpus_coverage,
                            non_coverage=corpus_non_coverage,
                            usage_notes=corpus_usage_notes,
                            source_examples=source_examples,
                            domain_keywords=_split_profile_values(corpus_domain_keywords),
                            primary_entities=_split_profile_values(corpus_primary_entities),
                            recommended_questions=_split_profile_values(
                                corpus_recommended_questions
                            ),
                            forbidden_questions=_split_profile_values(
                                corpus_forbidden_questions
                            ),
                            preferred_answer_style=corpus_answer_style,
                        )

                        out_lines = [
                            f"已保存知识库画像: {profile_path}",
                            f"本次构建模式: {_humanize_index_mode(selected_index_mode)}",
                        ]

                        if file_paths:
                            cfg = settings.indexer_config()
                            cfg["index_mode"] = selected_index_mode
                            indexer = Indexer(cfg)
                            for i, file_path in enumerate(file_paths, start=1):
                                progress(
                                    (i - 1) / max(len(file_paths), 1),
                                    desc=f"正在索引: {Path(file_path).name}",
                                )
                                logger.info("Indexing file from UI: %s", file_path)
                                indexer.index(file_path)
                                out_lines.append(f"已索引: {file_path}")

                            _invalidate_cache()
                            progress(1.0, desc="完成")
                            out_lines.append(f"FAISS: {settings.faiss_dir}")
                            out_lines.append(f"BM25: {settings.bm25_path}")
                        else:
                            out_lines.append("本次未上传新文件，仅更新了知识库画像。")

                        profile_text = format_corpus_profile(load_corpus_profile(settings.index_dir))
                        index_stats = _load_index_stats(settings)
                        return "\n".join(out_lines), profile_text, index_stats

                    def refresh_profile():
                        return (
                            format_corpus_profile(load_corpus_profile(settings.index_dir)),
                            _load_index_stats(settings),
                        )

                    def refresh_profile_text():
                        return format_corpus_profile(load_corpus_profile(settings.index_dir))

                    index_btn.click(
                        do_index,
                        inputs=[
                            kb_name,
                            kb_summary,
                            kb_coverage,
                            kb_non_coverage,
                            kb_usage_notes,
                            kb_domain_keywords,
                            kb_primary_entities,
                            kb_recommended_questions,
                            kb_forbidden_questions,
                            kb_answer_style,
                            index_mode,
                            files,
                        ],
                        outputs=[status, corpus_profile_box, index_stats_box],
                    )
                    refresh_profile_btn.click(
                        refresh_profile,
                        inputs=None,
                        outputs=[corpus_profile_box, index_stats_box],
                    )

            with gr.Tab("智能问答"):
                with gr.Column(elem_classes="panel"):
                    gr.Markdown(
                        "<div class='profile-card'><div class='section-title'>当前知识库边界</div><div class='subtle'>提问前先确认这套知识库主要覆盖什么，能减少超范围提问带来的误检索。</div></div>"
                    )
                    chat_profile_box = gr.Textbox(
                        label="当前知识库画像",
                        lines=8,
                        interactive=False,
                        value=initial_profile_text,
                    )
                    gr.Markdown(
                        "<div class='hint-card'><div class='section-title'>问答方式</div><div class='subtle'>这里不仅展示最终答案，也展示 route、query plan、rerank 和 packed context，方便观察系统为什么这样回答。</div></div>"
                    )
                    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
                    chatbot = gr.Chatbot(height=520)
                    with gr.Accordion("证据引用", open=False):
                        citation_box = gr.Markdown(
                            value="当前回答的引用会显示在这里。",
                            elem_classes="prose",
                        )
                    with gr.Accordion("调试面板", open=False):
                        route_decision_box = gr.JSON(
                            label="Route Decision",
                            value=default_route,
                        )
                        query_plan_box = gr.JSON(
                            label="Query Plan",
                            value=default_plan,
                        )
                        rewritten_queries_box = gr.JSON(
                            label="Rewritten Queries",
                            value=default_queries,
                        )
                        retrieved_candidates_box = gr.JSON(
                            label="Retrieved Candidates",
                            value=default_retrieved,
                        )
                        reranked_box = gr.JSON(
                            label="Reranked Top Passages",
                            value=default_rerank,
                        )
                        packed_context_box = gr.JSON(
                            label="Packed Context",
                            value=default_packed,
                        )
                    with gr.Accordion("命中文档树位置", open=False):
                        tree_hits_box = gr.Markdown(
                            value="当前没有可展示的命中文档树位置。",
                            elem_classes="prose",
                        )
                    msg = gr.Textbox(
                        placeholder="例如：这套知识库里关于检索流程重构的设计重点是什么？",
                        show_label=False,
                    )
                    with gr.Row():
                        reload_btn = gr.Button("重新加载索引")
                        refresh_chat_profile_btn = gr.Button("刷新知识库画像", variant="secondary")
                        new_chat_btn = gr.Button("新建对话")
                        clear_btn = gr.Button("清空对话", variant="secondary")
                        clear_btn.click(
                            lambda: (
                                "",
                                [],
                                "当前回答的引用会显示在这里。",
                                *_default_debug_outputs(),
                                "当前没有可展示的命中文档树位置。",
                            ),
                            inputs=None,
                            outputs=[
                                msg,
                                chatbot,
                                citation_box,
                                route_decision_box,
                                query_plan_box,
                                rewritten_queries_box,
                                retrieved_candidates_box,
                                reranked_box,
                                packed_context_box,
                                tree_hits_box,
                            ],
                        )

                    def reload_index():
                        _invalidate_cache()
                        graph = _get_graph(settings)
                        profile_text = format_corpus_profile(load_corpus_profile(settings.index_dir))
                        if graph is None:
                            return (
                                "未找到索引。请先在“知识库构建”中保存画像并导入资料。",
                                [],
                                profile_text,
                                "当前回答没有可展示的结构化引用。",
                                *_default_debug_outputs(),
                                "当前没有可展示的命中文档树位置。",
                            )
                        return (
                            "索引已加载。",
                            [],
                            profile_text,
                            "当前回答的引用会显示在这里。",
                            *_default_debug_outputs(),
                            "当前没有可展示的命中文档树位置。",
                        )

                    def new_chat():
                        new_session_id = str(uuid.uuid4())
                        return (
                            [],
                            new_session_id,
                            "当前回答的引用会显示在这里。",
                            *_default_debug_outputs(),
                            "当前没有可展示的命中文档树位置。",
                        )

                    def user_msg(user_message: str, history):
                        return "", history + [{"role": "user", "content": user_message}]

                    async def bot_msg(history, session_id):
                        offline = settings.offline_mode
                        graph = None if offline else _get_graph(settings)

                        user_message = history[-1]["content"]
                        history.append({"role": "assistant", "content": ""})
                        yield _pending_debug_response(history)

                        try:
                            if graph is None:
                                retriever = build_retriever(settings)
                                if retriever is None:
                                    history[-1]["content"] = (
                                        "未加载索引。请先在“知识库构建”中保存画像并导入资料。"
                                    )
                                    yield _empty_debug_response(
                                        history,
                                        "当前回答没有可展示的结构化引用。",
                                    )
                                    return

                                docs = retriever.invoke(user_message)
                                answer = format_retrieval_only_answer(user_message, docs)
                                history[-1]["content"] = answer
                                yield _empty_debug_response(
                                    history,
                                    "离线模式下仅展示检索摘录，节点级 citation 与调试面板不可用。",
                                )
                                return

                            input_state = {"messages": [HumanMessage(content=user_message)]}
                            config = {"configurable": {"thread_id": session_id}}

                            streamed = ""
                            async for event in graph.astream_events(
                                cast(GraphState, input_state),
                                config=config,
                                version="v2",
                            ):
                                kind = event.get("event", "")
                                if kind == "on_chat_model_stream":
                                    chunk = event.get("data", {}).get("chunk")
                                    if chunk and hasattr(chunk, "content") and chunk.content:
                                        streamed += chunk.content
                                        history[-1]["content"] = streamed
                                        yield _empty_debug_response(
                                            history,
                                            "正在整理本次回答的证据引用…",
                                        )

                            result: dict[str, Any] | None = None
                            if not streamed:
                                result = graph.invoke(
                                    cast(GraphState, input_state),
                                    config=config,
                                )
                                messages = (
                                    result.get("messages", [])
                                    if isinstance(result, dict)
                                    else []
                                )
                                answer = (
                                    getattr(messages[-1], "content", str(messages[-1]))
                                    if messages
                                    else str(result)
                                )
                                history[-1]["content"] = answer
                            else:
                                snapshot = graph.get_state(config)
                                result = (
                                    snapshot.values if hasattr(snapshot, "values") else None
                                )

                            yield (
                                history,
                                _extract_citations(result),
                                *_extract_debug_outputs(result),
                            )

                        except ConnectionError as exc:
                            error_msg = "连接 AI 服务失败。请检查您的 API 配置是否正确。"
                            logger.error("Connection error: %s", exc)
                            history[-1]["content"] = error_msg
                            yield _empty_debug_response(
                                history,
                                "当前回答没有可展示的结构化引用。",
                            )
                        except ValueError as exc:
                            if "API key" in str(exc) or "api_key" in str(exc).lower():
                                error_msg = "API 密钥未配置。请在 .env 文件中设置 OPENAI_API_KEY。"
                            else:
                                error_msg = f"配置错误: {exc}"
                            logger.error("Value error: %s", exc)
                            history[-1]["content"] = error_msg
                            yield _empty_debug_response(
                                history,
                                "当前回答没有可展示的结构化引用。",
                            )
                        except TimeoutError as exc:
                            error_msg = "请求超时，请重试。"
                            logger.error("Timeout error: %s", exc)
                            history[-1]["content"] = error_msg
                            yield _empty_debug_response(
                                history,
                                "当前回答没有可展示的结构化引用。",
                            )
                        except Exception as exc:
                            error_msg = f"发生错误: {exc}，请重试。"
                            logger.error("Unexpected error: %s", exc, exc_info=True)
                            history[-1]["content"] = error_msg
                            yield _empty_debug_response(
                                history,
                                "当前回答没有可展示的结构化引用。",
                            )

                    reload_btn.click(
                        reload_index,
                        inputs=None,
                        outputs=[
                            msg,
                            chatbot,
                            chat_profile_box,
                            citation_box,
                            route_decision_box,
                            query_plan_box,
                            rewritten_queries_box,
                            retrieved_candidates_box,
                            reranked_box,
                            packed_context_box,
                            tree_hits_box,
                        ],
                    )
                    refresh_chat_profile_btn.click(
                        refresh_profile_text,
                        inputs=None,
                        outputs=chat_profile_box,
                    )
                    new_chat_btn.click(
                        new_chat,
                        inputs=None,
                        outputs=[
                            chatbot,
                            session_id_state,
                            citation_box,
                            route_decision_box,
                            query_plan_box,
                            rewritten_queries_box,
                            retrieved_candidates_box,
                            reranked_box,
                            packed_context_box,
                            tree_hits_box,
                        ],
                    )
                    msg.submit(
                        user_msg,
                        [msg, chatbot],
                        [msg, chatbot],
                        queue=False,
                    ).then(
                        bot_msg,
                        [chatbot, session_id_state],
                        [
                            chatbot,
                            citation_box,
                            route_decision_box,
                            query_plan_box,
                            rewritten_queries_box,
                            retrieved_candidates_box,
                            reranked_box,
                            packed_context_box,
                            tree_hits_box,
                        ],
                    )

        gr.Markdown(
            "<div class='subtle'>索引存储位置：<code>data/index/</code>。知识库画像会保存为 <code>data/index/corpus_profile.json</code>。</div>"
        )

    return demo
