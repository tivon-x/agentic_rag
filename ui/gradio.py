from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import cast

import gradio as gr
import gradio.themes as themes
from langchain_core.messages import HumanMessage

from agent.states import GraphState
from core.corpus_profile import (
    format_corpus_profile,
    load_corpus_profile,
    save_corpus_profile,
)
from core.factory import build_retriever, build_graph
from core.rag_answer import (
    format_retrieval_only_answer,
    render_grounded_citations,
)
from core.settings import AppSettings
from indexing.indexer import Indexer


logger = logging.getLogger(__name__)


_CACHE: dict[str, object] = {"graph": None, "fingerprint": None}


SUPPORTED_SOURCE_TYPES = [".pdf", ".md", ".txt"]



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
                    kb_usage_notes = gr.Textbox(
                        label="使用说明",
                        lines=2,
                        placeholder="例如：优先回答产品实现、架构设计和 API 细节，不回答通用百科问题。",
                        value=str(existing_profile.get("usage_notes", "")),
                    )

                    gr.Markdown("<div class='section-title'>2. 导入知识源</div>")
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

                    def do_index(
                        corpus_name: str,
                        corpus_summary: str,
                        corpus_coverage: str,
                        corpus_usage_notes: str,
                        file_paths: list[str] | None,
                        progress=gr.Progress(),
                    ):
                        if not corpus_name.strip() and not corpus_summary.strip():
                            return (
                                "请至少填写“知识库名称”或“内容摘要”，让系统知道这批语料大致是什么。",
                                initial_profile_text,
                            )

                        source_examples = [Path(p).name for p in (file_paths or [])][:10]
                        profile_path = save_corpus_profile(
                            settings.index_dir,
                            name=corpus_name,
                            summary=corpus_summary,
                            coverage=corpus_coverage,
                            usage_notes=corpus_usage_notes,
                            source_examples=source_examples,
                        )

                        out_lines = [f"已保存知识库画像: {profile_path}"]

                        if file_paths:
                            cfg = settings.indexer_config()
                            indexer = Indexer(cfg)
                            for i, p in enumerate(file_paths, start=1):
                                progress(
                                    (i - 1) / max(len(file_paths), 1),
                                    desc=f"正在索引: {Path(p).name}",
                                )
                                logger.info("Indexing file from UI: %s", p)
                                indexer.index(p)
                                out_lines.append(f"已索引: {p}")

                            _invalidate_cache()
                            progress(1.0, desc="完成")
                            out_lines.append(f"FAISS: {settings.faiss_dir}")
                            out_lines.append(f"BM25: {settings.bm25_path}")
                        else:
                            out_lines.append("本次未上传新文件，仅更新了知识库画像。")

                        profile_text = format_corpus_profile(load_corpus_profile(settings.index_dir))
                        return "\n".join(out_lines), profile_text

                    def refresh_profile():
                        return format_corpus_profile(load_corpus_profile(settings.index_dir))

                    index_btn.click(
                        do_index,
                        inputs=[kb_name, kb_summary, kb_coverage, kb_usage_notes, files],
                        outputs=[status, corpus_profile_box],
                    )
                    refresh_profile_btn.click(
                        refresh_profile,
                        inputs=None,
                        outputs=corpus_profile_box,
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
                        "<div class='hint-card'><div class='section-title'>问答方式</div><div class='subtle'>这里的回答默认应该围绕当前知识库来进行。如果问题明显超出知识库边界，后续可以引导为直接回答或提示超出范围。</div></div>"
                    )
                    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
                    chatbot = gr.Chatbot(height=520)
                    with gr.Accordion("证据引用", open=False):
                        citation_box = gr.Markdown(
                            value="当前回答的引用会显示在这里。",
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
                            lambda: ("", [], "当前回答的引用会显示在这里。"),
                            inputs=None,
                            outputs=[msg, chatbot, citation_box],
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
                                "当前回答的引用会显示在这里。",
                            )
                        return "索引已加载。", [], profile_text, "当前回答的引用会显示在这里。"

                    def new_chat():
                        new_session_id = str(uuid.uuid4())
                        return [], new_session_id, "当前回答的引用会显示在这里。"

                    def user_msg(user_message: str, history):
                        return "", history + [{"role": "user", "content": user_message}]

                    def _extract_citations(result: dict | None) -> str:
                        if not isinstance(result, dict):
                            return "当前回答没有可展示的结构化引用。"
                        grounded = result.get("groundedAnswer", {})
                        if isinstance(grounded, dict) and grounded.get("evidence"):
                            return render_grounded_citations(grounded)
                        return "当前回答没有可展示的结构化引用。"

                    async def bot_msg(history, session_id):
                        offline = settings.offline_mode
                        graph = None if offline else _get_graph(settings)

                        user_message = history[-1]["content"]
                        history.append({"role": "assistant", "content": ""})
                        yield history, "正在整理本次回答的证据引用…"

                        try:
                            if graph is None:
                                retriever = build_retriever(settings)
                                if retriever is None:
                                    history[-1]["content"] = (
                                        "未加载索引。请先在“知识库构建”中保存画像并导入资料。"
                                    )
                                    yield history, "当前回答没有可展示的结构化引用。"
                                    return
                                docs = retriever.invoke(user_message)
                                answer = format_retrieval_only_answer(user_message, docs)
                                history[-1]["content"] = answer
                                yield history, "离线模式下仅展示检索摘录，节点级 citation 面板不可用。"
                            else:
                                input_state = {
                                    "messages": [HumanMessage(content=user_message)]
                                }
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
                                        if (
                                            chunk
                                            and hasattr(chunk, "content")
                                            and chunk.content
                                        ):
                                            streamed += chunk.content
                                            history[-1]["content"] = streamed
                                            yield history, "正在整理本次回答的证据引用…"

                                result: dict | None = None
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

                                yield history, _extract_citations(result)

                        except ConnectionError as e:
                            error_msg = "连接 AI 服务失败。请检查您的 API 配置是否正确。"
                            logger.error("Connection error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history, "当前回答没有可展示的结构化引用。"
                        except ValueError as e:
                            if "API key" in str(e) or "api_key" in str(e).lower():
                                error_msg = "API 密钥未配置。请在 .env 文件中设置 OPENAI_API_KEY。"
                            else:
                                error_msg = f"配置错误: {e}"
                            logger.error("Value error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history, "当前回答没有可展示的结构化引用。"
                        except TimeoutError as e:
                            error_msg = "请求超时，请重试。"
                            logger.error("Timeout error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history, "当前回答没有可展示的结构化引用。"
                        except Exception as e:
                            error_msg = f"发生错误: {e}，请重试。"
                            logger.error("Unexpected error: %s", e, exc_info=True)
                            history[-1]["content"] = error_msg
                            yield history, "当前回答没有可展示的结构化引用。"

                    reload_btn.click(
                        reload_index,
                        inputs=None,
                        outputs=[msg, chatbot, chat_profile_box, citation_box],
                    )
                    refresh_chat_profile_btn.click(
                        refresh_profile,
                        inputs=None,
                        outputs=chat_profile_box,
                    )
                    new_chat_btn.click(
                        new_chat,
                        inputs=None,
                        outputs=[chatbot, session_id_state, citation_box],
                    )
                    msg.submit(
                        user_msg, [msg, chatbot], [msg, chatbot], queue=False
                    ).then(bot_msg, [chatbot, session_id_state], [chatbot, citation_box])

        gr.Markdown(
            "<div class='subtle'>索引存储位置：<code>data/index/</code>。知识库画像会保存为 <code>data/index/corpus_profile.json</code>。</div>"
        )

    return demo
