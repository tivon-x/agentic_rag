from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import cast

import gradio as gr
import gradio.themes as themes
from langchain_core.messages import HumanMessage

from agent.graph_state import State
from core.factory import build_retriever, build_graph
from indexing.indexer import Indexer
from core.settings import AppSettings
from core.rag_answer import format_retrieval_only_answer


logger = logging.getLogger(__name__)


_CACHE: dict[str, object] = {"graph": None, "fingerprint": None}


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
        # No index loaded
        return None


def build_ui(settings: AppSettings) -> gr.Blocks:
    css = """
    .gradio-container {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        color: #1c1c1e;
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .title h1 {
        color: #1c1c1e;
        letter-spacing: -0.5px;
        font-weight: 700;
        font-size: 2rem;
    }
    .subtle {
        color: #6b7280;
        font-size: 1rem;
    }
    /* Gradio 组件亮色主题覆盖 */
    .gradio-tabs {
        background: transparent !important;
    }
    .tab-nav button {
        background: transparent !important;
        color: #6b7280 !important;
        border-bottom: 2px solid transparent !important;
    }
    .tab-nav button.selected {
        color: #10a37f !important;
        border-bottom: 2px solid #10a37f !important;
    }
    /* 输入框亮色样式 */
    .input textarea, .input input {
        background: #f7f7f7 !important;
        border: 1px solid #e5e5e5 !important;
        color: #1c1c1e !important;
    }
    .input textarea:focus, .input input:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16,163,127,0.1) !important;
    }
    /* 按钮亮色样式 */
    button.primary {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
    }
    button.primary:hover {
        background: #0e8f6f !important;
    }
    button.secondary {
        background: #f7f7f7 !important;
        color: #1c1c1e !important;
        border: 1px solid #e5e5e5 !important;
    }
    button.secondary:hover {
        background: #e5e5e5 !important;
    }
    /* 聊天机器人样式 */
    .chatbot {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
    }
    /* 文件上传区域 */
    .upload-button {
        background: #f7f7f7 !important;
        border: 2px dashed #d1d5db !important;
    }
    .upload-button:hover {
        border-color: #10a37f !important;
        background: #f0fdf4 !important;
    }
    /* Markdown 内容 */
    .prose {
        color: #1c1c1e !important;
    }
    """

    with gr.Blocks(theme=themes.Soft(), css=css) as demo:
        gr.Markdown(
            "# 智能 RAG 助手\n<span class='subtle'>上传 PDF 文档，建立索引，然后进行智能对话问答。</span>",
            elem_classes="title",
        )

        with gr.Tabs():
            with gr.Tab("文档索引"):
                with gr.Column(elem_classes="panel"):
                    files = gr.File(
                        label="上传 PDF 文件",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    index_btn = gr.Button("开始索引", variant="primary")
                    status = gr.Textbox(label="状态", lines=6, interactive=False)

                    def do_index(file_paths: list[str] | None, progress=gr.Progress()):
                        if not file_paths:
                            return "未选择任何文件。"

                        cfg = settings.indexer_config()
                        indexer = Indexer(cfg)

                        out_lines: list[str] = []
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
                        return "\n".join(out_lines)

                    index_btn.click(do_index, inputs=files, outputs=status)

            with gr.Tab("智能对话"):
                with gr.Column(elem_classes="panel"):
                    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
                    chatbot = gr.Chatbot(height=520)
                    msg = gr.Textbox(
                        placeholder="输入您的问题，例如：这份文档的主要内容是什么？",
                        show_label=False,
                    )
                    with gr.Row():
                        reload_btn = gr.Button("重新加载索引")
                        new_chat_btn = gr.Button("新建对话")
                        clear_btn = gr.Button("清空对话", variant="secondary")
                        clear_btn.click(
                            lambda: ("", []), inputs=None, outputs=[msg, chatbot]
                        )

                    def reload_index():
                        _invalidate_cache()
                        graph = _get_graph(settings)
                        if graph is None:
                            return (
                                "未找到索引。请先在「文档索引」标签页上传并索引文档。",
                                [],
                            )
                        return "索引已加载。", []

                    def new_chat():
                        new_session_id = str(uuid.uuid4())
                        return [], new_session_id

                    def user_msg(user_message: str, history):
                        return "", history + [{"role": "user", "content": user_message}]

                    async def bot_msg(history, session_id):
                        offline = settings.offline_mode
                        graph = None if offline else _get_graph(settings)

                        user_message = history[-1]["content"]
                        history.append({"role": "assistant", "content": ""})
                        yield history

                        try:
                            if graph is None:
                                retriever = build_retriever(settings)
                                if retriever is None:
                                    history[-1]["content"] = (
                                        "未加载索引。请先在「文档索引」标签页上传并索引文档。"
                                    )
                                    yield history
                                    return
                                docs = retriever.invoke(user_message)
                                answer = format_retrieval_only_answer(
                                    user_message, docs
                                )
                                history[-1]["content"] = answer
                                yield history
                            else:
                                # Real streaming with LangGraph astream_events
                                input_state = {
                                    "messages": [HumanMessage(content=user_message)]
                                }
                                config = {"configurable": {"thread_id": session_id}}

                                streamed = ""
                                async for event in graph.astream_events(  # type: ignore[attr-defined,arg-type]
                                    cast(State, input_state),
                                    config=config,  # type: ignore[arg-type]
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
                                            yield history

                                # Fallback if no streaming occurred
                                if not streamed:
                                    result = graph.invoke(  # type: ignore[attr-defined,arg-type]
                                        cast(State, input_state),
                                        config=config,  # type: ignore[arg-type]
                                    )
                                    messages = (
                                        result.get("messages", [])
                                        if isinstance(result, dict)
                                        else []
                                    )
                                    answer = (
                                        getattr(
                                            messages[-1], "content", str(messages[-1])
                                        )
                                        if messages
                                        else str(result)
                                    )
                                    history[-1]["content"] = answer
                                    yield history

                        except ConnectionError as e:
                            error_msg = (
                                "连接 AI 服务失败。请检查您的 API 配置是否正确。"
                            )
                            logger.error("Connection error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history
                        except ValueError as e:
                            if "API key" in str(e) or "api_key" in str(e).lower():
                                error_msg = "API 密钥未配置。请在 .env 文件中设置 OPENAI_API_KEY。"
                            else:
                                error_msg = f"配置错误: {e}"
                            logger.error("Value error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history
                        except TimeoutError as e:
                            error_msg = "请求超时，请重试。"
                            logger.error("Timeout error: %s", e)
                            history[-1]["content"] = error_msg
                            yield history
                        except Exception as e:
                            error_msg = f"发生错误: {e}，请重试。"
                            logger.error("Unexpected error: %s", e, exc_info=True)
                            history[-1]["content"] = error_msg
                            yield history

                    reload_btn.click(reload_index, inputs=None, outputs=[msg, chatbot])
                    new_chat_btn.click(
                        new_chat, inputs=None, outputs=[chatbot, session_id_state]
                    )
                    msg.submit(
                        user_msg, [msg, chatbot], [msg, chatbot], queue=False
                    ).then(bot_msg, [chatbot, session_id_state], chatbot)

        gr.Markdown(
            "<div class='subtle'>本地存储：FAISS 索引 + BM25 索引保存在 <code>data/index/</code> 目录下。</div>"
        )

    return demo
