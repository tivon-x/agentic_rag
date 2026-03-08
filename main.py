from __future__ import annotations

import argparse
import logging
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.states import GraphState
from core.factory import build_graph, build_retriever
from indexing.indexer import Indexer
from core.settings import configure_logging, load_settings
from core.rag_answer import format_retrieval_only_answer

load_dotenv()  # 加载环境变量

logger = logging.getLogger(__name__)


def _offline_answer(settings, question: str) -> str:
    retriever = build_retriever(settings)
    if retriever is None:
        return "No index loaded. Run `python main.py index <path>` first."
    docs = retriever.invoke(question)
    return format_retrieval_only_answer(question, docs)


def cmd_index(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)
    cfg = settings.indexer_config()
    indexer = Indexer(cfg)

    for p in args.paths:
        logger.info("Indexing: %s", p)
        indexer.index(p)

    logger.info("Index saved to %s", settings.faiss_dir)
    logger.info("BM25 bundle saved to %s", settings.bm25_path)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)

    if (
        settings.offline_mode
        or not settings.llm_api_key
        or not settings.llm_api_base
        or not settings.llm_model
    ):
        content = _offline_answer(settings, args.question)
        logger.info("Answer (offline):\n%s", content)
        return 0

    graph = build_graph(settings)
    input_state = {"messages": [HumanMessage(content=args.question)]}
    result = graph.invoke(
        cast(GraphState, input_state),
        config={"configurable": {"thread_id": "cli"}},
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    content = (
        getattr(messages[-1], "content", str(messages[-1])) if messages else str(result)
    )
    logger.info("Answer:\n%s", content)
    return 0


def cmd_ui(_: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)
    from ui.gradio import build_ui

    demo = build_ui(settings)
    demo.queue()
    demo.launch()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index PDF(s) or directories")
    p_index.add_argument("paths", nargs="+", help="File or directory paths")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question against the local index")
    p_ask.add_argument("question", help="Question text")
    p_ask.set_defaults(func=cmd_ask)

    p_ui = sub.add_parser("ui", help="Launch Gradio UI")
    p_ui.set_defaults(func=cmd_ui)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
