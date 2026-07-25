from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.states import GraphState
from core.factory import build_graph, build_retriever
from core.settings import configure_logging, load_settings
from core.rag_answer import format_retrieval_only_answer
from evals.runner import parse_eval_runner_config, run_eval_suite
from indexing.indexer import Indexer

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
    if args.mode:
        cfg["index_mode"] = args.mode
    if args.leaf_node_type:
        cfg["leaf_node_type"] = args.leaf_node_type
    if args.parent_embed_pooling:
        cfg["parent_embed_pooling"] = args.parent_embed_pooling
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


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)
    config = parse_eval_runner_config(args)
    report = run_eval_suite(settings, config)

    logger.info("Eval suite: %s", report["suite"])
    logger.info("Embedding mode: %s", report["embedding_mode"])
    logger.info("LLM enabled: %s", report["llm_enabled"])
    for variant_name, variant_report in report["variants"].items():
        logger.info("Variant: %s", variant_name)
        for suite_name, suite_report in variant_report.get("suites", {}).items():
            logger.info("  %s metrics: %s", suite_name, suite_report.get("metrics", {}))
    for suite_name, rows in report.get("leaderboard", {}).items():
        top = rows[0] if rows else None
        if top is not None:
            logger.info(
                "Leaderboard winner for %s: %s (score=%s, comparable=%s)",
                suite_name,
                top["variant"],
                top["score"],
                top.get("comparable", True),
            )
    logger.info("Artifacts: %s", report.get("artifacts", {}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index PDF(s) or directories")
    p_index.add_argument("paths", nargs="+", help="File or directory paths")
    p_index.add_argument(
        "--mode",
        choices=["flat", "hierarchical"],
        help="Index mode override. Defaults to INDEX_MODE or flat.",
    )
    p_index.add_argument(
        "--leaf-node-type",
        choices=["paragraph", "section", "document"],
        help="Leaf node type used in hierarchical mode.",
    )
    p_index.add_argument(
        "--parent-embed-pooling",
        choices=["mean", "none"],
        help="Parent embedding aggregation strategy in hierarchical mode.",
    )
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question against the local index")
    p_ask.add_argument("question", help="Question text")
    p_ask.set_defaults(func=cmd_ask)

    p_ui = sub.add_parser("ui", help="Launch Gradio UI")
    p_ui.set_defaults(func=cmd_ui)

    p_api = sub.add_parser("api", help="Launch FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1", help="Host for FastAPI server")
    p_api.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    p_api.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for FastAPI development",
    )
    p_api.set_defaults(func=cmd_api)

    p_eval = sub.add_parser("eval", help="Run routing/retrieval/answer evaluation suites")
    p_eval.add_argument(
        "--suite",
        choices=["routing", "retrieval", "answer", "all"],
        default="all",
        help="Evaluation suite to run.",
    )
    p_eval.add_argument(
        "--output-format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Report formats to write.",
    )
    p_eval.add_argument(
        "--output-dir",
        default=str(Path("data") / "eval_reports"),
        help="Directory for generated reports and eval indexes.",
    )
    p_eval.add_argument(
        "--dataset-dir",
        default=str(Path("evals") / "datasets"),
        help="Directory containing eval JSONL datasets.",
    )
    p_eval.add_argument(
        "--corpus-dir",
        default="evals",
        help="Directory containing the eval source documents.",
    )
    p_eval.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild eval indexes even if cached artifacts already exist.",
    )
    p_eval.add_argument(
        "--offline",
        action="store_true",
        help="Force FakeEmbeddings and skip LLM-based judging.",
    )
    p_eval.add_argument(
        "--variant",
        action="append",
        choices=["baseline_flat", "flat_rerank", "hierarchical"],
        help="Variant(s) to evaluate. Defaults to all three.",
    )
    p_eval.set_defaults(func=cmd_eval)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
