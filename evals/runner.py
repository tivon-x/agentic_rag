from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.graph import create_agent_graph
from agent.nodes.decide_retrieval import decide_retrieval
from agent.states import GraphState
from agent.tools import ToolFactory
from core.corpus_profile import (
    analyze_corpus_profile_match,
    build_corpus_profile_context,
    load_corpus_profile,
    save_corpus_profile,
)
from core.persistence import load_bm25_bundle
from core.settings import AppSettings
from evals.metrics import (
    answer_completeness,
    citation_precision,
    groundedness_score,
    hallucination_rate_rule,
    ndcg_at_k,
    normalize_identifier,
    reciprocal_rank,
    recall_at_k,
    redundancy_rate,
    route_accuracy,
)
from indexing.bm25_index import create_lexical_store
from indexing.indexer import Indexer
from indexing.retriever import FusionRetriever
from indexing.stores.lexical_store import LexicalStore
from indexing.stores.node_store import NodeStore, create_node_store
from llms.llm import configure_llm_router, get_llm


logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_DIRECT_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|你好|您好|谢谢|thanks)\b", re.IGNORECASE),
    re.compile(r"\btranslate\b|\b翻译\b", re.IGNORECASE),
    re.compile(r"^\s*[\d\s\+\-\*/=\(\)]+\s*$"),
]
_OUT_OF_SCOPE_HINTS = [
    "weather",
    "stock",
    "finance",
    "tesla",
    "sports",
    "politics",
    "总统",
    "股价",
    "天气",
]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_route: str
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_node_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""
    difficulty: str = "medium"
    notes: str = ""


@dataclass(frozen=True)
class EvalVariant:
    name: str
    index_mode: Literal["flat", "hierarchical"]
    reranker_backend: str
    description: str


@dataclass(frozen=True)
class EvalRunnerConfig:
    suite: Literal["routing", "retrieval", "answer", "all"] = "all"
    output_format: Literal["markdown", "json", "both"] = "both"
    output_dir: Path = Path("data/eval_reports")
    dataset_dir: Path = Path("evals/datasets")
    corpus_dir: Path = Path("evals")
    force_reindex: bool = False
    offline: bool = False
    variants: tuple[str, ...] = (
        "baseline_flat",
        "flat_rerank",
        "hierarchical",
    )


class HallucinationJudge(BaseModel):
    unsupported_claim_fraction: float = Field(ge=0.0, le=1.0)
    reason: str


VARIANTS: dict[str, EvalVariant] = {
    "baseline_flat": EvalVariant(
        name="baseline_flat",
        index_mode="flat",
        reranker_backend="none",
        description="Flat chunk RAG without rerank.",
    ),
    "flat_rerank": EvalVariant(
        name="flat_rerank",
        index_mode="flat",
        reranker_backend="flashrank",
        description="Flat chunk RAG with rerank enabled.",
    ),
    "hierarchical": EvalVariant(
        name="hierarchical",
        index_mode="hierarchical",
        reranker_backend="flashrank",
        description="Hierarchical RAG with rerank enabled.",
    ),
}


def parse_eval_runner_config(args: argparse.Namespace) -> EvalRunnerConfig:
    variants = tuple(args.variant or list(VARIANTS))
    return EvalRunnerConfig(
        suite=args.suite,
        output_format=args.output_format,
        output_dir=Path(args.output_dir),
        dataset_dir=Path(args.dataset_dir),
        corpus_dir=Path(args.corpus_dir),
        force_reindex=bool(args.force_reindex),
        offline=bool(args.offline),
        variants=variants,
    )


def run_eval_suite(settings: AppSettings, config: EvalRunnerConfig) -> dict[str, Any]:
    dataset_dir = config.dataset_dir
    corpus_dir = config.corpus_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_by_suite = _load_cases_for_suite(dataset_dir, config.suite)
    use_fake_embeddings = config.offline or not (
        settings.embedding_api_key and settings.embedding_api_base
    )
    use_llm = _has_llm_config(settings) and not config.offline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "suite": config.suite,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_dir": str(corpus_dir),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "embedding_mode": "fake" if use_fake_embeddings else "cloud",
        "llm_enabled": use_llm,
        "variants": {},
    }

    for variant_name in config.variants:
        variant = VARIANTS[variant_name]
        logger.info("Running eval variant: %s", variant.name)
        variant_settings = _prepare_variant_settings(settings, output_dir / "indexes", variant)
        _ensure_variant_index(
            variant_settings,
            corpus_dir=corpus_dir,
            variant=variant,
            use_fake_embeddings=use_fake_embeddings,
            force_reindex=config.force_reindex,
        )
        _ensure_eval_corpus_profile(variant_settings.index_dir)
        retriever = _build_retriever_for_eval(
            variant_settings,
            variant=variant,
            use_fake_embeddings=use_fake_embeddings,
        )
        profile = load_corpus_profile(variant_settings.index_dir)
        graph = (
            _build_graph_for_eval(variant_settings, retriever, profile)
            if use_llm
            else None
        )

        suites: dict[str, Any] = {}
        if "routing" in cases_by_suite:
            suites["routing"] = _evaluate_routing_suite(
                cases_by_suite["routing"],
                settings=variant_settings,
                profile=profile,
                use_llm=use_llm,
            )
        if "retrieval" in cases_by_suite:
            suites["retrieval"] = _evaluate_retrieval_suite(
                cases_by_suite["retrieval"],
                retriever=retriever,
                k=variant_settings.retriever_k,
            )
        if "answer" in cases_by_suite:
            suites["answer"] = _evaluate_answer_suite(
                cases_by_suite["answer"],
                retriever=retriever,
                graph=graph,
                llm_config=variant_settings.llm_config(),
                llm_judge_available=use_llm,
            )

        report["variants"][variant.name] = {
            "description": variant.description,
            "index_mode": variant.index_mode,
            "reranker_backend": variant.reranker_backend,
            "suites": suites,
        }

    report["comparisons"] = _build_comparisons(report["variants"])
    report["leaderboard"] = _build_leaderboard(report["variants"])
    report["artifacts"] = _write_reports(report, output_dir, timestamp, config.output_format)
    return report


def _load_cases_for_suite(
    dataset_dir: Path,
    suite: Literal["routing", "retrieval", "answer", "all"],
) -> dict[str, list[EvalCase]]:
    requested = ("routing", "retrieval", "answer") if suite == "all" else (suite,)
    return {
        item: _load_eval_cases(dataset_dir / f"{item}_cases.jsonl")
        for item in requested
    }


def _load_eval_cases(path: Path) -> list[EvalCase]:
    records: list[EvalCase] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        records.append(
            EvalCase(
                case_id=str(payload.get("case_id", f"case-{index}")),
                question=str(payload["question"]).strip(),
                expected_route=str(payload.get("expected_route", "retrieve")).strip(),
                gold_doc_ids=[str(item) for item in payload.get("gold_doc_ids", []) if str(item).strip()],
                gold_node_ids=[str(item) for item in payload.get("gold_node_ids", []) if str(item).strip()],
                reference_answer=str(payload.get("reference_answer", "")).strip(),
                difficulty=str(payload.get("difficulty", "medium")).strip() or "medium",
                notes=str(payload.get("notes", "")).strip(),
            )
        )
    return records


def _prepare_variant_settings(
    settings: AppSettings,
    base_output_dir: Path,
    variant: EvalVariant,
) -> AppSettings:
    root = base_output_dir / variant.name
    data_dir = root / "data"
    index_dir = data_dir / "index"
    prepared = replace(
        settings,
        data_dir=data_dir,
        index_dir=index_dir,
        faiss_dir=index_dir / "faiss",
        bm25_path=index_dir / "bm25.pkl",
        nodes_path=index_dir / "nodes.jsonl",
        doc_trees_path=index_dir / "doc_trees.json",
    )
    prepared.ensure_dirs()
    return prepared


def _ensure_variant_index(
    settings: AppSettings,
    *,
    corpus_dir: Path,
    variant: EvalVariant,
    use_fake_embeddings: bool,
    force_reindex: bool,
) -> None:
    has_flat_index = settings.faiss_dir.exists() and settings.bm25_path.exists()
    has_hierarchical_index = settings.nodes_path.exists() and settings.doc_trees_path.exists()
    if not force_reindex and has_flat_index and (
        variant.index_mode == "flat" or has_hierarchical_index
    ):
        return

    config = settings.indexer_config()
    config["index_mode"] = variant.index_mode
    config["retriever"]["reranker_backend"] = variant.reranker_backend
    if use_fake_embeddings:
        config["embedding"]["type"] = "fake"

    indexer = Indexer(config)
    indexer.index(str(corpus_dir))


def _build_retriever_for_eval(
    settings: AppSettings,
    *,
    variant: EvalVariant,
    use_fake_embeddings: bool,
) -> FusionRetriever:
    cfg = settings.indexer_config()
    cfg["index_mode"] = variant.index_mode
    cfg["retriever"]["reranker_backend"] = variant.reranker_backend
    if use_fake_embeddings:
        cfg["embedding"]["type"] = "fake"

    indexer = Indexer(cfg)

    lexical_store: LexicalStore
    if settings.bm25_path.exists():
        lexical_store = create_lexical_store(
            settings.lexical_backend,
            bundle=load_bm25_bundle(settings.bm25_path),
        )
    else:
        lexical_store = create_lexical_store(
            settings.lexical_backend,
            documents=indexer.vector_store.get_all_documents(),
        )

    node_store: NodeStore | None = None
    if settings.nodes_path.exists() and settings.doc_trees_path.exists():
        node_store = create_node_store(
            settings.node_backend,
            nodes_path=settings.nodes_path,
            doc_trees_path=settings.doc_trees_path,
        )

    return FusionRetriever(
        vectorstore=indexer.vector_store,
        lexical_store=lexical_store,
        alpha=settings.fusion_alpha,
        k=settings.retriever_k,
        reranker_backend=variant.reranker_backend,
        flashrank_model=settings.flashrank_model,
        flashrank_cache_dir=settings.flashrank_cache_dir,
        flashrank_top_n=settings.flashrank_top_n,
        node_store=node_store,
        corpus_profile=load_corpus_profile(settings.index_dir),
    )


def _ensure_eval_corpus_profile(index_dir: Path) -> None:
    save_corpus_profile(
        index_dir,
        name="Neural Architecture Papers Eval Corpus",
        summary=(
            "A mixed corpus of classic papers and blog posts about transformers, recurrent neural "
            "networks, LSTMs, regularization, scaling laws, ResNets, and related deep learning ideas."
        ),
        coverage=(
            "Deep learning architecture design, model scaling, attention, RNN/LSTM behavior, "
            "regularization, computer vision architectures, and sequence modeling."
        ),
        non_coverage="Weather, finance, stock prices, sports, politics, travel, and personal scheduling.",
        usage_notes=(
            "Prefer grounded answers that cite the exact source document. Comparison questions "
            "across papers are allowed when evidence is drawn from the retrieved sources."
        ),
        source_examples=[
            "01-Attention Is All You Need.pdf",
            "02-The Annotated Transformer.md",
            "05-Understanding LSTM Networks.md",
            "23-Scaling Laws for Neural Language Models.pdf",
        ],
        recommended_questions=[
            "What does the Transformer replace recurrence with?",
            "How do LSTMs address long-term dependencies?",
            "What do scaling laws say about compute-efficient training?",
        ],
        forbidden_questions=[
            "What is today's weather?",
            "Should I buy Tesla stock this week?",
        ],
        domain_keywords=[
            "transformer",
            "attention",
            "self-attention",
            "recurrent neural network",
            "RNN",
            "LSTM",
            "dropout",
            "regularization",
            "scaling laws",
            "ResNet",
        ],
        preferred_answer_style="Concise technical synthesis with explicit citations and limitations.",
        primary_entities=["Transformer", "LSTM", "RNN", "ResNet", "GPipe", "scaling laws"],
    )


def _build_graph_for_eval(settings: AppSettings, retriever: FusionRetriever, profile: dict[str, Any]):
    configure_llm_router(settings.llm_config())
    tool_factory = ToolFactory(retriever)
    return create_agent_graph(
        tool_factory.create_tools(),
        corpus_profile=build_corpus_profile_context(profile),
        corpus_profile_data=profile,
        tool_factory=tool_factory,
    )


def _evaluate_routing_suite(
    cases: list[EvalCase],
    *,
    settings: AppSettings,
    profile: dict[str, Any],
    use_llm: bool,
) -> dict[str, Any]:
    if use_llm:
        configure_llm_router(settings.llm_config())

    rows = []
    for case in cases:
        predicted = _predict_route(case.question, profile, use_llm=use_llm)
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_route": case.expected_route,
                "predicted_route": predicted,
                "accuracy": route_accuracy(case.expected_route, predicted),
                "difficulty": case.difficulty,
            }
        )

    return {
        "metrics": {
            "route_accuracy": round(mean(row["accuracy"] for row in rows), 4) if rows else 0.0,
            "case_count": len(rows),
            "mode": "llm" if use_llm else "heuristic_fallback",
        },
        "cases": rows,
    }


def _predict_route(question: str, profile: dict[str, Any], *, use_llm: bool) -> str:
    if use_llm:
        try:
            state = cast(
                GraphState,
                {
                    "messages": [HumanMessage(content=question)],
                    "conversation_summary": "",
                    "corpusProfile": build_corpus_profile_context(profile),
                    "corpusProfileData": profile,
                },
            )
            return str(decide_retrieval(state)["routingDecision"])
        except Exception as exc:
            logger.warning("Routing eval fell back to heuristic mode: %s", exc)

    lowered = question.casefold()
    if any(pattern.search(question) for pattern in _DIRECT_PATTERNS):
        return "direct_answer"
    if analyze_corpus_profile_match(question, profile)["force_out_of_scope"]:
        return "out_of_scope"
    if any(hint in lowered for hint in _OUT_OF_SCOPE_HINTS):
        return "out_of_scope"
    return "retrieve"


def _evaluate_retrieval_suite(
    cases: list[EvalCase],
    *,
    retriever: FusionRetriever,
    k: int,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        packed = retriever.retrieve(case.question)
        relevances = _ranked_relevances(packed.passages[:k], case)
        predicted_passage_doc_ids = [_best_document_id(document) for document in packed.passages[:k]]
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "gold_doc_ids": case.gold_doc_ids,
                "predicted_doc_ids": _ordered_unique(predicted_passage_doc_ids),
                "predicted_passage_doc_ids": predicted_passage_doc_ids,
                "recall_at_k": round(recall_at_k(relevances, len(case.gold_doc_ids), k=k), 4),
                "mrr": round(reciprocal_rank(relevances), 4),
                "ndcg": round(ndcg_at_k(relevances, len(case.gold_doc_ids), k=k), 4),
                "redundancy_rate": round(
                    redundancy_rate([document.page_content for document in packed.passages[:k]]),
                    4,
                ),
                "difficulty": case.difficulty,
            }
        )

    return {
        "metrics": {
            "recall_at_k": round(mean(row["recall_at_k"] for row in rows), 4) if rows else 0.0,
            "mrr": round(mean(row["mrr"] for row in rows), 4) if rows else 0.0,
            "ndcg": round(mean(row["ndcg"] for row in rows), 4) if rows else 0.0,
            "redundancy_rate": round(mean(row["redundancy_rate"] for row in rows), 4) if rows else 0.0,
            "case_count": len(rows),
            "k": k,
        },
        "cases": rows,
    }


def _evaluate_answer_suite(
    cases: list[EvalCase],
    *,
    retriever: FusionRetriever,
    graph,
    llm_config: dict[str, Any],
    llm_judge_available: bool,
) -> dict[str, Any]:
    judge = get_llm(llm_config) if llm_judge_available else None
    rows = []
    answer_modes: set[str] = set()

    for case in cases:
        payload = _generate_answer_payload(case.question, retriever=retriever, graph=graph)
        answer = str(payload.get("answer", "")).strip()
        evidence = payload.get("evidence", []) or []
        answer_mode = str(payload.get("answer_mode", "unknown")).strip() or "unknown"
        answer_modes.add(answer_mode)
        evidence_quotes = [
            str(item.get("quote", "")).strip()
            for item in evidence
            if str(item.get("quote", "")).strip()
        ]
        cited_doc_ids = [str(item.get("doc_id") or item.get("source") or "") for item in evidence]
        cited_node_ids = [str(item.get("node_id") or "") for item in evidence]
        llm_hallucination = (
            _judge_hallucination_rate(
                judge,
                question=case.question,
                answer=answer,
                evidence_quotes=evidence_quotes,
                reference_answer=case.reference_answer,
            )
            if judge is not None
            else None
        )

        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "answer_mode": answer_mode,
                "answer_preview": answer[:240],
                "groundedness": round(groundedness_score(answer, evidence_quotes), 4),
                "citation_precision": round(
                    citation_precision(
                        cited_doc_ids,
                        case.gold_doc_ids,
                        cited_node_ids=cited_node_ids,
                        gold_node_ids=case.gold_node_ids,
                    ),
                    4,
                ),
                "answer_completeness": round(answer_completeness(answer, case.reference_answer), 4),
                "hallucination_rate_rule": round(
                    hallucination_rate_rule(
                        answer,
                        evidence_quotes,
                        reference_answer=case.reference_answer,
                    ),
                    4,
                ),
                "hallucination_rate_llm_judge": llm_hallucination,
                "difficulty": case.difficulty,
            }
        )

    metrics: dict[str, Any] = {
        "groundedness": round(mean(row["groundedness"] for row in rows), 4) if rows else 0.0,
        "citation_precision": round(mean(row["citation_precision"] for row in rows), 4) if rows else 0.0,
        "answer_completeness": round(mean(row["answer_completeness"] for row in rows), 4) if rows else 0.0,
        "hallucination_rate_rule": round(mean(row["hallucination_rate_rule"] for row in rows), 4) if rows else 0.0,
        "case_count": len(rows),
        "answer_mode": sorted(answer_modes)[0] if len(answer_modes) == 1 else sorted(answer_modes),
        "evaluation_mode": (
            "generative_grounded"
            if answer_modes == {"graph_grounded"}
            else "offline_extractive_fallback"
        ),
    }
    llm_values = [
        cast(float, row["hallucination_rate_llm_judge"])
        for row in rows
        if isinstance(row["hallucination_rate_llm_judge"], float)
    ]
    metrics["hallucination_rate_llm_judge"] = round(mean(llm_values), 4) if llm_values else None

    return {"metrics": metrics, "cases": rows}


def _generate_answer_payload(question: str, *, retriever: FusionRetriever, graph) -> dict[str, Any]:
    if graph is not None:
        try:
            result = graph.invoke(
                cast(GraphState, {"messages": [HumanMessage(content=question)]}),
                config={"configurable": {"thread_id": f"eval-{hash(question)}"}},
            )
            grounded = result.get("groundedAnswer") if isinstance(result, dict) else None
            if isinstance(grounded, dict) and grounded.get("answer"):
                grounded = dict(grounded)
                grounded.setdefault("answer_mode", "graph_grounded")
                return grounded
        except Exception as exc:
            logger.warning("Graph answer generation fell back to offline synthesis: %s", exc)

    packed = retriever.retrieve(question)
    evidence = []
    for document in packed.passages[:4]:
        metadata = document.metadata
        evidence.append(
            {
                "doc_id": _best_document_id(document),
                "node_id": str(metadata.get("node_id", "")).strip(),
                "source": str(metadata.get("source", "")).strip() or _best_document_id(document),
                "section_path": _section_path(metadata),
                "page": metadata.get("page") if isinstance(metadata.get("page"), int) else None,
                "quote": document.page_content[:320].strip(),
                "score": metadata.get("score") if isinstance(metadata.get("score"), int | float) else None,
                "relevance": None,
            }
        )

    quotes = [item["quote"] for item in evidence if item["quote"]]
    answer = _build_offline_extractive_answer(
        question,
        evidence_quotes=quotes,
    )
    return {
        "answer": answer or "No grounded answer could be synthesized from the retrieved passages.",
        "reasoning_summary": (
            f"Offline extractive synthesis from {len(evidence)} retrieved evidence item(s); "
            "this fallback ranks and compresses evidence sentences instead of using generative aggregation."
        ),
        "evidence": evidence,
        "confidence": round(min(0.9, 0.2 + (0.15 * len(evidence))), 2),
        "limitations": "Fallback answer is extractive and deterministic; it approximates answer quality without LLM synthesis.",
        "answer_mode": "offline_extractive",
    }


def _judge_hallucination_rate(
    judge,
    *,
    question: str,
    answer: str,
    evidence_quotes: list[str],
    reference_answer: str,
) -> float | None:
    try:
        result = judge.with_structured_output(HallucinationJudge).invoke(
            [
                HumanMessage(
                    content=(
                        "You are grading whether an answer contains unsupported claims.\n"
                        "Return a structured score where unsupported_claim_fraction is between 0 and 1.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Answer:\n{answer}\n\n"
                        f"Reference answer:\n{reference_answer}\n\n"
                        "Evidence quotes:\n"
                        + "\n\n".join(f"- {quote}" for quote in evidence_quotes[:8])
                    )
                )
            ]
        )
        return round(float(result.unsupported_claim_fraction), 4)
    except Exception as exc:
        logger.warning("LLM judge failed, skipping hallucination judge metric: %s", exc)
        return None


def _document_matches_case(document: Document, case: EvalCase) -> bool:
    return _matching_identifier(document, case) is not None


def _matching_identifier(document: Document, case: EvalCase) -> str | None:
    metadata = document.metadata
    doc_ids = {_best_document_id(document)}
    normalized_doc_id = normalize_identifier(str(metadata.get("doc_id", "")).strip())
    if normalized_doc_id:
        doc_ids.add(normalized_doc_id)
    gold_doc_ids = {normalize_identifier(item) for item in case.gold_doc_ids if normalize_identifier(item)}
    doc_matches = doc_ids & gold_doc_ids
    if doc_matches:
        return sorted(doc_matches)[0]

    node_ids = {str(metadata.get("node_id", "")).strip().casefold()}
    node_ids.update(
        str(item).strip().casefold()
        for item in metadata.get("merged_node_ids", []) or []
        if str(item).strip()
    )
    gold_node_ids = {item.casefold() for item in case.gold_node_ids if item.strip()}
    node_matches = node_ids & gold_node_ids
    if node_matches:
        return sorted(node_matches)[0]
    return None


def _ranked_relevances(documents: list[Document], case: EvalCase) -> list[int]:
    relevances: list[int] = []
    seen_matches: set[str] = set()
    for document in documents:
        match = _matching_identifier(document, case)
        if match and match not in seen_matches:
            seen_matches.add(match)
            relevances.append(1)
        else:
            relevances.append(0)
    return relevances


def _best_document_id(document: Document) -> str:
    metadata = document.metadata
    source = normalize_identifier(str(metadata.get("source", "")).strip())
    if source:
        return source
    return normalize_identifier(str(metadata.get("doc_id", "")).strip())


def _section_path(metadata: dict[str, Any]) -> list[str]:
    section_path = metadata.get("section_path") or metadata.get("title_path") or []
    if isinstance(section_path, str):
        return [section_path]
    return [str(item) for item in section_path if str(item).strip()]


def _build_comparisons(variant_reports: dict[str, Any]) -> dict[str, Any]:
    baseline = variant_reports.get("baseline_flat", {})
    baseline_suites = baseline.get("suites", {})
    comparisons: dict[str, Any] = {}

    for variant_name, variant_report in variant_reports.items():
        if variant_name == "baseline_flat":
            continue
        suite_deltas: dict[str, Any] = {}
        for suite_name, suite_report in variant_report.get("suites", {}).items():
            metrics = suite_report.get("metrics", {})
            baseline_metrics = baseline_suites.get(suite_name, {}).get("metrics", {})
            deltas: dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, float) and isinstance(baseline_metrics.get(key), float):
                    deltas[key] = round(value - baseline_metrics[key], 4)
            suite_deltas[suite_name] = deltas
        comparisons[variant_name] = suite_deltas
    return comparisons


def _build_leaderboard(variant_reports: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    leaderboard: dict[str, list[dict[str, Any]]] = {}
    suite_names = sorted(
        {
            suite_name
            for variant in variant_reports.values()
            for suite_name in variant.get("suites", {})
        }
    )
    for suite_name in suite_names:
        rows: list[dict[str, Any]] = []
        for variant_name, variant_report in variant_reports.items():
            metrics = variant_report.get("suites", {}).get(suite_name, {}).get("metrics", {})
            if not metrics:
                continue
            comparable = _is_comparable_suite_result(suite_name, metrics)
            rows.append(
                {
                    "variant": variant_name,
                    "score": round(_suite_score(suite_name, metrics), 4),
                    "metrics": metrics,
                    "comparable": comparable,
                }
            )
        leaderboard[suite_name] = sorted(
            rows,
            key=lambda item: (item["comparable"], item["score"]),
            reverse=True,
        )
    return leaderboard


def _suite_score(suite_name: str, metrics: dict[str, Any]) -> float:
    if suite_name == "routing":
        return float(metrics.get("route_accuracy", 0.0))
    if suite_name == "retrieval":
        return (
            (0.5 * float(metrics.get("ndcg", 0.0)))
            + (0.3 * float(metrics.get("mrr", 0.0)))
            + (0.2 * float(metrics.get("recall_at_k", 0.0)))
            - (0.1 * float(metrics.get("redundancy_rate", 0.0)))
        )
    if suite_name == "answer":
        hallucination = metrics.get("hallucination_rate_llm_judge")
        if not isinstance(hallucination, float):
            hallucination = float(metrics.get("hallucination_rate_rule", 0.0))
        return (
            (0.35 * float(metrics.get("groundedness", 0.0)))
            + (0.3 * float(metrics.get("answer_completeness", 0.0)))
            + (0.25 * float(metrics.get("citation_precision", 0.0)))
            - (0.2 * float(hallucination))
        )
    return 0.0


def _is_comparable_suite_result(suite_name: str, metrics: dict[str, Any]) -> bool:
    if suite_name != "answer":
        return True
    return str(metrics.get("evaluation_mode", "")).strip() == "generative_grounded"


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_offline_extractive_answer(
    question: str,
    *,
    evidence_quotes: list[str],
    max_sentences: int = 3,
    max_chars: int = 420,
) -> str:
    question_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", question.casefold()))
    scored_sentences: list[tuple[int, float, str]] = []
    seen_sentences: set[str] = set()

    for quote in evidence_quotes:
        for raw_sentence in _SENTENCE_SPLIT_RE.split(quote):
            sentence = " ".join(raw_sentence.split()).strip()
            if len(sentence) < 20:
                continue
            normalized = sentence.casefold()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            sentence_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", normalized))
            overlap = len(question_terms & sentence_terms)
            score = overlap + min(len(sentence) / 200, 1.0)
            scored_sentences.append((overlap, score, sentence))

    if any(overlap > 0 for overlap, _, _ in scored_sentences):
        scored_sentences = [item for item in scored_sentences if item[0] > 0]

    scored_sentences.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[str] = []
    total_chars = 0
    for _, _, sentence in scored_sentences:
        projected = total_chars + len(sentence) + (1 if selected else 0)
        if projected > max_chars and selected:
            continue
        selected.append(sentence)
        total_chars = projected
        if len(selected) >= max_sentences:
            break

    if not selected and evidence_quotes:
        fallback = " ".join(evidence_quotes[0].split()).strip()
        return fallback[:max_chars].rstrip()
    return " ".join(selected).strip()


def _write_reports(
    report: dict[str, Any],
    output_dir: Path,
    timestamp: str,
    output_format: Literal["markdown", "json", "both"],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if output_format in {"json", "both"}:
        json_path = output_dir / f"eval_report_{report['suite']}_{timestamp}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["json"] = str(json_path)
    if output_format in {"markdown", "both"}:
        markdown_path = output_dir / f"eval_report_{report['suite']}_{timestamp}.md"
        markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
        artifacts["markdown"] = str(markdown_path)
    return artifacts


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Report",
        "",
        f"- Suite: `{report['suite']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Embedding mode: `{report['embedding_mode']}`",
        f"- LLM enabled: `{report['llm_enabled']}`",
        "",
    ]
    for variant_name, variant_report in report["variants"].items():
        lines.extend([f"## {variant_name}", "", variant_report["description"], ""])
        for suite_name, suite_report in variant_report.get("suites", {}).items():
            lines.extend([f"### {suite_name}", ""])
            lines.extend(_metrics_table(suite_report.get("metrics", {})))
            lines.append("")

    if report.get("leaderboard"):
        lines.extend(["## Leaderboard", ""])
        for suite_name, rows in report["leaderboard"].items():
            lines.extend(
                [
                    f"### {suite_name}",
                    "",
                    "| Rank | Variant | Score | Comparable |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for index, row in enumerate(rows, start=1):
                comparable = "yes" if row.get("comparable") else "fallback-only"
                lines.append(f"| {index} | {row['variant']} | {row['score']} | {comparable} |")
            if suite_name == "answer":
                lines.append("")
                lines.append(
                    "Answer leaderboard rows marked `fallback-only` were scored in offline extractive mode and should not be interpreted as fully comparable to generative grounded answering."
                )
            lines.append("")

    if report.get("comparisons"):
        lines.extend(["## Comparisons vs baseline_flat", ""])
        for variant_name, suite_deltas in report["comparisons"].items():
            lines.extend([f"### {variant_name}", ""])
            for suite_name, deltas in suite_deltas.items():
                lines.append(f"- `{suite_name}`: {json.dumps(deltas, ensure_ascii=False)}")
            lines.append("")

    if report.get("artifacts"):
        lines.extend(["## Artifacts", ""])
        for key, path in report["artifacts"].items():
            lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines).strip() + "\n"


def _metrics_table(metrics: dict[str, Any]) -> list[str]:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    return lines


def _has_llm_config(settings: AppSettings) -> bool:
    return bool(settings.llm_model and settings.llm_api_key and settings.llm_api_base)


def main() -> None:
    import sys

    import yaml

    from evals.m4_1_runner import run_from_config
    from evals.v2_runner import main as v2_main

    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 < len(sys.argv):
            candidate = Path(sys.argv[index + 1])
            if candidate.exists():
                payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and str(payload.get("kind", "")).startswith("m4_1_"):
                    run_from_config(candidate)
                    return
    v2_main()


if __name__ == "__main__":
    main()
