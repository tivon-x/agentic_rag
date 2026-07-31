"""LangGraph wrapper for the M4.1 bounded adaptive evidence loop."""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.adaptive import build_live_loop
from agent.edges import route_after_adaptive_decision
from agent.states import AdaptiveGraphState
from core.settings import AppSettings


def create_adaptive_graph(settings: AppSettings, retriever):
    """Create the non-persistent M4.1 graph without changing the fixed graph."""
    loop = build_live_loop(settings, retriever)
    builder = StateGraph(AdaptiveGraphState)
    builder.add_node("adaptive_decide", _adaptive_decide)
    builder.add_node("adaptive_direct", _adaptive_direct)
    builder.add_node("adaptive_refuse", _adaptive_refuse)
    builder.add_node("adaptive_fact", _adaptive_fact(loop))
    builder.add_edge(START, "adaptive_decide")
    builder.add_conditional_edges("adaptive_decide", route_after_adaptive_decision)
    builder.add_edge("adaptive_direct", END)
    builder.add_edge("adaptive_refuse", END)
    builder.add_edge("adaptive_fact", END)
    return builder.compile(checkpointer=InMemorySaver())


def _adaptive_decide(state: AdaptiveGraphState) -> dict:
    query = str(state["messages"][-1].content).strip()
    decision = _route(query)
    return {
        "runId": str(uuid4()),
        "query": query,
        "strategy": decision,
        "scopeIds": list(state.get("scopeIds", []))[:100],
        "budgets": {"max_rounds": 2, "max_tool_calls": 4, "max_evidence": 12, "max_context_tokens": 12_000},
    }


def _route(query: str) -> str:
    """Reserve pre-retrieval routing for non-factual safety boundaries only.

    Whether a paper question is fixed or adaptive is deliberately decided after
    the first B1 evidence pass inside ``AdaptiveEvidenceLoop``.
    """
    normalized = query.casefold().strip()
    if any(
        token in normalized
        for token in (
            "天气",
            "股价",
            "股票",
            "基金",
            "比特币",
            "机票",
            "餐厅",
            "诺贝尔",
            "利率",
            "alphafold",
            "库外",
            "gpu",
            "今天",
            "明天",
            "刚结束",
            "最新",
            "目前",
            "现在",
            "今日",
            "today",
            "weather",
            "stock",
        )
    ):
        return "refuse"
    if any(
        token in normalized
        for token in (
            "你好",
            "好的",
            "明白",
            "谢谢",
            "收到",
            "确认一下",
            "改成",
            "压缩成",
            "不用展开",
            "保持简短",
            "更简洁",
            "加粗",
            "重排",
            "引用格式",
            "引用列表",
            "用中文",
            "论文库回答",
            "不用继续",
            "保留这个结论",
        )
    ):
        return "direct"
    return "fact"


def _adaptive_direct(state: AdaptiveGraphState) -> dict:
    return {"messages": [AIMessage(content="好的。我会只按你要求的格式处理已有内容，不补充新的论文事实。")], "terminationReason": "direct_no_retrieval", "finalAnswer": {"answer": "", "claims": [], "limitations": ""}}


def _adaptive_refuse(state: AdaptiveGraphState) -> dict:
    return {"messages": [AIMessage(content="这个请求需要论文库外或实时信息，我不能基于当前论文库可靠回答。")], "terminationReason": "refuse_out_of_scope", "finalAnswer": {"answer": "", "claims": [], "limitations": "outside corpus"}}


def _adaptive_fact(loop):
    def execute(state: AdaptiveGraphState) -> dict:
        result = loop.run(
            str(state.get("query") or state["messages"][-1].content),
            scope=list(state.get("scopeIds", [])),
            cancelled=lambda: bool(state.get("cancelRequested", False)),
        )
        content = _render_result(result.final_answer, result.evidence)
        return {
            "messages": [AIMessage(content=content)],
            "strategy": result.strategy,
            "planItems": result.plan_items,
            "round": result.rounds,
            "candidateIds": [item["evidence_id"] for item in result.evidence][:30],
            "evidenceIds": [item["evidence_id"] for item in result.evidence],
            "coverage": result.coverage,
            "budgets": {"tool_calls": result.tool_calls, "context_tokens": result.context_tokens, "max_tool_calls": 4, "max_evidence": 12, "max_context_tokens": 12_000},
            "terminationReason": result.termination_reason,
            "finalAnswer": {**result.final_answer, "evidence": result.evidence, "latency_ms": result.latency_ms},
        }
    return execute


def _render_result(answer: dict, evidence: list[dict]) -> str:
    text = str(answer.get("answer", "")).strip() or "当前论文库中的证据不足以可靠回答这个问题。"
    claims = answer.get("claims", []) or []
    lines = [text]
    if claims:
        lines.extend(["", "## Claim evidence"])
        for claim in claims:
            ids = ", ".join(str(item) for item in claim.get("evidence_ids", []))
            lines.append(f"- {claim.get('claim', '')} [{ids}]")
    if answer.get("limitations"):
        lines.extend(["", "## Limitations", str(answer["limitations"])])
    if evidence:
        lines.extend(["", "## Evidence"])
        for item in evidence:
            lines.append(f"- {item['evidence_id']} · {item['source']} p.{item['page']}")
    return "\n".join(lines)
