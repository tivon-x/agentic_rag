from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import get_aggregation_prompt
from agent.schemas import GroundedAnswer
from agent.states import GraphState
from core.corpus_profile import build_answer_style_instruction
from core.rag_answer import render_grounded_answer
from llms.llm import get_llm_by_type


def aggregate_answers(state: GraphState):
    evidence_groups = state.get("evidenceGroups", [])
    if not evidence_groups:
        return {"messages": [AIMessage(content="No answers were generated.")]}

    packed_contexts = sorted(
        state.get("packedContexts", []),
        key=lambda item: str(item.get("subquery", "")),
    )
    retrieval_evidence = sorted(
        state.get("retrievalEvidence", []),
        key=lambda item: str(item.get("subquery", "")),
    )
    sorted_groups = sorted(
        evidence_groups,
        key=lambda item: str(item.get("subquery", "")),
    )

    payload = {
        "question": state.get("originalQuery", ""),
        "query_plan": state.get("queryPlan", {}),
        "packed_context": packed_contexts,
        "evidence_groups": sorted_groups,
        "retrieval_evidence": retrieval_evidence,
    }

    llm = get_llm_by_type("aggregate_answers")
    preferred_answer_style = build_answer_style_instruction(
        state.get("corpusProfileData", {})
    )
    try:
        structured_llm = llm.with_config(temperature=0).with_structured_output(
            GroundedAnswer
        )
        grounded_answer = structured_llm.invoke(
            [
                SystemMessage(
                    content=get_aggregation_prompt(
                        preferred_answer_style=preferred_answer_style,
                    )
                ),
                HumanMessage(content=str(payload)),
            ]
        ).model_dump()
    except Exception:
        collected_evidence: list[dict] = []
        for group in sorted_groups:
            for item in group.get("evidence", []):
                collected_evidence.append(item)

        unique_evidence: list[dict] = []
        seen_keys: set[str] = set()
        for item in collected_evidence:
            key = (
                f"{item.get('doc_id', '')}:{item.get('node_id', '')}:"
                f"{item.get('quote', '')}"
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_evidence.append(item)

        evidence_count = len(unique_evidence)
        answer = (
            "I couldn't find any relevant information in the available sources to answer your question."
            if evidence_count == 0
            else " ".join(
                item.get("quote", "").strip()
                for item in unique_evidence[:3]
                if item.get("quote", "").strip()
            ).strip()
        )
        grounded_answer = {
            "answer": answer
            or "I couldn't find any relevant information in the available sources to answer your question.",
            "reasoning_summary": (
                f"Synthesized from {len(sorted_groups)} evidence group(s) and "
                f"{evidence_count} unique evidence item(s)."
            ),
            "evidence": unique_evidence[:5],
            "confidence": min(0.95, 0.25 + (0.12 * evidence_count)),
            "limitations": (
                "Available evidence is limited to the retrieved passages."
                if evidence_count
                else "No structured evidence was captured from retrieval."
            ),
        }

    content = render_grounded_answer(grounded_answer)
    return {
        "groundedAnswer": grounded_answer,
        "messages": [AIMessage(content=content)],
    }
