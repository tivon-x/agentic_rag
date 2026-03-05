from typing import Literal
from langgraph.types import Send
from .graph_state import State

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"] | list[Send]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]