from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import get_direct_answer_prompt
from agent.states import GraphState
from llms.llm import get_llm_by_type


def direct_answer(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    user_input = (
        f"Conversation Summary:\n{conversation_summary}\n\n"
        if conversation_summary.strip()
        else ""
    ) + f"Latest User Message:\n{last_message.content}"

    llm = get_llm_by_type("direct_answer")
    response = llm.invoke(
        [
            SystemMessage(content=get_direct_answer_prompt()),
            HumanMessage(content=user_input),
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}
